from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


class DistributionRuleError(ValueError):
    """Raised when a user distribution rule cannot be parsed or safely applied."""


@dataclass(frozen=True)
class DistributionRule:
    """A single requested distribution rule.

    Supported examples:
        25% of accounts.account_type = checking
        20% of transactions.amount >= 5000
        50% of employees should have a salary of $10000 or greater
        25% of products should be associated with the category "Cat_2"
    """

    raw: str
    table: str
    column: str
    percent: float
    operator: str
    value: Any
    exact: bool = True

    @property
    def ratio(self) -> float:
        return self.percent / 100.0

    def target_count(self, total_rows: int) -> int:
        if total_rows <= 0:
            return 0
        return int(round(total_rows * self.ratio))

    def label(self) -> str:
        return f"{self.percent:g}% of {self.table}.{self.column} {self.operator} {self.value}"


@dataclass(frozen=True)
class CrossTableRule:
    """Cross-table conditional distribution rule resolved via schema relationships."""

    raw: str
    source_table: str
    target_table: str
    condition_column: str
    condition_operator: str
    condition_value: Any
    target_column: str
    target_operator: str
    target_value: Any
    percent: float
    relationship_path: Tuple[str, ...]
    hop_count: int
    exact: bool = True

    @property
    def rule_type(self) -> str:
        return "cross_table"

    @property
    def ratio(self) -> float:
        return self.percent / 100.0

    def target_count(self, scope_rows: int) -> int:
        if scope_rows <= 0:
            return 0
        return int(round(scope_rows * self.ratio))


@dataclass(frozen=True)
class WeightedRule:
    """Probability lift rule (multiplier), not an exact percentage target."""

    raw: str
    table: str
    scope_column: str
    scope_operator: str
    scope_value: Any
    target_column: str
    target_value: Any
    multiplier: float
    source_table: Optional[str] = None
    target_table: Optional[str] = None
    relationship_path: Tuple[str, ...] = ()
    hop_count: int = 0

    @property
    def rule_type(self) -> str:
        return "weighted"


RuleSpec = Union[DistributionRule, CrossTableRule, WeightedRule]


def percent_rules(rules: Sequence[RuleSpec]) -> List[DistributionRule]:
    return [rule for rule in rules if isinstance(rule, DistributionRule)]


def cross_table_rules(rules: Sequence[RuleSpec]) -> List[CrossTableRule]:
    return [rule for rule in rules if isinstance(rule, CrossTableRule)]


def weighted_rules(rules: Sequence[RuleSpec]) -> List[WeightedRule]:
    return [rule for rule in rules if isinstance(rule, WeightedRule)]


_ALLOWED_OPERATORS = {"=", "==", "!=", ">=", ">", "<=", "<"}
_EQUALITY_OPERATORS = {"=", "=="}
_COMPARISON_OPERATORS = {">=", ">", "<=", "<"}

_PROTECTED_TYPES = {
    "foreign_key",
    "uuid",
    "email",
    "phone",
    "url",
    "address",
    "bank_account",
    "account_number",
    "routing_number",
    "ssn",
    "aadhaar",
    "iban",
    "swift_bic",
    "ifsc_code",
    "credit_card",
}

_PROTECTED_NAME_PARTS = {
    "id",
    "uuid",
    "guid",
    "ssn",
    "aadhaar",
    "routing",
    "account_number",
    "accountnumber",
    "bank_account",
    "iban",
    "swift",
    "ifsc",
    "credit_card",
    "card_number",
    "email",
    "phone",
    "mobile",
    "address",
    "case_note",
    "note",
    "memo",
    "comment",
    "description",
    "narrative",
    "text",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_distribution_rules(raw: Union[str, Iterable[str], None]) -> List[RuleSpec]:
    """Parse user-entered distribution rules.

    Blank lines and comments are ignored.

    Expected main format:
        30% of customers.risk_tier = high
        20% of transactions.amount >= 5000
    """
    if raw is None:
        return []

    lines = raw.splitlines() if isinstance(raw, str) else list(raw)

    rules: List[RuleSpec] = []
    for line in lines:
        original = str(line or "").strip()
        if not original or original.startswith("#"):
            continue
        rules.append(_parse_one_rule(original))

    return rules


def apply_distribution_rules(
    tables: Mapping[str, pd.DataFrame],
    schema: Any,
    rules: Sequence[RuleSpec],
    *,
    seed: Optional[int] = None,
    exact: bool = True,
    inplace: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Apply distribution rules to generated tables.

    Design:
        - Applies rules after generation.
        - Keeps the logic generic.
        - Does not hardcode banking table names.
        - Does not modify protected IDs, FKs, PII, banking identifiers, or text-heavy fields.
        - Preserves natural variety in non-target rows instead of flattening them.
    """
    if not rules:
        return dict(tables) if inplace else {name: df.copy() for name, df in tables.items()}

    output: Dict[str, pd.DataFrame] = (
        dict(tables) if inplace else {name: df.copy() for name, df in tables.items()}
    )

    validate_distribution_rules(schema, rules, output)

    base_seed = int(seed if seed is not None else 0)
    pct_rules = percent_rules(rules)

    for rule_index, rule in enumerate(pct_rules):
        df = output[rule.table]

        if len(df) == 0:
            continue

        rng = np.random.default_rng(base_seed + rule_index + 9973)
        effective_exact = bool(exact and rule.exact)

        if rule.operator in _EQUALITY_OPERATORS:
            _apply_equality_rule(df, schema, rule, rng, effective_exact)
        elif rule.operator in _COMPARISON_OPERATORS:
            _apply_numeric_comparison_rule(df, rule, rng, effective_exact)
        elif rule.operator == "!=":
            _apply_not_equal_rule(df, schema, rule, rng, effective_exact)
        else:
            raise DistributionRuleError(f"Unsupported operator: {rule.operator}")

    for rule_index, rule in enumerate(cross_table_rules(rules)):
        rng = np.random.default_rng(base_seed + rule_index + 17713)
        _apply_cross_table_rule(output, schema, rule, rng, exact=bool(exact and rule.exact))

    for rule_index, rule in enumerate(weighted_rules(rules)):
        rng = np.random.default_rng(base_seed + rule_index + 27109)
        _apply_weighted_rule(output, schema, rule, rng)

    return output


def evaluate_distribution_rules(
    tables: Mapping[str, pd.DataFrame],
    rules: Sequence[RuleSpec],
    *,
    tolerance_percent: float = 0.25,
    schema: Any = None,
) -> List[Dict[str, Any]]:
    """Return dashboard evidence that generated data satisfies requested rules."""
    records: List[Dict[str, Any]] = []

    for rule in rules:
        if isinstance(rule, CrossTableRule):
            records.append(
                _evaluate_cross_table_rule(tables, rule, schema=schema, tolerance_percent=tolerance_percent)
            )
            continue
        if isinstance(rule, WeightedRule):
            records.append(_evaluate_weighted_rule(tables, rule))
            continue
        if not isinstance(rule, DistributionRule):
            continue
        if rule.table not in tables:
            records.append(_missing_record(rule, f"Table '{rule.table}' not found"))
            continue

        df = tables[rule.table]

        if rule.column not in df.columns:
            records.append(_missing_record(rule, f"Column '{rule.column}' not found"))
            continue

        total = int(len(df))
        target_rows = rule.target_count(total)

        mask = _condition_mask(df[rule.column], rule.operator, rule.value)
        actual_rows = int(mask.sum())
        actual_percent = 0.0 if total == 0 else (actual_rows / total) * 100.0
        difference = actual_percent - rule.percent

        passed = abs(difference) <= tolerance_percent or actual_rows == target_rows

        records.append(
            {
                "rule": rule.raw,
                "table": rule.table,
                "column": rule.column,
                "condition": f"{rule.column} {rule.operator} {rule.value}",
                "target_percent": round(float(rule.percent), 3),
                "actual_percent": round(float(actual_percent), 3),
                "target_rows": target_rows,
                "actual_rows": actual_rows,
                "difference_pp": round(float(difference), 3),
                "passed": bool(passed),
                "status": "Passed" if passed else "Failed",
            }
        )

    return records


def validate_distribution_rules(
    schema: Any,
    rules: Sequence[RuleSpec],
    tables: Optional[Mapping[str, pd.DataFrame]] = None,
) -> None:
    """Validate rules against schema metadata and safety guardrails."""
    for rule in rules:
        if isinstance(rule, CrossTableRule):
            _validate_cross_table_rule(schema, rule, tables)
            continue
        if isinstance(rule, WeightedRule):
            _validate_weighted_rule(schema, rule, tables)
            continue
        if not isinstance(rule, DistributionRule):
            continue
        if not (0 <= rule.percent <= 100):
            raise DistributionRuleError(f"Rule percent must be between 0 and 100: {rule.raw}")

        if rule.operator not in _ALLOWED_OPERATORS:
            raise DistributionRuleError(f"Unsupported operator '{rule.operator}' in rule: {rule.raw}")

        column_spec = _get_schema_column(schema, rule.table, rule.column)

        if column_spec is None:
            raise DistributionRuleError(
                f"Rule references unknown column '{rule.table}.{rule.column}': {rule.raw}"
            )

        if _is_protected_column(schema, rule.table, rule.column, column_spec):
            raise DistributionRuleError(
                f"Distribution rules cannot modify protected/private/key column "
                f"'{rule.table}.{rule.column}'. Choose a safe business column instead."
            )

        col_type = str(getattr(column_spec, "type", "") or "").lower()

        if rule.operator in _COMPARISON_OPERATORS:
            known_numeric_types = {"int", "float", "decimal", "money", "currency"}

            if col_type not in known_numeric_types:
                if (
                    tables is None
                    or rule.table not in tables
                    or rule.column not in tables[rule.table].columns
                    or not pd.api.types.is_numeric_dtype(tables[rule.table][rule.column])
                ):
                    raise DistributionRuleError(
                        f"Comparison rule requires a numeric column, got '{col_type}' "
                        f"for {rule.table}.{rule.column}"
                    )

        if tables is not None:
            if rule.table not in tables:
                raise DistributionRuleError(f"Generated table '{rule.table}' not found for rule: {rule.raw}")

            if rule.column not in tables[rule.table].columns:
                raise DistributionRuleError(
                    f"Generated column '{rule.table}.{rule.column}' not found for rule: {rule.raw}"
                )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_one_rule(raw: str) -> RuleSpec:
    normalized = _normalize_rule_text(raw)

    weighted = _try_parse_weighted_rule(raw, normalized)
    if weighted is not None:
        return weighted

    cross = _try_parse_cross_table_rule(raw, normalized)
    if cross is not None:
        return cross

    # Canonical form:
    # 25% of accounts.account_type = checking
    m = re.match(
        r"^(?P<pct>\d+(?:\.\d+)?)\s*%\s+of\s+(?:the\s+)?"
        r"(?P<table>[A-Za-z_][\w]*)\.(?P<column>[A-Za-z_][\w]*)\s*"
        r"(?P<op>>=|<=|==|!=|=|>|<)\s*(?P<value>.+?)\s*$",
        normalized,
        flags=re.IGNORECASE,
    )
    if m:
        return _rule_from_match(raw, m)

    # 30% of accounts where account_type = savings
    m = re.match(
        r"^(?P<pct>\d+(?:\.\d+)?)\s*%\s+of\s+(?:the\s+)?(?P<table>[A-Za-z_][\w]*)\s+"
        r"where\s+(?P<column>[A-Za-z_][\w]*)\s*(?P<op>>=|<=|==|!=|=|>|<)\s*"
        r"(?P<value>.+?)\s*$",
        normalized,
        flags=re.IGNORECASE,
    )
    if m:
        return _rule_from_match(raw, m)

    # 50% of employees should have a salary of $10000 or greater
    m = re.match(
        r"^(?P<pct>\d+(?:\.\d+)?)\s*%\s+of\s+(?:the\s+)?(?P<table>[A-Za-z_][\w]*)\s+"
        r"should\s+have\s+(?:a\s+|an\s+)?(?P<column>[A-Za-z_][\w]*)\s+"
        r"(?:of\s+)?(?P<value>.+?)\s+or\s+"
        r"(?P<direction>greater|more|higher|less|lower)\s*$",
        normalized,
        flags=re.IGNORECASE,
    )
    if m:
        direction = m.group("direction").lower()
        op = ">=" if direction in {"greater", "more", "higher"} else "<="

        return DistributionRule(
            raw=raw,
            table=m.group("table"),
            column=m.group("column"),
            percent=float(m.group("pct")),
            operator=op,
            value=_parse_value(m.group("value")),
        )

    # 25% of products should be associated with the category "Cat_2"
    m = re.match(
        r"^(?P<pct>\d+(?:\.\d+)?)\s*%\s+of\s+(?:the\s+)?(?P<table>[A-Za-z_][\w]*)\s+"
        r"should\s+be\s+associated\s+with\s+(?:the\s+)?(?P<column>[A-Za-z_][\w]*)\s+"
        r"(?P<value>.+?)\s*$",
        normalized,
        flags=re.IGNORECASE,
    )
    if m:
        return DistributionRule(
            raw=raw,
            table=m.group("table"),
            column=m.group("column"),
            percent=float(m.group("pct")),
            operator="=",
            value=_parse_value(m.group("value")),
        )

    # 25% of products should have category equal to Cat_2
    m = re.match(
        r"^(?P<pct>\d+(?:\.\d+)?)\s*%\s+of\s+(?:the\s+)?(?P<table>[A-Za-z_][\w]*)\s+"
        r"should\s+have\s+(?:a\s+|an\s+)?(?P<column>[A-Za-z_][\w]*)\s+"
        r"(?:(?P<op_text>equal\s+to|equals|=|==|not\s+equal\s+to|!=)\s+)?"
        r"(?P<value>.+?)\s*$",
        normalized,
        flags=re.IGNORECASE,
    )
    if m:
        op_text = (m.group("op_text") or "=").lower().strip()
        op = "!=" if "not" in op_text or op_text == "!=" else "="

        return DistributionRule(
            raw=raw,
            table=m.group("table"),
            column=m.group("column"),
            percent=float(m.group("pct")),
            operator=op,
            value=_parse_value(m.group("value")),
        )

    raise DistributionRuleError(
        "Could not parse distribution rule. Use a format like "
        "'25% of accounts.account_type = checking' or "
        "'50% of employees.salary >= 10000'. "
        f"Got: {raw!r}"
    )


def _rule_from_match(raw: str, match: re.Match[str]) -> DistributionRule:
    return DistributionRule(
        raw=raw,
        table=match.group("table"),
        column=match.group("column"),
        percent=float(match.group("pct")),
        operator=match.group("op"),
        value=_parse_value(match.group("value")),
    )


def _normalize_rule_text(text: str) -> str:
    value = str(text).strip()

    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "≥": ">=",
        "≤": "<=",
        "greater than or equal to": ">=",
        "more than or equal to": ">=",
        "less than or equal to": "<=",
        "greater than": ">",
        "more than": ">",
        "less than": "<",
        "equal to": "=",
        "equals": "=",
    }

    for old, new in replacements.items():
        value = re.sub(re.escape(old), new, value, flags=re.IGNORECASE)

    return re.sub(r"\s+", " ", value).strip()


def _parse_value(raw: str) -> Any:
    value = str(raw).strip().rstrip(".")
    value = value.strip().strip('"').strip("'")
    value = value.replace("$", "").replace(",", "")

    lowered = value.lower()

    if lowered in {"true", "yes"}:
        return True

    if lowered in {"false", "no"}:
        return False

    if lowered in {"null", "none"}:
        return None

    if re.fullmatch(r"[-+]?\d+", value):
        try:
            return int(value)
        except ValueError:
            pass

    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)", value):
        try:
            return float(value)
        except ValueError:
            pass

    return value


# ---------------------------------------------------------------------------
# Rule application
# ---------------------------------------------------------------------------


def _apply_equality_rule(
    df: pd.DataFrame,
    schema: Any,
    rule: DistributionRule,
    rng: np.random.Generator,
    exact: bool,
) -> None:
    target_value = _coerce_for_series(rule.value, df[rule.column])
    current_mask = _condition_mask(df[rule.column], rule.operator, target_value)

    if not exact:
        target_n = rule.target_count(len(df))
        needed = max(0, target_n - int(current_mask.sum()))

        if needed <= 0:
            return

        false_index = df.loc[~current_mask].index.to_numpy()
        chosen = _sample_index(false_index, needed, rng)
        df.loc[chosen, rule.column] = target_value
        return

    target_n = rule.target_count(len(df))
    target_index = _choose_exact_target_index(df.index, current_mask, target_n, rng)

    selected_mask = pd.Series(False, index=df.index)
    selected_mask.loc[target_index] = True

    # Selected rows must satisfy the rule.
    selected_not_matching = selected_mask & ~current_mask
    df.loc[selected_not_matching, rule.column] = target_value

    # Non-selected rows must not satisfy the rule, but preserve variety.
    updated_mask = _condition_mask(df[rule.column], rule.operator, target_value)
    excess_mask = ~selected_mask & updated_mask
    excess_index = df.loc[excess_mask].index

    if len(excess_index) > 0:
        replacements = _alternate_values_for_indices(
            series=df[rule.column],
            schema=schema,
            rule=rule,
            blocked_value=target_value,
            indices=excess_index,
            rng=rng,
        )
        df.loc[excess_index, rule.column] = replacements


def _apply_not_equal_rule(
    df: pd.DataFrame,
    schema: Any,
    rule: DistributionRule,
    rng: np.random.Generator,
    exact: bool,
) -> None:
    blocked_value = _coerce_for_series(rule.value, df[rule.column])
    current_mask = _condition_mask(df[rule.column], "!=", blocked_value)

    if not exact:
        target_n = rule.target_count(len(df))
        needed = max(0, target_n - int(current_mask.sum()))

        if needed <= 0:
            return

        false_index = df.loc[~current_mask].index.to_numpy()
        chosen = _sample_index(false_index, needed, rng)

        replacements = _alternate_values_for_indices(
            series=df[rule.column],
            schema=schema,
            rule=rule,
            blocked_value=blocked_value,
            indices=chosen,
            rng=rng,
        )
        df.loc[chosen, rule.column] = replacements
        return

    target_n = rule.target_count(len(df))
    target_index = _choose_exact_target_index(df.index, current_mask, target_n, rng)

    selected_mask = pd.Series(False, index=df.index)
    selected_mask.loc[target_index] = True

    # Selected rows must be different from blocked_value.
    selected_invalid = selected_mask & ~current_mask
    selected_invalid_index = df.loc[selected_invalid].index

    if len(selected_invalid_index) > 0:
        replacements = _alternate_values_for_indices(
            series=df[rule.column],
            schema=schema,
            rule=rule,
            blocked_value=blocked_value,
            indices=selected_invalid_index,
            rng=rng,
        )
        df.loc[selected_invalid_index, rule.column] = replacements

    # Non-selected rows should equal blocked_value for exact percentage.
    updated_mask = _condition_mask(df[rule.column], "!=", blocked_value)
    excess_mask = ~selected_mask & updated_mask
    df.loc[excess_mask, rule.column] = blocked_value


def _apply_numeric_comparison_rule(
    df: pd.DataFrame,
    rule: DistributionRule,
    rng: np.random.Generator,
    exact: bool,
) -> None:
    threshold = _numeric_value(rule.value, rule.raw)
    current_mask = _condition_mask(df[rule.column], rule.operator, threshold)

    if not exact:
        target_n = rule.target_count(len(df))
        needed = max(0, target_n - int(current_mask.sum()))

        if needed <= 0:
            return

        false_index = df.loc[~current_mask].index.to_numpy()
        chosen = _sample_index(false_index, needed, rng)

        replacements = _varied_numeric_values(
            series=df[rule.column],
            operator=rule.operator,
            threshold=threshold,
            count=len(chosen),
            satisfy=True,
            rng=rng,
        )
        df.loc[chosen, rule.column] = replacements
        return

    target_n = rule.target_count(len(df))
    target_index = _choose_exact_target_index(df.index, current_mask, target_n, rng)

    selected_mask = pd.Series(False, index=df.index)
    selected_mask.loc[target_index] = True

    # Selected rows must satisfy the numeric condition.
    selected_invalid = selected_mask & ~current_mask
    selected_invalid_index = df.loc[selected_invalid].index

    if len(selected_invalid_index) > 0:
        replacements = _varied_numeric_values(
            series=df[rule.column],
            operator=rule.operator,
            threshold=threshold,
            count=len(selected_invalid_index),
            satisfy=True,
            rng=rng,
        )
        df.loc[selected_invalid_index, rule.column] = replacements

    # Non-selected rows must fail the numeric condition, but should be varied.
    updated_mask = _condition_mask(df[rule.column], rule.operator, threshold)
    excess_mask = ~selected_mask & updated_mask
    excess_index = df.loc[excess_mask].index

    if len(excess_index) > 0:
        replacements = _varied_numeric_values(
            series=df[rule.column],
            operator=rule.operator,
            threshold=threshold,
            count=len(excess_index),
            satisfy=False,
            rng=rng,
        )
        df.loc[excess_index, rule.column] = replacements


# ---------------------------------------------------------------------------
# Selection and replacement helpers
# ---------------------------------------------------------------------------


def _choose_exact_target_index(
    index: pd.Index,
    current_mask: pd.Series,
    target_n: int,
    rng: np.random.Generator,
) -> pd.Index:
    """Pick exactly target_n rows that should satisfy a rule.

    It prefers rows that already satisfy the condition to minimize distortion.
    """
    target_n = max(0, min(int(target_n), len(index)))

    if target_n == 0:
        return pd.Index([])

    true_index = current_mask[current_mask].index.to_numpy()
    false_index = current_mask[~current_mask].index.to_numpy()

    if len(true_index) >= target_n:
        chosen = _sample_index(true_index, target_n, rng)
        return pd.Index(chosen)

    needed = target_n - len(true_index)
    extra = _sample_index(false_index, needed, rng)
    combined = list(true_index) + list(extra)
    return pd.Index(combined)


def _sample_index(values: Union[np.ndarray, Sequence[Any]], count: int, rng: np.random.Generator) -> List[Any]:
    values_array = np.asarray(list(values), dtype=object)
    count = max(0, min(int(count), len(values_array)))

    if count == 0:
        return []

    if count == len(values_array):
        return values_array.tolist()

    return rng.choice(values_array, size=count, replace=False).tolist()


def _alternate_values_for_indices(
    *,
    series: pd.Series,
    schema: Any,
    rule: DistributionRule,
    blocked_value: Any,
    indices: Sequence[Any],
    rng: np.random.Generator,
) -> List[Any]:
    """Return varied replacement values that are not equal to blocked_value."""
    count = len(indices)

    if count <= 0:
        return []

    candidates = _candidate_alternate_values(series, schema, rule, blocked_value)

    if not candidates:
        candidates = ["Other" if str(blocked_value) != "Other" else "Not Other"]

    choices = rng.choice(np.asarray(candidates, dtype=object), size=count, replace=True).tolist()
    return [_coerce_for_series(value, series) for value in choices]


def _candidate_alternate_values(
    series: pd.Series,
    schema: Any,
    rule: DistributionRule,
    blocked_value: Any,
) -> List[Any]:
    column_spec = _get_schema_column(schema, rule.table, rule.column)
    params = getattr(column_spec, "distribution_params", {}) or {}

    candidates: List[Any] = []

    for value in list(params.get("choices") or []):
        if str(value) != str(blocked_value):
            candidates.append(value)

    existing = [
        value
        for value in series.dropna().unique().tolist()
        if str(value) != str(blocked_value)
    ]

    for value in existing:
        if str(value) != str(blocked_value) and value not in candidates:
            candidates.append(value)

    if not candidates:
        if isinstance(blocked_value, bool):
            candidates.append(not blocked_value)
        elif isinstance(blocked_value, (int, float)) and not isinstance(blocked_value, bool):
            candidates.append(blocked_value + 1)
        else:
            candidates.append("Other" if str(blocked_value) != "Other" else "Not Other")

    return candidates


def _varied_numeric_values(
    *,
    series: pd.Series,
    operator: str,
    threshold: float,
    count: int,
    satisfy: bool,
    rng: np.random.Generator,
) -> List[Union[int, float]]:
    """Generate varied numeric values that either satisfy or fail a threshold rule.

    This fixes the previous flattening problem where all non-target rows became
    4999.99 or all category rows became one fallback value.
    """
    count = max(0, int(count))

    if count == 0:
        return []

    numeric = pd.to_numeric(series, errors="coerce").dropna()

    if numeric.empty:
        observed_min = threshold - 100.0
        observed_max = threshold + 100.0
    else:
        observed_min = float(numeric.min())
        observed_max = float(numeric.max())

    is_integer = pd.api.types.is_integer_dtype(series)
    decimals = 0 if is_integer else _infer_decimal_places(series)
    step = 1.0 if is_integer else 10 ** (-decimals)

    low, high = _numeric_range_for_condition(
        operator=operator,
        threshold=threshold,
        observed_min=observed_min,
        observed_max=observed_max,
        satisfy=satisfy,
        step=step,
    )

    if high < low:
        low, high = high, low

    if abs(high - low) < step:
        values = np.full(count, low)
    else:
        values = rng.uniform(low, high, size=count)

    if is_integer:
        return [int(round(v)) for v in values]

    rounded = [round(float(v), decimals) for v in values]

    # Final safety correction in case rounding crosses the threshold.
    corrected = []
    for value in rounded:
        if _single_value_matches(value, operator, threshold) != satisfy:
            value = _boundary_numeric_value(operator, threshold, satisfy=satisfy, decimals=decimals)
        corrected.append(value)

    return corrected


def _numeric_range_for_condition(
    *,
    operator: str,
    threshold: float,
    observed_min: float,
    observed_max: float,
    satisfy: bool,
    step: float,
) -> tuple[float, float]:
    spread = max(abs(observed_max - observed_min), abs(threshold) * 0.5, 100.0)

    natural_min = min(observed_min, threshold - spread)
    natural_max = max(observed_max, threshold + spread)

    if operator == ">=":
        if satisfy:
            return threshold, max(natural_max, threshold + spread)
        return min(natural_min, threshold - spread), threshold - step

    if operator == ">":
        if satisfy:
            return threshold + step, max(natural_max, threshold + spread)
        return min(natural_min, threshold - spread), threshold

    if operator == "<=":
        if satisfy:
            return min(natural_min, threshold - spread), threshold
        return threshold + step, max(natural_max, threshold + spread)

    if operator == "<":
        if satisfy:
            return min(natural_min, threshold - spread), threshold - step
        return threshold, max(natural_max, threshold + spread)

    raise DistributionRuleError(f"Unsupported numeric operator: {operator}")


def _boundary_numeric_value(
    operator: str,
    threshold: float,
    *,
    satisfy: bool,
    decimals: int,
) -> float:
    step = 10 ** (-decimals) if decimals > 0 else 1.0

    if operator == ">=":
        value = threshold if satisfy else threshold - step
    elif operator == ">":
        value = threshold + step if satisfy else threshold
    elif operator == "<=":
        value = threshold if satisfy else threshold + step
    elif operator == "<":
        value = threshold - step if satisfy else threshold
    else:
        value = threshold

    return round(float(value), decimals)


def _infer_decimal_places(series: pd.Series, default: int = 2) -> int:
    sample = pd.to_numeric(series, errors="coerce").dropna().head(50)

    if sample.empty:
        return default

    max_decimals = 0

    for value in sample:
        text = f"{float(value):.8f}".rstrip("0").rstrip(".")
        if "." in text:
            max_decimals = max(max_decimals, len(text.split(".", 1)[1]))

    return min(max(max_decimals, default), 4)


def _single_value_matches(value: Any, operator: str, threshold: float) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False

    if operator == ">=":
        return numeric >= threshold
    if operator == ">":
        return numeric > threshold
    if operator == "<=":
        return numeric <= threshold
    if operator == "<":
        return numeric < threshold

    return False


# ---------------------------------------------------------------------------
# Condition and schema helpers
# ---------------------------------------------------------------------------


def _condition_mask(series: pd.Series, operator: str, value: Any) -> pd.Series:
    if operator in _COMPARISON_OPERATORS:
        numeric = pd.to_numeric(series, errors="coerce")
        threshold = float(value)

        if operator == ">=":
            return numeric >= threshold
        if operator == ">":
            return numeric > threshold
        if operator == "<=":
            return numeric <= threshold
        if operator == "<":
            return numeric < threshold

    comparable = _coerce_for_series(value, series)

    if operator in _EQUALITY_OPERATORS:
        if isinstance(comparable, str):
            return series.astype(str) == str(comparable)
        return series == comparable

    if operator == "!=":
        if isinstance(comparable, str):
            return series.astype(str) != str(comparable)
        return series != comparable

    raise DistributionRuleError(f"Unsupported operator: {operator}")


def _coerce_for_series(value: Any, series: pd.Series) -> Any:
    if value is None:
        return value

    if pd.api.types.is_integer_dtype(series):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return value

    if pd.api.types.is_float_dtype(series):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value

    if pd.api.types.is_bool_dtype(series):
        if isinstance(value, bool):
            return value
        return str(value).lower() in {"true", "yes", "1"}

    return str(value)


def _numeric_value(value: Any, raw_rule: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise DistributionRuleError(f"Numeric rule has non-numeric value in rule: {raw_rule}") from exc


def _get_schema_column(schema: Any, table_name: str, column_name: str) -> Any:
    if schema is None:
        return None

    if hasattr(schema, "get_columns"):
        columns = schema.get_columns(table_name)
    else:
        columns = getattr(schema, "columns", {}).get(table_name, [])

    for column in columns or []:
        if getattr(column, "name", None) == column_name:
            return column

    return None


def _is_protected_column(schema: Any, table_name: str, column_name: str, column_spec: Any) -> bool:
    name = column_name.lower()
    col_type = str(getattr(column_spec, "type", "") or "").lower()

    if bool(getattr(column_spec, "unique", False)):
        return True

    if col_type in _PROTECTED_TYPES:
        return True

    if _is_relationship_key(schema, table_name, column_name):
        return True

    if name == "id" or name.endswith("_id") or (name.endswith("id") and len(name) <= 12):
        return True

    compact = name.replace("_", "")

    for part in _PROTECTED_NAME_PARTS:
        part_compact = part.replace("_", "")

        if name == part or compact == part_compact or part in name:
            return True

    return False


def _is_relationship_key(schema: Any, table_name: str, column_name: str) -> bool:
    for rel in getattr(schema, "relationships", []) or []:
        if getattr(rel, "parent_table", None) == table_name and getattr(rel, "parent_key", None) == column_name:
            return True

        if getattr(rel, "child_table", None) == table_name and getattr(rel, "child_key", None) == column_name:
            return True

    return False


def _try_parse_weighted_rule(raw: str, normalized: str) -> Optional[WeightedRule]:
    patterns = [
        r"^(?:in\s+(?P<table>[A-Za-z_][\w]*),\s*)?rows\s+(?:with|where)\s+(?:(?P<scope_table>[A-Za-z_][\w]*)\.)?"
        r"(?P<scope_col>[A-Za-z_][\w]*)\s*(?P<scope_op>>=|<=|==|!=|=|>|<)\s*(?P<scope_val>.+?)\s+"
        r"should\s+be\s+(?P<mult>\d+(?:\.\d+)?)\s*x\s+more\s+likely\s+to\s+have\s+"
        r"(?P<target_col>[A-Za-z_][\w]*)\s*(?:=|==)\s*(?P<target_val>.+?)\s*$",
        r"^records\s+where\s+(?:(?P<scope_table>[A-Za-z_][\w]*)\.)?"
        r"(?P<scope_col>[A-Za-z_][\w]*)\s*(?P<scope_op>>=|<=|==|!=|=|>|<)\s*(?P<scope_val>.+?)\s+"
        r"should\s+be\s+(?P<mult>\d+(?:\.\d+)?)\s*x\s+more\s+likely\s+to\s+have\s+"
        r"(?P<target_col>[A-Za-z_][\w]*)\s*(?:=|==)\s*(?P<target_val>.+?)\s*$",
        r"^for\s+rows\s+where\s+(?:(?P<scope_table>[A-Za-z_][\w]*)\.)?"
        r"(?P<scope_col>[A-Za-z_][\w]*)\s*(?P<scope_op>>=|<=|==|!=|=|>|<)\s*(?P<scope_val>.+?),\s+"
        r"make\s+(?P<target_col>[A-Za-z_][\w]*)\s*(?:=|==)\s*(?P<target_val>.+?)\s+"
        r"(?P<mult>\d+(?:\.\d+)?)\s*x\s+more\s+likely\s*$",
    ]
    for pattern in patterns:
        match = re.match(pattern, normalized, flags=re.IGNORECASE)
        if not match:
            continue
        table = match.groupdict().get("table") or match.groupdict().get("scope_table") or "source_table"
        return WeightedRule(
            raw=raw,
            table=str(table),
            scope_column=match.group("scope_col"),
            scope_operator=match.group("scope_op"),
            scope_value=_parse_value(match.group("scope_val")),
            target_column=match.group("target_col"),
            target_value=_parse_value(match.group("target_val")),
            multiplier=float(match.group("mult")),
        )
    return None


def _try_parse_cross_table_rule(raw: str, normalized: str) -> Optional[CrossTableRule]:
    patterns = [
        r"^for\s+(?P<target_table>[A-Za-z_][\w]*)\s+linked\s+to\s+(?P<source_table>[A-Za-z_][\w]*)\s+where\s+"
        r"(?:(?P<cond_table>[A-Za-z_][\w]*)\.)?(?P<cond_col>[A-Za-z_][\w]*)\s*"
        r"(?P<cond_op>>=|<=|==|!=|=|>|<)\s*(?P<cond_val>.+?),\s+"
        r"make\s+(?P<pct>\d+(?:\.\d+)?)\s*%\s+of\s+(?:(?P<target_table2>[A-Za-z_][\w]*)\.)?"
        r"(?P<target_col>[A-Za-z_][\w]*)\s*(?P<target_op>>=|<=|==|!=|=|>|<)\s*(?P<target_val>.+?)\s*$",
        r"^for\s+(?P<target_table>[A-Za-z_][\w]*)\s+rows\s+linked\s+to\s+(?P<source_table>[A-Za-z_][\w]*)\s+rows\s+where\s+"
        r"(?:(?P<cond_table>[A-Za-z_][\w]*)\.)?(?P<cond_col>[A-Za-z_][\w]*)\s*"
        r"(?P<cond_op>>=|<=|==|!=|=|>|<)\s*(?P<cond_val>.+?),\s+"
        r"make\s+(?P<pct>\d+(?:\.\d+)?)\s*%\s+of\s+(?:(?P<target_table2>[A-Za-z_][\w]*)\.)?"
        r"(?P<target_col>[A-Za-z_][\w]*)\s*(?P<target_op>>=|<=|==|!=|=|>|<)\s*(?P<target_val>.+?)\s*$",
    ]
    for pattern in patterns:
        match = re.match(pattern, normalized, flags=re.IGNORECASE)
        if not match:
            continue
        source_table = match.group("source_table")
        target_table = match.group("target_table2") or match.group("target_table")
        cond_table = match.group("cond_table") or source_table
        if cond_table != source_table:
            raise DistributionRuleError(
                f"Ambiguous parent condition table '{cond_table}' for source '{source_table}': {raw!r}"
            )
        return CrossTableRule(
            raw=raw,
            source_table=source_table,
            target_table=target_table,
            condition_column=match.group("cond_col"),
            condition_operator=match.group("cond_op"),
            condition_value=_parse_value(match.group("cond_val")),
            target_column=match.group("target_col"),
            target_operator=match.group("target_op"),
            target_value=_parse_value(match.group("target_val")),
            percent=float(match.group("pct")),
            relationship_path=(),
            hop_count=0,
        )
    return None


def resolve_relationship_path(schema: Any, source_table: str, target_table: str) -> Tuple[Tuple[str, ...], int]:
    """Resolve a parent→child relationship path between two tables."""
    if source_table == target_table:
        raise DistributionRuleError(
            f"Cross-table rule requires distinct source and target tables; both are '{source_table}'."
        )
    relationships = list(getattr(schema, "relationships", []) or [])
    for rel in relationships:
        if rel.parent_table == source_table and rel.child_table == target_table:
            return ((f"{source_table}->{target_table}",), 1)
    for rel1 in relationships:
        if rel1.parent_table != source_table:
            continue
        mid = rel1.child_table
        for rel2 in relationships:
            if rel2.parent_table == mid and rel2.child_table == target_table:
                return (
                    (
                        f"{source_table}->{mid}",
                        f"{mid}->{target_table}",
                    ),
                    2,
                )
    raise DistributionRuleError(
        f"No relationship path from '{source_table}' to '{target_table}'. "
        "Define SchemaConfig.relationships or qualify tables unambiguously."
    )


def _relationship_for_path(schema: Any, source_table: str, target_table: str) -> Any:
    for rel in getattr(schema, "relationships", []) or []:
        if rel.parent_table == source_table and rel.child_table == target_table:
            return rel
    for rel1 in getattr(schema, "relationships", []) or []:
        if rel1.parent_table != source_table:
            continue
        mid = rel1.child_table
        for rel2 in getattr(schema, "relationships", []) or []:
            if rel2.parent_table == mid and rel2.child_table == target_table:
                return rel2
    raise DistributionRuleError(f"Missing relationship path {source_table}->{target_table}")


def _validate_cross_table_rule(
    schema: Any,
    rule: CrossTableRule,
    tables: Optional[Mapping[str, pd.DataFrame]],
) -> None:
    if not (0 <= rule.percent <= 100):
        raise DistributionRuleError(f"Rule percent must be between 0 and 100: {rule.raw}")
    resolve_relationship_path(schema, rule.source_table, rule.target_table)
    column_spec = _get_schema_column(schema, rule.target_table, rule.target_column)
    if column_spec is None:
        raise DistributionRuleError(
            f"Cross-table rule references unknown target column '{rule.target_table}.{rule.target_column}'"
        )
    if _is_protected_column(schema, rule.target_table, rule.target_column, column_spec):
        raise DistributionRuleError(
            f"Cross-table rule cannot modify protected column '{rule.target_table}.{rule.target_column}'"
        )
    if tables is not None:
        if rule.source_table not in tables or rule.target_table not in tables:
            raise DistributionRuleError(f"Generated tables missing for cross-table rule: {rule.raw}")


def _validate_weighted_rule(
    schema: Any,
    rule: WeightedRule,
    tables: Optional[Mapping[str, pd.DataFrame]],
) -> None:
    table_name = rule.table
    if tables is not None and table_name not in tables and rule.source_table and rule.target_table:
        table_name = rule.target_table or table_name
    column_spec = _get_schema_column(schema, table_name, rule.target_column)
    if column_spec is None and hasattr(schema, "columns"):
        profile_cols = getattr(schema, "columns", {})
        if isinstance(profile_cols, dict) and table_name in profile_cols:
            column_spec = _get_schema_column(schema, table_name, rule.target_column)
    if column_spec is None and (tables is None or table_name not in tables):
        raise DistributionRuleError(
            f"Weighted rule references unknown column '{table_name}.{rule.target_column}'"
        )
    if column_spec is not None and _is_protected_column(schema, table_name, rule.target_column, column_spec):
        raise DistributionRuleError(
            f"Weighted rule cannot modify protected column '{table_name}.{rule.target_column}'"
        )


def _child_scope_mask(
    tables: Mapping[str, pd.DataFrame],
    schema: Any,
    rule: CrossTableRule,
) -> pd.Series:
    rel = _relationship_for_path(schema, rule.source_table, rule.target_table)
    child_df = tables[rule.target_table]
    parent_df = tables[rule.source_table]
    if rel.child_key not in child_df.columns or rel.parent_key not in parent_df.columns:
        raise DistributionRuleError(
            f"Missing join keys for cross-table rule: {rule.raw}"
        )
    parent_mask = _condition_mask(parent_df[rule.condition_column], rule.condition_operator, rule.condition_value)
    eligible_parent_ids = set(parent_df.loc[parent_mask, rel.parent_key].dropna().tolist())
    return child_df[rel.child_key].isin(eligible_parent_ids)


def _apply_cross_table_rule(
    tables: Mapping[str, pd.DataFrame],
    schema: Any,
    rule: CrossTableRule,
    rng: np.random.Generator,
    *,
    exact: bool,
) -> None:
    path, hops = resolve_relationship_path(schema, rule.source_table, rule.target_table)
    child_df = tables[rule.target_table]
    scope_mask = _child_scope_mask(tables, schema, rule)
    scope_rows = int(scope_mask.sum())
    if scope_rows <= 0:
        return
    target_n = rule.target_count(scope_rows)
    target_value = _coerce_for_series(rule.target_value, child_df[rule.target_column])
    current_mask = scope_mask & _condition_mask(child_df[rule.target_column], rule.target_operator, target_value)
    if exact:
        scope_index = child_df.index[scope_mask]
        target_index = _choose_exact_target_index(scope_index, current_mask.loc[scope_index], target_n, rng)
        child_df.loc[target_index, rule.target_column] = target_value
        return
    needed = max(0, target_n - int(current_mask.sum()))
    if needed <= 0:
        return
    candidates = child_df.index[scope_mask & ~current_mask]
    chosen = _sample_index(candidates.to_numpy(), needed, rng)
    child_df.loc[chosen, rule.target_column] = target_value


def _apply_weighted_rule(
    tables: Mapping[str, pd.DataFrame],
    schema: Any,
    rule: WeightedRule,
    rng: np.random.Generator,
) -> None:
    table_name = rule.target_table or rule.table
    if table_name not in tables:
        if rule.table in tables:
            table_name = rule.table
        else:
            return
    df = tables[table_name]
    if rule.scope_column not in df.columns or rule.target_column not in df.columns:
        return
    scope_mask = _condition_mask(df[rule.scope_column], rule.scope_operator, rule.scope_value)
    target_value = _coerce_for_series(rule.target_value, df[rule.target_column])
    baseline_rate = float((df[rule.target_column] == target_value).mean()) if len(df) else 0.0
    scoped = df.loc[scope_mask]
    if scoped.empty:
        return
    scoped_rate = float((scoped[rule.target_column] == target_value).mean())
    desired_rate = min(0.99, scoped_rate * rule.multiplier if scoped_rate > 0 else baseline_rate * rule.multiplier)
    current_hits = int((scoped[rule.target_column] == target_value).sum())
    target_hits = int(round(desired_rate * len(scoped)))
    needed = max(0, target_hits - current_hits)
    if needed <= 0:
        return
    candidates = scoped.index[scoped[rule.target_column] != target_value]
    chosen = _sample_index(candidates.to_numpy(), min(needed, len(candidates)), rng)
    df.loc[chosen, rule.target_column] = target_value


def _evaluate_cross_table_rule(
    tables: Mapping[str, pd.DataFrame],
    rule: CrossTableRule,
    *,
    schema: Any,
    tolerance_percent: float,
) -> Dict[str, Any]:
    if rule.target_table not in tables or rule.source_table not in tables:
        return {
            "rule": rule.raw,
            "rule_type": "cross_table",
            "passed": False,
            "failure_reason": "Missing table for cross-table evaluation",
            "fix_suggestion": "Ensure both parent and child tables are generated.",
        }
    if schema is None:
        return {
            "rule": rule.raw,
            "rule_type": "cross_table",
            "passed": False,
            "failure_reason": "Schema required for cross-table evaluation",
        }
    try:
        path, hops = resolve_relationship_path(schema, rule.source_table, rule.target_table)
    except DistributionRuleError as exc:
        return {
            "rule": rule.raw,
            "rule_type": "cross_table",
            "passed": False,
            "failure_reason": str(exc),
            "fix_suggestion": "Add an explicit relationship in SchemaConfig.relationships.",
        }

    scope_mask = _child_scope_mask(tables, schema, rule)
    child_df = tables[rule.target_table]
    scope_rows = int(scope_mask.sum())
    target_rows = rule.target_count(scope_rows)
    target_mask = scope_mask & _condition_mask(child_df[rule.target_column], rule.target_operator, rule.target_value)
    actual_rows = int(target_mask.sum())
    actual_percent = 0.0 if scope_rows == 0 else (actual_rows / scope_rows) * 100.0
    difference = actual_percent - rule.percent
    passed = abs(difference) <= tolerance_percent or actual_rows == target_rows
    return {
        "rule": rule.raw,
        "rule_type": "cross_table",
        "source_table": rule.source_table,
        "target_table": rule.target_table,
        "condition_column": rule.condition_column,
        "condition_operator": rule.condition_operator,
        "condition_value": rule.condition_value,
        "target_column": rule.target_column,
        "target_operator": rule.target_operator,
        "target_value": rule.target_value,
        "relationship_path": list(path),
        "hop_count": hops,
        "parent_scope_rows": scope_rows,
        "child_scope_rows": scope_rows,
        "target_percent": round(float(rule.percent), 3),
        "actual_percent": round(float(actual_percent), 3),
        "target_rows": target_rows,
        "actual_rows": actual_rows,
        "difference_pp": round(float(difference), 3),
        "passed": bool(passed),
        "status": "Passed" if passed else "Failed",
    }


def _evaluate_weighted_rule(tables: Mapping[str, pd.DataFrame], rule: WeightedRule) -> Dict[str, Any]:
    table_name = rule.target_table or rule.table
    if table_name not in tables:
        table_name = rule.table
    if table_name not in tables:
        return {
            "rule": rule.raw,
            "rule_type": "weighted",
            "passed": False,
            "failure_reason": f"Table '{table_name}' not found",
        }
    df = tables[table_name]
    if rule.scope_column not in df.columns or rule.target_column not in df.columns:
        return {
            "rule": rule.raw,
            "rule_type": "weighted",
            "passed": False,
            "failure_reason": "Missing scope or target column",
        }
    scope_mask = _condition_mask(df[rule.scope_column], rule.scope_operator, rule.scope_value)
    scoped = df.loc[scope_mask]
    scope_rows = int(len(scoped))
    baseline_rate = float((df[rule.target_column] == rule.target_value).mean()) if len(df) else 0.0
    scoped_before_rate = baseline_rate
    scoped_after_rate = float((scoped[rule.target_column] == rule.target_value).mean()) if scope_rows else 0.0
    multiplier_achieved = (scoped_after_rate / baseline_rate) if baseline_rate > 0 else None
    passed = scoped_after_rate >= baseline_rate if rule.multiplier >= 1 else scoped_after_rate <= baseline_rate
    warnings: List[str] = []
    if multiplier_achieved is not None and abs(multiplier_achieved - rule.multiplier) > 0.5:
        warnings.append("Achieved multiplier differs from requested lift")
        passed = False
    return {
        "rule": rule.raw,
        "rule_type": "weighted",
        "table": table_name,
        "baseline_rate": round(baseline_rate, 4),
        "scoped_before_rate": round(scoped_before_rate, 4),
        "scoped_after_rate": round(scoped_after_rate, 4),
        "multiplier_requested": rule.multiplier,
        "multiplier_achieved": round(multiplier_achieved, 4) if multiplier_achieved is not None else None,
        "scope_rows": scope_rows,
        "eligible_rows": scope_rows,
        "target_percent": "N/A",
        "target_rows": "N/A",
        "actual_rows": int((scoped[rule.target_column] == rule.target_value).sum()) if scope_rows else 0,
        "passed": bool(passed),
        "status": "Passed" if passed else "Review",
        "warnings": warnings,
    }


def _missing_record(rule: DistributionRule, message: str) -> Dict[str, Any]:
    return {
        "rule": rule.raw,
        "table": rule.table,
        "column": rule.column,
        "condition": f"{rule.column} {rule.operator} {rule.value}",
        "target_percent": round(float(rule.percent), 3),
        "actual_percent": None,
        "target_rows": None,
        "actual_rows": None,
        "difference_pp": None,
        "passed": False,
        "status": message,
    }


@dataclass
class GlobalRuleBudget:
    """Track global distribution-rule targets across streamed chunks."""

    rules: List[DistributionRule]
    table_totals: Dict[str, int]
    assigned: Dict[str, int]

    @classmethod
    def from_rules(cls, rules: Sequence[RuleSpec], table_totals: Mapping[str, int]) -> "GlobalRuleBudget":
        pct = percent_rules(rules)
        return cls(
            rules=pct,
            table_totals={name: int(total) for name, total in table_totals.items()},
            assigned={rule.raw: 0 for rule in pct},
        )

    def chunk_target(self, rule: DistributionRule, *, rows_before: int, chunk_size: int) -> int:
        total = self.table_totals.get(rule.table, 0)
        if total <= 0 or chunk_size <= 0:
            return 0
        global_target = rule.target_count(total)
        end_rows = min(total, rows_before + chunk_size)
        expected_end = int(round(global_target * end_rows / total))
        expected_start = int(round(global_target * rows_before / total))
        return max(0, expected_end - expected_start)


def apply_distribution_rules_to_chunk(
    df: pd.DataFrame,
    schema: Any,
    rules: Sequence[RuleSpec],
    budget: GlobalRuleBudget,
    *,
    table_name: str,
    rows_before: int,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Apply distribution rules to one chunk using globally allocated targets."""
    pct = percent_rules(rules)
    if not pct or df.empty:
        return df

    output = df.copy()
    table_rules = [rule for rule in pct if rule.table == table_name]
    if table_rules:
        validate_distribution_rules(schema, table_rules, {table_name: output})
    base_seed = int(seed if seed is not None else 0)

    for rule_index, rule in enumerate(pct):
        if rule.table != table_name:
            continue
        chunk_target = budget.chunk_target(rule, rows_before=rows_before, chunk_size=len(output))
        if chunk_target <= 0:
            continue
        rng = np.random.default_rng(base_seed + rule_index + rows_before + 9973)
        partial_rule = DistributionRule(
            raw=rule.raw,
            table=rule.table,
            column=rule.column,
            percent=min(100.0, (chunk_target / max(len(output), 1)) * 100.0),
            operator=rule.operator,
            value=rule.value,
            exact=rule.exact,
        )
        if rule.operator in _EQUALITY_OPERATORS:
            _apply_equality_rule(output, schema, partial_rule, rng, exact=False)
        elif rule.operator in _COMPARISON_OPERATORS:
            _apply_numeric_comparison_rule(output, partial_rule, rng, exact=False)
        elif rule.operator == "!=":
            _apply_not_equal_rule(output, schema, partial_rule, rng, exact=False)
        mask = _condition_mask(output[rule.column], rule.operator, rule.value)
        budget.assigned[rule.raw] = budget.assigned.get(rule.raw, 0) + int(mask.sum())

    return output


def aggregate_rule_evidence(
    chunk_records: Sequence[Sequence[Dict[str, Any]]],
    *,
    rules: Sequence[RuleSpec],
    table_totals: Mapping[str, int],
    tolerance_percent: float = 0.25,
) -> List[Dict[str, Any]]:
    """Merge per-chunk rule evidence into global pass/fail records."""
    if not rules:
        return []

    pct = percent_rules(rules)
    actual_by_rule: Dict[str, int] = {rule.raw: 0 for rule in pct}
    for chunk in chunk_records:
        for record in chunk:
            key = str(record.get("rule"))
            if key in actual_by_rule and record.get("actual_rows") is not None:
                actual_by_rule[key] += int(record["actual_rows"])

    aggregated: List[Dict[str, Any]] = []
    for rule in pct:
        total = int(table_totals.get(rule.table, 0))
        target_rows = rule.target_count(total)
        actual_rows = int(actual_by_rule.get(rule.raw, 0))
        actual_percent = 0.0 if total == 0 else (actual_rows / total) * 100.0
        difference = actual_percent - rule.percent
        passed = abs(difference) <= tolerance_percent or actual_rows == target_rows
        aggregated.append(
            {
                "rule": rule.raw,
                "table": rule.table,
                "column": rule.column,
                "condition": f"{rule.column} {rule.operator} {rule.value}",
                "target_percent": round(float(rule.percent), 3),
                "actual_percent": round(float(actual_percent), 3),
                "target_rows": target_rows,
                "actual_rows": actual_rows,
                "difference_pp": round(float(difference), 3),
                "passed": bool(passed),
                "status": "Passed" if passed else "Failed",
                "aggregated": True,
            }
        )
    return aggregated


__all__ = [
    "CrossTableRule",
    "DistributionRule",
    "DistributionRuleError",
    "GlobalRuleBudget",
    "RuleSpec",
    "WeightedRule",
    "aggregate_rule_evidence",
    "apply_distribution_rules",
    "apply_distribution_rules_to_chunk",
    "cross_table_rules",
    "evaluate_distribution_rules",
    "parse_distribution_rules",
    "percent_rules",
    "resolve_relationship_path",
    "validate_distribution_rules",
    "weighted_rules",
]