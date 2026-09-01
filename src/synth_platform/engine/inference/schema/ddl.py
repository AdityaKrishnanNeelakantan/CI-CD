"""
SQL DDL → SchemaConfig parser.

Converts CREATE TABLE statements into a MVP SchemaConfig that can be passed
directly to ``mvp.generate_from_schema()``.

Supported SQL dialects: PostgreSQL, MySQL, SQLite, generic ANSI SQL.
"""

import re
import warnings
from typing import Dict, List, Optional, Tuple

from synth_platform.engine.inference.schema.schema import Column, Relationship, SchemaConfig, Table


# SQL type → MVP type
_TYPE_MAP: List[Tuple[str, str]] = [
    (r"bool(?:ean)?",                                   "boolean"),
    (r"(?:big|small|tiny)?int(?:eger)?(?:\s*\(\d+\))?|serial|bigserial|smallserial", "int"),
    (r"(?:double\s+precision|float|real|decimal|numeric|money)(?:\s*\(\d+(?:,\s*\d+)?\))?", "float"),
    (r"(?:timestamp(?:tz)?|datetime)(?:\s+with(?:out)?\s+time\s+zone)?(?:\s*\(\d+\))?", "date"),
    (r"date",                                            "date"),
    (r"(?:var)?char(?:acter)?(?:\s+varying)?(?:\s*\(\d+\))?|text|string|clob|varchar2", "text"),
    (r"uuid",                                            "text"),
    (r"json(?:b)?",                                      "text"),
    (r"time(?:\s+with(?:out)?\s+time\s+zone)?(?:\s*\(\d+\))?", "text"),
]

_COMPILED: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^" + p + r"$", re.IGNORECASE), t)
    for p, t in _TYPE_MAP
]


def _map_sql_type(raw: str) -> str:
    raw = raw.strip()
    for pattern, mvp_type in _COMPILED:
        if pattern.match(raw):
            return mvp_type
    return "text"


def _strip_comments(ddl: str) -> str:
    """Remove SQL line comments (--) and block comments (/* */)."""
    ddl = re.sub(r"/\*.*?\*/", " ", ddl, flags=re.DOTALL)
    ddl = re.sub(r"--[^\n]*", " ", ddl)
    return ddl


def _split_column_defs(body: str) -> List[str]:
    """Split the body of a CREATE TABLE into individual definitions, respecting parentheses."""
    parts: List[str] = []
    depth = 0
    current: List[str] = []
    for ch in body:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return [p for p in parts if p]


def from_ddl(
    ddl: str,
    *,
    infer_fks: bool = True,
    default_rows: int = 1000,
) -> SchemaConfig:
    """Parse SQL DDL CREATE TABLE statements into a :class:`SchemaConfig`.

    Supports PostgreSQL, MySQL, SQLite, and generic ANSI SQL. Inline
    ``REFERENCES`` clauses and standalone ``FOREIGN KEY`` constraints both
    produce :class:`Relationship` entries.

    Args:
        ddl:          One or more ``CREATE TABLE`` statements as a string.
        infer_fks:    If True (default), columns named ``<table>_id`` that
                      don't have an explicit ``REFERENCES`` clause are treated
                      as foreign keys to the table named by the prefix.
                      Set to False to only use explicit constraints.
        default_rows: Row count assigned to each generated table (default 1000).

    Returns:
        :class:`SchemaConfig` ready for :func:`mvp.generate_from_schema`.

    Example::

        schema = mvp.from_ddl(\"\"\"
            CREATE TABLE users (
                id SERIAL PRIMARY KEY,
                email VARCHAR(255) NOT NULL,
                created_at TIMESTAMP
            );
            CREATE TABLE orders (
                id SERIAL PRIMARY KEY,
                user_id INT REFERENCES users(id),
                amount DECIMAL(10, 2),
                placed_at TIMESTAMP
            );
        \"\"\")
        tables = mvp.generate_from_schema(schema)
        print(tables["orders"].head())
    """
    ddl = _strip_comments(ddl)

    # Match CREATE TABLE headers and then walk character-by-character to find the
    # matching closing paren — handles nested parens like DECIMAL(10,2) and
    # REFERENCES users(id) without the non-greedy .*? truncation bug.
    header_pattern = re.compile(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
        r"(?:\"?[\w]+\"?\.)?"       # optional schema prefix
        r"\"?([\w]+)\"?"            # table name (group 1)
        r"\s*\(",
        re.IGNORECASE,
    )

    def _extract_tables(ddl_text: str) -> List[Tuple[str, str]]:
        results = []
        for hdr in header_pattern.finditer(ddl_text):
            tname = hdr.group(1)
            depth = 1
            i = hdr.end()
            while i < len(ddl_text) and depth > 0:
                ch = ddl_text[i]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                i += 1
            results.append((tname, ddl_text[hdr.end(): i - 1]))
        return results

    fk_inline = re.compile(
        r"REFERENCES\s+(?:\"?[\w]+\"?\.)?"
        r"\"?([\w]+)\"?"
        r"\s*\(\s*\"?([\w]+)\"?\s*\)",
        re.IGNORECASE,
    )
    fk_constraint = re.compile(
        r"FOREIGN\s+KEY\s*\(\s*\"?([\w]+)\"?\s*\)"
        r"\s+REFERENCES\s+(?:\"?[\w]+\"?\.)?"
        r"\"?([\w]+)\"?"
        r"\s*\(\s*\"?([\w]+)\"?\s*\)",
        re.IGNORECASE,
    )
    skip_pattern = re.compile(
        r"^\s*(?:PRIMARY\s+KEY|UNIQUE|CHECK|CONSTRAINT\s+\w+\s+(?:PRIMARY|UNIQUE|CHECK)|INDEX)\b",
        re.IGNORECASE,
    )
    col_pattern = re.compile(r"^\"?([\w]+)\"?\s+([\w]+(?:\s*\(\s*\d+(?:\s*,\s*\d+)?\s*\))?)", re.IGNORECASE)

    tables: List[Table] = []
    columns_map: Dict[str, List[Column]] = {}
    relationships: List[Relationship] = []
    all_table_names: List[str] = []
    # Deferred FK specs: (child_table, child_col, parent_table, parent_col)
    pending_fks: List[Tuple[str, str, str, str]] = []

    for table_name, body in _extract_tables(ddl):
        all_table_names.append(table_name)
        cols: List[Column] = []
        explicit_fk_cols: set = set()
        pk_cols: set = set()

        # Table-level PRIMARY KEY (col) / PRIMARY KEY (a, b)
        table_pk = re.search(
            r"PRIMARY\s+KEY\s*\(\s*([^)]+)\)",
            body,
            re.IGNORECASE,
        )
        if table_pk:
            for part in table_pk.group(1).split(","):
                pk_name = part.strip().strip('`"[]')
                if pk_name:
                    pk_cols.add(pk_name)

        for line in _split_column_defs(body):
            # Standalone FOREIGN KEY constraint
            fk_match = fk_constraint.search(line)
            if fk_match:
                child_col, parent_table, parent_col = fk_match.groups()
                pending_fks.append((table_name, child_col, parent_table, parent_col))
                explicit_fk_cols.add(child_col)
                continue

            # Skip other table-level constraints
            if skip_pattern.match(line):
                continue

            # Column definition
            col_match = col_pattern.match(line)
            if not col_match:
                continue

            col_name = col_match.group(1)
            sql_type = col_match.group(2)
            mvp_type = _map_sql_type(sql_type)
            upper_line = line.upper()
            is_pk = "PRIMARY KEY" in upper_line or col_name in pk_cols
            if is_pk:
                pk_cols.add(col_name)
            nullable = "NOT NULL" not in upper_line and not is_pk

            # Inline REFERENCES
            inline = fk_inline.search(line)
            if inline:
                parent_table, parent_col = inline.group(1), inline.group(2)
                pending_fks.append((table_name, col_name, parent_table, parent_col))
                explicit_fk_cols.add(col_name)
                mvp_type = "foreign_key"
                distribution_params: Dict = {"references": f"{parent_table}.{parent_col}"}
            else:
                distribution_params = {}

            cols.append(
                Column(
                    name=col_name,
                    type=mvp_type,
                    nullable=nullable,
                    unique=is_pk,
                    distribution_params=distribution_params,
                )
            )

        # FK inference from _id suffix — never on primary keys / self id columns
        if infer_fks:
            for col in cols:
                if col.name in pk_cols or col.name in explicit_fk_cols:
                    continue
                if col.name in (f"{table_name}_id", "id"):
                    continue
                if not col.name.endswith("_id"):
                    continue
                guessed_parent = col.name[:-3]
                pending_fks.append((table_name, col.name, guessed_parent, "id"))

        tables.append(Table(name=table_name, row_count=default_rows))
        columns_map[table_name] = cols

    if not tables:
        raise ValueError(
            "No CREATE TABLE statements found in DDL. "
            "Make sure each statement follows the standard syntax: "
            "CREATE TABLE name (col_definitions);"
        )

    known = set(all_table_names)
    valid_rels: List[Relationship] = []
    dropped = 0
    valid_fk_children: Dict[str, set] = {t: set() for t in all_table_names}

    for child_table, child_col, parent_table, parent_col in pending_fks:
        if parent_table not in known or parent_table == child_table:
            dropped += 1
            continue
        # Prefer parent PK name when guessed as bare "id" but parent uses "{singular}_id"
        parent_cols = {c.name for c in columns_map.get(parent_table, [])}
        resolved_parent_key = parent_col
        if parent_col == "id" and "id" not in parent_cols:
            candidates = [c for c in parent_cols if c.endswith("_id")]
            # Prefer {parent}_id or singular form matching child_col prefix
            preferred = f"{parent_table}_id"
            if preferred in parent_cols:
                resolved_parent_key = preferred
            elif child_col in parent_cols:
                resolved_parent_key = child_col
            elif len(candidates) == 1:
                resolved_parent_key = candidates[0]
        if resolved_parent_key not in parent_cols and parent_col not in parent_cols:
            dropped += 1
            continue
        if resolved_parent_key not in parent_cols:
            resolved_parent_key = parent_col

        valid_rels.append(
            Relationship(
                parent_table=parent_table,
                child_table=child_table,
                parent_key=resolved_parent_key,
                child_key=child_col,
            )
        )
        valid_fk_children[child_table].add(child_col)

    if dropped:
        warnings.warn(
            f"{dropped} inferred FK relationship(s) dropped because the referenced "
            "table was not found in the DDL. Pass infer_fks=False to suppress.",
            UserWarning,
            stacklevel=2,
        )

    # Promote ONLY columns that remain on valid relationships to foreign_key.
    # Invalid inferences must not leave orphan foreign_key column types behind.
    for table_name, cols in columns_map.items():
        fk_names = valid_fk_children.get(table_name, set())
        new_cols: List[Column] = []
        for col in cols:
            if col.name in fk_names and col.type != "foreign_key":
                parent_rel = next(
                    r for r in valid_rels if r.child_table == table_name and r.child_key == col.name
                )
                new_cols.append(
                    Column(
                        name=col.name,
                        type="foreign_key",
                        nullable=col.nullable,
                        unique=col.unique,
                        distribution_params={
                            "references": f"{parent_rel.parent_table}.{parent_rel.parent_key}"
                        },
                    )
                )
            elif col.type == "foreign_key" and col.name not in fk_names:
                # Explicit inline REFERENCES that somehow lost its relationship — keep text
                new_cols.append(
                    Column(
                        name=col.name,
                        type="text",
                        nullable=col.nullable,
                        unique=col.unique,
                        distribution_params={},
                    )
                )
            else:
                new_cols.append(col)
        columns_map[table_name] = new_cols

    return SchemaConfig(
        name="from_ddl",
        tables=tables,
        columns=columns_map,
        relationships=valid_rels,
    )
