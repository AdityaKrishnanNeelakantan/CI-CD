# Schema Twin data-quality fixes

## Part A: linked column consistency

### What was broken

Schema-driven generation dispatches one column at a time. Categorical columns
such as `City` and `Country` were therefore sampled independently from their
own marginal `choices` distributions. YAML parsing in
`engine/inference/schema/yaml_schema.py` preserved those separate marginal
definitions and did not infer a link between them.

The repository already had two relevant mechanisms:

- Explicit conditional generation through a column's `depends_on` and
  `mapping` parameters.
- `EntityCoherenceEngine`, including a country-aware geography repair.

Neither prevented the reported output. Conditional generation must be
declared explicitly and depends on the source column having already been
generated. Coherence is opt-in and defaults to `off`; its geography rule also
matched only lowercase column headers. In addition, a flat capsule city
vocabulary could establish that `Dallas` was a known city, but not that it was
invalid for `India`. Finally, the pipeline's later duplicate repair could
mutate one member of a valid tuple independently.

### What changed

Schema generation now applies a high-confidence linked-field correctness rule
at generation and pipeline boundaries. It:

- Recognizes case-insensitive names such as `City`/`Country`, as well as
  prefixed forms such as `branch_city`/`branch_country`.
- Canonicalizes common country aliases such as `US`, `USA`, and `UK`.
- Validates and conditionally samples city values using the existing
  `CITIES_BY_COUNTRY` lookup, so known cities are paired with a country-valid
  city instead of being sampled independently.
- Applies the same concrete repair to State/Country where
  `STATES_BY_COUNTRY` has coverage.
- Aligns `branch_name` with the repaired City while preserving its branch,
  office, or service-center suffix.
- Reapplies the linked-field rule after distribution and duplicate repairs so
  a later single-column mutation cannot reintroduce the defect.
- Resamples any city outside the configured country pool when the country is
  known, including locale-generated cities absent from the bundled lookup.

No new geography dependency or general-purpose context synthesizer was added.

### Related risks found but deferred

The search found other independently sampled relationships that need separate,
domain-specific treatment:

- City and State are each conditioned on Country but are not mutually linked,
  so a valid country can still contain an inconsistent City/State pair.
- Postal code, latitude, and longitude each select their own independent row
  from `CITY_GEODATA`; they are not linked to each other or to City/Country.
- The smart-value composite address template chooses City and State from
  unrelated lists.
- First name and gender have no conditional relationship. Identity coherence
  covers lowercase name/email/username combinations only when enabled, but it
  does not model gender.
- Row-level Country does not drive phone format, national ID, IBAN, SWIFT/BIC,
  or address locale; those generators use schema/locale configuration.
- Explicit `depends_on` mappings remain generation-order-sensitive because the
  parent column must already exist in the partial row frame.

## Part B: relational row counts

### What was broken

`generate_from_schema()` treated `default_record_count` / the UI's former
“Rows per table” value as a flat count. `prepare_schema_for_generation()`
overwrote every `Table.row_count` with that value. The downstream pipeline then
preserved those equal proportions, producing identical counts for every table.

Schema-driven relationship detection was already available:

- YAML and flexible JSON load explicit `relationships`.
- SQL DDL loading runs with foreign-key inference enabled.
- Semantic enrichment can infer missing parent/child edges from `*_id`
  columns and promotes child keys to `foreign_key`.

The engine also already contained `GenerationPlanner`, semantic cardinality
defaults, `RealismConfig.relationship_multipliers`, and per-table
`row_count_overrides`. Database Twin separately accepts
`row_counts_by_table`, populated from source counts or user overrides. Database
Twin was not changed.

### What changed

Schema Twin now uses the global row count as a **base count**:

1. Semantic enrichment resolves the relationship graph.
2. Root tables receive the base count.
   Small reference dimensions such as branches use the planner's bounded
   reference-table behavior.
3. Existing `GenerationPlanner` cardinality patterns scale child tables (for
   example, customer-to-order and account-to-transaction relationships).
4. Existing explicit relationship multipliers remain configurable through
   `RealismConfig.relationship_multipliers`.
5. Planned counts are materialized onto the schema before the pipeline runs,
   keeping generation, validation, export expectations, unique-ID ranges, and
   result metadata aligned.

The Streamlit and web labels now say “Base rows” and explain FK-aware scaling.
API result metadata reports `row_count_mode: "fk_aware"`. If no relationship
can be detected, tables retain the base count; this phase does not invent
relationships between unrelated tables.

For children with multiple parents, every relationship is now evaluated and
the strongest cardinality signal wins. This prevents a weak
merchant-to-transaction edge encountered first from masking the established
account-to-transaction ratio. Banking table constellations without an explicit
domain are assigned the existing `fintech` priors, which provide bounded loan
interest rates and realistic monetary distributions. Account Type and Account
Status are also inferred separately.

Bounded advisory samples no longer run referential-integrity checks against
independently sampled parent rows. Exact exported PK/FK validation remains the
authority, eliminating contradictory orphan warnings and misleading low
quality scores. Locale-city checks likewise defer to explicit row-level
Country values for multi-country tables.

`default_record_count` remains the default base when the caller supplies no
run-specific value.

## Verification

Added regression coverage proves that:

- A branches-like table using title-cased `City` and `Country` headers emits
  only pairs present in `CITIES_BY_COUNTRY`.
- A parent/child schema with a configured `3.0` relationship multiplier emits
  12 parent rows and 36 child rows, with valid foreign keys.
- Locale-only cities absent from the lookup cannot escape Country conditioning,
  branch names match their row City, and banking schemas receive bounded
  fintech defaults.
- Multi-parent row planning uses the strongest edge, and sampled advisory
  reports do not invent FK orphans when the complete export passes.

Executed results:

- Focused regressions: `7 passed in 2.37s`.
- Schema unit/integration and Schema Twin API/UI suites: `253 passed, 11
  warnings in 59.59s`.
- Web tests: `23 passed`.
- Full pytest suite: `1 failed, 1454 passed, 11 skipped, 74 warnings in
  96.65s`.

The stated baseline was `1407 passed, 11 skipped`; the current dirty worktree
collects additional tests and reached 1454 passes with the same skip count.
The sole failure is
`tests/integration/test_api_intent.py::test_database_sample_source_creates_readable_sqlite_session`:
an unrelated, pre-existing Database Twin worktree change returns dictionaries
in `source["tables"]` while that test calls `set()` as if they were strings.
That route is outside this task and Database Twin's existing logic was
explicitly excluded from modification.

## Remaining scope

This fix guarantees known City/Country (and covered State/Country) consistency
in Schema Twin and removes forced flat counts where relationships are known.
It does not claim exhaustive worldwide geographic validation, infer business
cardinality for schemas without relationships, or solve the deferred linked
field risks listed above.
