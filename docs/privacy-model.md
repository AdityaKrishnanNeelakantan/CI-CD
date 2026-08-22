# Privacy Model

The platform treats privacy as an explicit policy/evaluation concern rather than making blanket anonymity guarantees.

## Policy concepts

The domain privacy layer models actions such as preserving safe public categories, transforming direct identifiers, handling rare/quasi-identifying values, excluding unsuitable free text, and redacting document content where required.

## Evaluation

Validation code includes privacy-oriented evaluation such as distance-to-closest-record style checks and membership-inference-style metrics where configured. Required checks that do not run should be surfaced as `NOT_RUN`/`SKIPPED` rather than silently reported as passing.

## Differential privacy

Do not claim formal differential privacy merely because noise, sanitization, or privacy metrics are present. A formal DP claim requires the DP mechanism/accountant and its report to have actually run with defined parameters and guarantees.

## PDF handling

PDF Twin separates source de-identification/redaction from synthetic generation. The redacted-source operation is optional and produces its own report; synthetic rendering has separate validation and ground-truth artifacts.
