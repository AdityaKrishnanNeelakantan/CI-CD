import pandas as pd

from synth_platform.engine.validation.schema.reporting import analyze_generation, build_validation_report
from synth_platform.engine.inference.schema.schema import Column, RealismConfig, Relationship, SchemaConfig, Table


def test_analyze_generation_returns_requested_reports():
    schema = SchemaConfig(
        name="Reporting",
        tables=[Table(name="users", row_count=3)],
        columns={
            "users": [
                Column(name="age", type="int", distribution_params={"distribution": "uniform", "min": 18, "max": 65}),
                Column(name="status", type="categorical", distribution_params={"choices": ["A", "B"]}),
            ]
        },
        realism=RealismConfig(reports=["privacy", "fidelity", "data_card"]),
    )

    tables = {
        "users": pd.DataFrame(
            {
                "age": [24, 33, 41],
                "status": ["A", "A", "B"],
            }
        )
    }

    reports = analyze_generation(tables, schema, reports=schema.realism.reports)

    assert set(reports.keys()) == {"privacy", "fidelity", "data_card"}
    assert reports["data_card"].name == "Reporting"


def test_build_validation_report_contains_guarantees_and_advisory_sections():
    schema = SchemaConfig(
        name="Oracle",
        seed=42,
        tables=[Table(name="users", row_count=3)],
        columns={
            "users": [
                Column(name="id", type="int", unique=True, distribution_params={"min": 1, "max": 4}),
                Column(name="age", type="int", distribution_params={"distribution": "uniform", "min": 18, "max": 65}),
            ]
        },
    )
    tables = {"users": pd.DataFrame({"id": [1, 2, 3], "age": [24, 33, 41]})}

    report = build_validation_report(tables, schema, seed=42)

    assert report["mvp_report"] == "validation"
    assert report["passed"] is True
    assert report["guarantees"]["row_count_fulfillment"]["passed"] is True
    assert "privacy" in report["advisory"]
    assert report["reproducibility"]["seed"] == 42


def test_oracle_locale_fit_checks_country_city_phone_and_national_id():
    schema = SchemaConfig(
        name="Brazil Oracle",
        tables=[Table(name="customers", row_count=2)],
        columns={
            "customers": [
                Column(name="country", type="text", distribution_params={"text_type": "country"}),
                Column(name="city", type="text", distribution_params={"text_type": "city"}),
                Column(name="phone", type="text", distribution_params={"text_type": "phone"}),
                Column(name="national_id", type="text", distribution_params={"text_type": "national_id"}),
            ]
        },
        realism=RealismConfig(locale="pt_BR"),
    )
    tables = {
        "customers": pd.DataFrame(
            {
                "country": ["Brazil", "Brazil"],
                "city": ["São Paulo", "Rio de Janeiro"],
                "phone": ["+55 11999999999", "+55 21988888888"],
                "national_id": ["123.456.789-10", "987.654.321-99"],
            }
        )
    }

    report = build_validation_report(tables, schema)

    locale_fit = report["advisory"]["locale_domain_fit"]
    assert locale_fit["passed"] is True
    assert locale_fit["locale"] == "pt_BR"


def test_full_export_validation_does_not_report_sampled_fk_false_positives():
    schema = SchemaConfig(
        name="Sampled relations",
        tables=[Table(name="parents", row_count=200), Table(name="children", row_count=200)],
        columns={
            "parents": [Column(name="id", type="int", unique=True)],
            "children": [Column(name="parent_id", type="foreign_key")],
        },
        relationships=[
            Relationship(parent_table="parents", parent_key="id", child_table="children", child_key="parent_id")
        ],
    )
    independently_sampled = {
        "parents": pd.DataFrame({"id": [1, 2]}),
        "children": pd.DataFrame({"parent_id": [150, 175]}),
    }
    export_validation = {
        "passed": True,
        "pk_passed": True,
        "fk_passed": True,
        "fk": [{"check": "children.parent_id->parents.id", "orphan_rows": 0, "passed": True}],
    }

    report = build_validation_report(
        independently_sampled,
        schema,
        row_counts={"parents": 200, "children": 200},
        sampled=True,
        export_validation=export_validation,
        validation_scope="full_export",
    )

    assert report["passed"] is True
    assert report["reproducibility"]["sampled"] is True
    assert report["guarantees"]["constraints"]["passed"] is True
    assert report["advisory"]["quality"]["passed"] is True
    assert not report["advisory"]["quality"]["issues"]


def test_credit_score_uses_credit_specific_quality_range():
    schema = SchemaConfig(
        name="Credit",
        tables=[Table(name="customers", row_count=3)],
        columns={"customers": [Column(name="credit_score", type="int")]},
    )
    report = build_validation_report(
        {"customers": pd.DataFrame({"credit_score": [300, 700, 850]})},
        schema,
    )

    issues = report["advisory"]["quality"]["issues"]
    assert not any(issue["column"] == "credit_score" for issue in issues)


def test_locale_city_check_defers_to_explicit_row_country():
    schema = SchemaConfig(
        name="Global branches",
        tables=[Table(name="branches", row_count=2)],
        columns={
            "branches": [
                Column(name="city", type="text"),
                Column(name="country", type="text"),
            ]
        },
        realism=RealismConfig(locale="en_US"),
    )
    report = build_validation_report(
        {
            "branches": pd.DataFrame(
                {"city": ["Mumbai", "Berlin"], "country": ["India", "Germany"]}
            )
        },
        schema,
    )

    assert report["advisory"]["locale_domain_fit"]["passed"] is True
    assert report["advisory"]["locale_domain_fit"]["checks"] == []
