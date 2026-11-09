from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Schemes.csv"
REPORT_PATH = ROOT / "reports" / "dataset_quality_report.json"
REQUIRED_COLUMNS = {
    "name",
    "description",
    "state",
    "category",
    "benefits",
    "eligibility_text",
    "documents_required",
    "apply_url",
    "official_url",
}
NULL_VALUES = {"", "null", "none", "nan", "n/a", "na"}


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().casefold() in NULL_VALUES | {"[]", "{}"}


def parse_value(value: Any) -> Any:
    if is_missing(value):
        return None
    return value


def parse_number(value: Any) -> float | None:
    if is_missing(value):
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else None


def parse_bool(value: Any) -> bool | None:
    if is_missing(value):
        return None
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def parse_json_list(value: Any) -> list[str] | None:
    if is_missing(value):
        return None
    try:
        decoded = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return [str(value).strip()]
    if isinstance(decoded, list):
        values = [str(item).strip() for item in decoded if str(item).strip()]
        return values or None
    return [str(decoded).strip()] if decoded else None


def normalize_state(value: Any) -> str | None:
    if is_missing(value):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    canonical = {
        "andaman and nicobar islands": "Andaman and Nicobar Islands",
        "dadra and nagar haveli and daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
        "jammu & kashmir": "Jammu and Kashmir",
        "orissa": "Odisha",
        "pondicherry": "Puducherry",
    }
    return canonical.get(text.casefold(), text.title() if text.casefold() != "central" else "Central")


def normalize_category(value: Any) -> str | None:
    if is_missing(value):
        return None
    return ", ".join(
        re.sub(r"\s+", " ", part).strip().title()
        for part in str(value).split(",")
        if part.strip()
    )


def _url_present(value: Any) -> bool:
    if is_missing(value):
        return False
    return urlparse(str(value).strip()).scheme in {"http", "https"}


def load_schemes(path: Path = DATA_PATH) -> list[dict[str, Any]]:
    """Load and normalize the supplied CSV without discarding source text."""
    if not path.is_file():
        raise FileNotFoundError(f"Scheme dataset not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    missing_columns = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    records: list[dict[str, Any]] = []
    seen_slugs: set[str] = set()
    for row in frame.to_dict(orient="records"):
        scheme = {key: parse_value(value) for key, value in row.items()}
        scheme["state"] = normalize_state(scheme.get("state"))
        scheme["category"] = normalize_category(scheme.get("category"))
        scheme["eligibility_state"] = parse_json_list(scheme.get("eligibility_state"))
        scheme["eligibility_caste"] = parse_json_list(scheme.get("eligibility_caste"))
        scheme["eligibility_age_min"] = parse_number(scheme.get("eligibility_age_min"))
        scheme["eligibility_age_max"] = parse_number(scheme.get("eligibility_age_max"))
        scheme["eligibility_income_max"] = parse_number(scheme.get("eligibility_income_max"))
        scheme["eligibility_disability"] = parse_bool(scheme.get("eligibility_disability"))
        scheme["eligibility_bpl"] = parse_bool(scheme.get("eligibility_bpl"))
        slug = str(scheme.get("slug") or "").strip()
        scheme["slug"] = slug or re.sub(r"[^a-z0-9]+", "-", str(scheme["name"]).casefold()).strip("-")
        if scheme["slug"] in seen_slugs:
            scheme["slug"] = f"{scheme['slug']}-{len(records)}"
        seen_slugs.add(scheme["slug"])
        records.append(scheme)
    return records


def build_quality_report(path: Path = DATA_PATH) -> dict[str, Any]:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    missing_columns = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    coverage: dict[str, dict[str, Any]] = {}
    for column in frame.columns:
        present = int((~frame[column].map(is_missing)).sum())
        coverage[column] = {
            "populated": present,
            "missing": int(len(frame) - present),
            "coverage_percent": round(present / len(frame) * 100, 2) if len(frame) else 0,
        }

    names = frame["name"].map(lambda value: re.sub(r"\s+", " ", value).strip().casefold())
    slugs = frame["slug"].map(lambda value: value.strip().casefold()) if "slug" in frame else pd.Series(dtype=str)
    return {
        "source": "Supplied CSV from Sumna/indian-government-schemes-2025 (third-party dataset)",
        "source_url": "https://huggingface.co/datasets/Sumna/indian-government-schemes-2025",
        "encoding": "utf-8-sig",
        "delimiter": ",",
        "quoting": "CSV double-quoted fields; parsed with pandas' CSV parser",
        "multiline_fields_preserved": True,
        "record_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": list(frame.columns),
        "required_columns_missing": missing_columns,
        "duplicate_name_rows": int(names.duplicated(keep="first").sum()),
        "duplicate_slug_rows": int(slugs.duplicated(keep="first").sum()) if len(slugs) else 0,
        "state_counts": {
            str(key): int(value)
            for key, value in frame["state"].map(normalize_state).value_counts(dropna=False).items()
        },
        "category_value_count": int(frame["category"].map(normalize_category).nunique(dropna=True)),
        "coverage": coverage,
        "structured_eligibility_coverage": {
            column: coverage.get(column, {"populated": 0, "missing": len(frame), "coverage_percent": 0})
            for column in (
                "eligibility_age_min",
                "eligibility_age_max",
                "eligibility_income_max",
                "eligibility_gender",
                "eligibility_residence",
                "eligibility_state",
                "eligibility_disability",
                "eligibility_bpl",
            )
        },
        "valid_http_url_count": {
            column: int(frame[column].map(_url_present).sum())
            for column in ("apply_url", "official_url")
            if column in frame
        },
        "limitations": [
            "The source is a third-party dataset and records have not been independently checked against official sources.",
            "Literal null-like strings are treated as missing; descriptive text and original source URLs are retained.",
            "Natural-language eligibility prose is supporting context and is not converted into unverified deterministic rules.",
        ],
    }


def write_quality_report(path: Path = DATA_PATH, output: Path = REPORT_PATH) -> dict[str, Any]:
    report = build_quality_report(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
