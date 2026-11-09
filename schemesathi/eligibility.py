from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any


@dataclass
class EligibilityResult:
    outcome: str
    satisfied: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)


def _same_state(profile_state: str, candidate: str) -> bool:
    return profile_state.strip().casefold() == candidate.strip().casefold()


OCCUPATION_TYPES = {
    "farmer": {"farmer", "farmers", "agricultural worker", "cultivator"},
    "student": {"student", "students", "pupil"},
    "unemployed": {"unemployed", "job seeker"},
    "business owner": {"entrepreneur", "business owner", "self-employed"},
    "worker": {"worker", "workers", "labourer", "laborer"},
    "fisher": {"fisher", "fisherman", "fisherwoman", "fisherfolk"},
    "senior citizen": {"senior citizen", "senior citizens", "pensioner", "pensioners"},
}
SOCIAL_CATEGORY_ALIASES = {
    "general": "general",
    "sc": "sc",
    "scheduled caste": "sc",
    "scheduled castes": "sc",
    "st": "st",
    "scheduled tribe": "st",
    "scheduled tribes": "st",
    "obc": "obc",
    "other backward class": "obc",
    "other backward classes": "obc",
    "ews": "ews",
    "economically weaker section": "ews",
    "minority community": "minority community",
    "other": "other",
}


def evaluate_eligibility(scheme: dict[str, Any], profile: dict[str, Any]) -> EligibilityResult:
    satisfied: list[str] = []
    failed: list[str] = []
    unknown: list[str] = []
    rules_checked = 0

    def compare(field: str, condition: str, passes: bool) -> None:
        (satisfied if passes else failed).append(f"{field}: {condition}")

    age_min = scheme.get("eligibility_age_min")
    age_max = scheme.get("eligibility_age_max")
    age_rule = age_min is not None or age_max is not None
    age = profile.get("age")
    if age_rule:
        rules_checked += 1
        if age is None:
            unknown.append("Age is needed to check the scheme's age range.")
        else:
            low_ok = age_min is None or float(age) >= float(age_min)
            high_ok = age_max is None or float(age) <= float(age_max)
            condition = f"{age_min if age_min is not None else 'no minimum'}–{age_max if age_max is not None else 'no maximum'} years"
            compare("Age", condition, low_ok and high_ok)

    income_max = scheme.get("eligibility_income_max")
    if income_max is not None:
        rules_checked += 1
        income = profile.get("annual_income")
        if income is None:
            unknown.append("Annual family income is needed to check the scheme's income limit.")
        else:
            compare("Annual family income", f"at or below {income_max:g}", float(income) <= float(income_max))

    allowed_states = scheme.get("eligibility_state")
    scheme_state = (scheme.get("state") or "").strip()
    if allowed_states and "All" not in allowed_states:
        rules_checked += 1
        state = profile.get("state")
        if not state:
            unknown.append("State is needed to check this state-specific scheme.")
        elif not any(_same_state(state, option) for option in allowed_states):
            failed.append(f"State: available in {', '.join(allowed_states)}")
        else:
            satisfied.append(f"State: available in {state}")
    elif scheme_state.casefold() == "central":
        if profile.get("state"):
            satisfied.append("State: Central scheme; available across states, subject to other conditions")
        else:
            unknown.append("State is not provided; confirm your state to check scheme availability.")
    elif scheme_state and scheme_state.casefold() != "all":
        rules_checked += 1
        state = profile.get("state")
        if not state:
            unknown.append("State is needed to check scheme availability.")
        elif _same_state(state, scheme_state):
            satisfied.append(f"State: available in {scheme_state}")
        else:
            failed.append(f"State: scheme is listed for {scheme_state}")

    required_gender = str(scheme.get("eligibility_gender") or "").strip().casefold()
    if required_gender and required_gender not in {"all", "any", "both"}:
        rules_checked += 1
        gender = str(profile.get("gender") or "").strip().casefold()
        if not gender:
            unknown.append("Gender is needed to check the scheme's gender condition.")
        elif gender != required_gender:
            failed.append(f"Gender: scheme specifies {required_gender}")
        else:
            satisfied.append(f"Gender: matches {required_gender}")

    required_residence = str(scheme.get("eligibility_residence") or "").strip().casefold()
    if required_residence and required_residence not in {"both", "all", "any"}:
        rules_checked += 1
        residence = str(profile.get("residence") or "").strip().casefold()
        if not residence:
            unknown.append("Rural or urban residence is needed to check this condition.")
        elif residence != required_residence:
            failed.append(f"Residence: scheme specifies {required_residence}")
        else:
            satisfied.append(f"Residence: matches {required_residence}")

    for key, profile_key, label in (
        ("eligibility_disability", "disability", "disability"),
        ("eligibility_bpl", "bpl", "BPL status"),
    ):
        if scheme.get(key) is True:
            rules_checked += 1
            value = profile.get(profile_key)
            if value is None:
                unknown.append(f"{label} information is needed to check this condition.")
            elif value is True:
                satisfied.append(f"{label}: condition met")
            else:
                failed.append(f"{label}: scheme requires this condition")

    castes = scheme.get("eligibility_caste")
    if castes:
        rules_checked += 1
        social_category = profile.get("social_category") or profile.get("caste")
        specific_community = profile.get("specific_community")
        selected_category = specific_community or social_category
        specific_requirement = any(
            str(item).strip().casefold() not in SOCIAL_CATEGORY_ALIASES for item in castes
        )
        if not selected_category:
            if specific_requirement:
                unknown.append(
                    "A specific caste/community name is needed for this scheme; you can provide it for this check."
                )
            else:
                unknown.append("Social category is needed to check this scheme condition.")
        elif any(
            SOCIAL_CATEGORY_ALIASES.get(str(selected_category).strip().casefold(), str(selected_category).strip().casefold())
            == SOCIAL_CATEGORY_ALIASES.get(str(item).strip().casefold(), str(item).strip().casefold())
            for item in castes
        ):
            satisfied.append("Social category: matches a listed category")
        elif (
            social_category
            and not specific_community
            and specific_requirement
        ):
            unknown.append(
                "A specific caste/community name is needed for this scheme; you can provide it for this check."
            )
        else:
            failed.append(f"Social category: scheme lists {', '.join(castes)}")

    beneficiary_types = {
        re.sub(r"\s+", " ", item).strip().casefold()
        for item in str(scheme.get("beneficiary_type") or "").split(",")
    }
    relevant_occupations = {
        occupation
        for occupation, labels in OCCUPATION_TYPES.items()
        if beneficiary_types.intersection(labels)
    }
    eligibility_text = str(scheme.get("eligibility_text") or "").casefold()
    for occupation, labels in OCCUPATION_TYPES.items():
        labels_pattern = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
        if re.search(
            rf"\b(?:applicant|beneficiary|candidate|individual)\b[^.\n]{{0,100}}"
            rf"\b(?:must|should|shall|has to|is required to)\b[^.\n]{{0,40}}"
            rf"\b(?:be\s+(?:an?\s+)?)?(?:{labels_pattern})\b",
            eligibility_text,
        ):
            relevant_occupations.add(occupation)
    if relevant_occupations:
        rules_checked += 1
        occupation = str(profile.get("occupation") or "").strip().casefold()
        if not occupation:
            unknown.append("Occupation is needed to check the scheme's stated beneficiary type.")
        elif any(
            occupation in labels or any(label in occupation for label in labels)
            for role, labels in OCCUPATION_TYPES.items()
            if role in relevant_occupations
        ):
            satisfied.append(f"Occupation: matches {', '.join(sorted(relevant_occupations))}")
        else:
            failed.append(f"Occupation: scheme identifies {', '.join(sorted(relevant_occupations))}")

    if not rules_checked and not failed and not unknown:
        unknown.append("No structured eligibility conditions were supplied; review the scheme's eligibility description.")
    outcome = "not_matching" if failed else ("needs_information" if unknown else "potentially_eligible")
    return EligibilityResult(outcome, satisfied, failed, unknown)
