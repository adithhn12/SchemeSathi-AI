from __future__ import annotations

import re
from typing import Any


STATE_NAMES = (
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal", "Andaman and Nicobar Islands", "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Jammu and Kashmir", "Ladakh",
    "Lakshadweep", "Puducherry",
)
OCCUPATIONS = {
    "farmer": ("farmer", "farming", "agriculture", "cultivator"),
    "student": ("student", "studying", "school", "college"),
    "unemployed": ("unemployed", "jobless", "looking for work"),
    "business owner": ("business owner", "entrepreneur", "self employed", "self-employed"),
    "worker": ("worker", "labourer", "laborer", "daily wage"),
    "fisher": ("fisher", "fisherman", "fisherwoman", "fishing"),
    "senior citizen": ("senior citizen", "retired", "pensioner"),
}
EVENTS = {
    "marriage": ("marriage", "wedding", "married"),
    "childbirth": ("childbirth", "newborn", "new baby", "pregnant", "maternity"),
    "education": ("education", "admission", "scholarship", "school fees"),
    "unemployment": ("unemployment", "lost my job", "job loss", "unemployed"),
    "disability": ("disability", "disabled", "differently abled"),
    "retirement": ("retirement", "retired", "pension"),
    "death of breadwinner": ("breadwinner died", "breadwinner death", "family member died"),
    "business or agriculture": ("start a business", "new business", "start farming", "agricultural activity"),
}


def extract_profile_details(text: str) -> dict[str, Any]:
    """Extract explicit details; ambiguous matches are returned for confirmation."""
    result: dict[str, Any] = {"values": {}, "needs_confirmation": []}
    age_matches = re.findall(r"\b(?:i am|i'm|age(?:d)?|turned)\s*(\d{1,3})\b|\b(\d{1,3})\s*years?\s*old\b", text, re.I)
    ages = {int(left or right) for left, right in age_matches if (left or right).isdigit() and 0 < int(left or right) < 121}
    if len(ages) == 1:
        result["values"]["age"] = ages.pop()
    elif ages:
        result["needs_confirmation"].append("I found more than one possible age; please confirm your age.")

    income_matches = re.findall(
        r"(?:annual|yearly|family)?\s*(?:income|earnings)\s*(?:is|of|:|around|about)?\s*(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d+)?)\s*(lakh| lakhs| lac| crores?|k)?",
        text,
        re.I,
    )
    incomes: set[int] = set()
    for amount_text, unit in income_matches:
        amount = float(amount_text.replace(",", ""))
        suffix = (unit or "").strip().casefold()
        if suffix in {"lakh", "lakhs", "lac"}:
            amount *= 100_000
        elif suffix in {"crore", "crores"}:
            amount *= 10_000_000
        elif suffix == "k":
            amount *= 1_000
        incomes.add(int(amount))
    if len(incomes) == 1:
        result["values"]["annual_income"] = incomes.pop()
    elif incomes:
        result["needs_confirmation"].append("I found more than one possible income amount; please confirm annual family income.")

    education = re.search(
        r"\b(?:studying|pursuing|education level is|in class|in grade)\s+"
        r"(?:in\s+)?([a-z0-9][a-z0-9 -]{1,35}?)(?=[,.!?;]|$)",
        text,
        re.I,
    )
    if education:
        result["values"]["education"] = education.group(1).strip()

    matches = [state for state in STATE_NAMES if re.search(rf"\b{re.escape(state)}\b", text, re.I)]
    if len(matches) == 1:
        result["values"]["state"] = matches[0]
    elif matches:
        result["needs_confirmation"].append("I found more than one state; please confirm your current state.")

    lowered = text.casefold()
    occupations = [
        occupation for occupation, phrases in OCCUPATIONS.items()
        if any(phrase in lowered for phrase in phrases)
    ]
    if len(occupations) == 1:
        result["values"]["occupation"] = occupations[0]
    elif len(occupations) > 1:
        result["needs_confirmation"].append("I found more than one possible occupation; please confirm which applies.")
    if re.search(r"\brural\b", lowered):
        result["values"]["residence"] = "rural"
    elif re.search(r"\burban\b|\bcity\b", lowered):
        result["values"]["residence"] = "urban"
    if re.search(r"\b(?:female|woman|women)\b", lowered):
        result["values"]["gender"] = "female"
    elif re.search(r"\b(?:male|man|men)\b", lowered):
        result["values"]["gender"] = "male"
    disability_match = re.search(
        r"\b(?:i am not disabled|i do not have a disability|no disability)\b", lowered
    )
    if disability_match:
        result["values"]["disability"] = False
    elif re.search(r"\b(?:i am disabled|i have a disability|have a disability|person with a disability)\b", lowered):
        result["values"]["disability"] = True
    if re.search(r"\b(?:not bpl|not below poverty line|above poverty line|non[- ]bpl)\b", lowered):
        result["values"]["bpl"] = False
    elif re.search(r"\b(?:bpl|below poverty line)\b", lowered):
        result["values"]["bpl"] = True
    for event, phrases in EVENTS.items():
        if any(phrase in lowered for phrase in phrases):
            result["values"]["life_event"] = event
            break
    return result
