from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .eligibility import EligibilityResult, evaluate_eligibility


ROOT = Path(__file__).resolve().parents[1]
INTENT_MODEL_PATH = ROOT / "models" / "intent_classifier.joblib"
INTENTS = (
    "scheme_discovery",
    "eligibility",
    "family_benefits",
    "life_events",
    "documents",
    "application",
    "scheme_information",
    "alternatives",
    "comparison",
    "general",
)


@lru_cache(maxsize=1)
def _intent_model():
    if not INTENT_MODEL_PATH.is_file():
        raise FileNotFoundError("Intent model is not trained. Run: python -m training.train_models")
    import joblib

    return joblib.load(INTENT_MODEL_PATH)


def classify_intent(text: str) -> str:
    return str(_intent_model().predict([text])[0])


def _scheme_text(scheme: dict[str, Any]) -> str:
    return " ".join(
        str(scheme.get(key) or "")
        for key in ("name", "description", "benefits", "eligibility_text", "category", "beneficiary_type", "state")
    )


class SchemeRecommender:
    def __init__(self, schemes: list[dict[str, Any]]) -> None:
        self.schemes = schemes
        self.vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            ngram_range=(1, 2),
            max_features=160_000,
            sublinear_tf=True,
            stop_words="english",
        )
        self.matrix = self.vectorizer.fit_transform([_scheme_text(item) for item in schemes])

    def recommend(
        self,
        profile: dict[str, Any],
        query: str = "",
        limit: int = 8,
        outcomes: set[str] | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        profile_terms = " ".join(
            str(profile.get(key) or "")
            for key in ("occupation", "state", "residence", "education", "life_event")
        )
        query_text = " ".join(part for part in (query, profile_terms) if part).strip()
        if query_text:
            similarities = cosine_similarity(self.vectorizer.transform([query_text]), self.matrix).ravel()
        else:
            similarities = [0.0] * len(self.schemes)

        ranked: list[dict[str, Any]] = []
        for index, scheme in enumerate(self.schemes):
            eligibility = evaluate_eligibility(scheme, profile)
            if outcomes and eligibility.outcome not in outcomes:
                continue
            score = float(similarities[index])
            if eligibility.failed:
                score -= 0.12
            elif eligibility.outcome == "potentially_eligible":
                score += 0.06
            state = str(profile.get("state") or "").casefold()
            scheme_state = str(scheme.get("state") or "").casefold()
            allowed = scheme.get("eligibility_state") or []
            if state and (scheme_state == "central" or not allowed or "All" in allowed or any(state == item.casefold() for item in allowed)):
                score += 0.12
            elif state and scheme_state and scheme_state != "central" and scheme_state != "all":
                score -= 0.18
            ranked.append(
                {
                    "scheme": scheme,
                    "relevance_score": round(max(0.0, score), 4),
                    "eligibility": eligibility,
                }
            )
        ranked.sort(key=lambda result: (result["relevance_score"], -len(result["eligibility"].failed)), reverse=True)
        return ranked[max(0, offset) : max(0, offset) + limit]

    def search_by_name(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        normalized = re.sub(r"\s+", " ", query.casefold()).strip()
        direct = [scheme for scheme in self.schemes if normalized in str(scheme.get("name") or "").casefold()]
        return direct[:limit]

    def faq_matches(self, query: str, limit: int = 3) -> list[tuple[dict[str, Any], float]]:
        vector = self.vectorizer.transform([query])
        scores = cosine_similarity(vector, self.matrix).ravel()
        indexes = scores.argsort()[::-1][:limit]
        return [(self.schemes[int(index)], float(scores[int(index)])) for index in indexes]
