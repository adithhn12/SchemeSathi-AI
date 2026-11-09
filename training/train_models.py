from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from schemesathi.ml import INTENTS, INTENT_MODEL_PATH


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_PATH = ROOT / "training" / "intent_examples.csv"
METRICS_PATH = ROOT / "reports" / "model_evaluation.json"

EXAMPLES = {
    "scheme_discovery": [
        "Find government schemes I can apply for",
        "What benefits are available for a farmer in Karnataka?",
        "Show schemes for my profile",
        "Help me discover government assistance",
        "Which government benefits may suit my family?",
        "Find scholarships and welfare schemes near me",
        "I want to know what schemes I may qualify for",
        "Show me relevant central and state schemes",
    ],
    "eligibility": [
        "Am I eligible for this scheme?",
        "Check whether my age and income meet the rules",
        "Why do I not match the eligibility conditions?",
        "What requirements am I missing?",
        "Does my state qualify for this benefit?",
        "Check my scheme eligibility",
        "Explain passed and failed conditions",
        "Can I apply based on my family income?",
    ],
    "family_benefits": [
        "Find benefits for everyone in my household",
        "Add my spouse to family benefits",
        "Show schemes for my children and parents",
        "Map government benefits across my family",
        "What can my family members apply for?",
        "Check benefits separately for each relative",
        "I want to add a family member",
        "Summarize our household assistance options",
    ],
    "life_events": [
        "I recently had a baby",
        "I am getting married and need assistance",
        "I lost my job, what help is available?",
        "Find support after the death of our breadwinner",
        "I am retiring soon",
        "My child is starting college",
        "I became disabled and need support",
        "I want to start a small business",
    ],
    "documents": [
        "Which documents are required for this scheme?",
        "Show my application document checklist",
        "I have my income certificate, what is still missing?",
        "Mark this document as available",
        "What paperwork do I need to apply?",
        "Check my document readiness",
        "List the certificates required",
        "I do not have a caste certificate",
    ],
    "application": [
        "How do I apply for the scheme?",
        "Open the official application website",
        "Where should I submit my application?",
        "Explain the application steps",
        "Can I apply online?",
        "Give me the scheme application link",
        "What is the process to receive this benefit?",
        "How can I contact the department?",
    ],
    "scheme_information": [
        "Tell me about this scheme",
        "What benefits does it provide?",
        "Explain the scheme details",
        "Who runs this program?",
        "What is the purpose of this government benefit?",
        "Give me information about the scholarship",
        "What does this scheme offer?",
        "Summarize this scheme for me",
    ],
    "alternatives": [
        "I do not meet the age limit, show alternatives",
        "Find another scheme with different conditions",
        "What other assistance can I get instead?",
        "Suggest alternatives to this benefit",
        "I failed the income rule, what else is available?",
        "Show schemes that may fit me better",
        "Find similar programs with looser requirements",
        "What is my next best option?",
    ],
    "comparison": [
        "Compare these two schemes",
        "Which scheme offers the better benefit?",
        "Show differences in eligibility and documents",
        "Compare the application procedures",
        "I want a side by side scheme comparison",
        "What is different between these programs?",
        "Compare benefits for these schemes",
        "Which one has fewer requirements?",
    ],
    "general": [
        "Hello there",
        "Thank you for helping me",
        "What can this bot do?",
        "I have a question",
        "Good morning",
        "Help",
        "How does this service work?",
        "I do not understand",
    ],
}


def main() -> None:
    if set(EXAMPLES) != set(INTENTS):
        raise ValueError("Every configured intent must have labeled examples.")
    rows = [(text, label) for label, texts in EXAMPLES.items() for text in texts]
    EXAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EXAMPLES_PATH.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["text", "intent"])
        writer.writerows(rows)

    texts = [row[0] for row in rows]
    labels = [row[1] for row in rows]
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True)),
            ("classifier", LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=42)),
        ]
    )
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predicted, labels=list(INTENTS), average="macro", zero_division=0
    )
    report = {
        "model": "TF-IDF + Logistic Regression",
        "training_examples": len(x_train),
        "test_examples": len(x_test),
        "intent_labels": list(INTENTS),
        "accuracy": accuracy_score(y_test, predicted),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "confusion_matrix_labels": list(INTENTS),
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=list(INTENTS)).tolist(),
        "evaluation_note": "Single stratified holdout of a small, hand-labeled starter set; metrics are not a production-quality estimate.",
    }
    INTENT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, INTENT_MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "confusion_matrix"}, indent=2))


if __name__ == "__main__":
    main()
