from __future__ import annotations

import html
import logging
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from .data import load_schemes
from .eligibility import EligibilityResult, evaluate_eligibility
from .extraction import extract_profile_details
from .ml import INTENT_MODEL_PATH, SchemeRecommender, classify_intent
from .storage import Store


ROOT = Path(__file__).resolve().parents[1]
PROFILE_FIELDS = (
    "age",
    "occupation",
    "education",
    "annual_income",
    "state",
    "residence",
    "gender",
    "social_category",
)
SOCIAL_CATEGORIES = (
    "General",
    "SC",
    "ST",
    "OBC",
    "EWS",
    "Minority community",
    "Other",
    "Prefer not to say",
)
NAVIGATION_ACTIONS = frozenset(
    {
        "home",
        "discover",
        "family",
        "event",
        "eligibility",
        "documents",
        "alternatives",
        "compare",
        "ask",
        "profile",
        "add_family",
        "delete",
        "confirm_delete",
        "family_edit",
        "edit_social_category",
    }
)
LEGACY_CALLBACK_ACTIONS = {
    "menu": "nav",
    "docs": "document_checklist",
    "doc": "document_status",
    "alternative": "alternatives",
    "page": "recommendations_page",
}
FIELD_PROMPTS = {
    "age": "What is your age in years? (or type skip)",
    "occupation": "What best describes your occupation? (or type skip)",
    "education": "What is your current or highest education level? (or type skip)",
    "annual_income": "What is your approximate annual family income in INR? (or type skip)",
    "state": "Which Indian state or union territory do you live in? (or type skip)",
    "residence": "Is your residence rural or urban? (or type skip)",
    "gender": "What gender should I use when checking scheme conditions? (or type skip)",
    "social_category": (
        "Optional and sensitive: share a social category only if you want it used to check a "
        "scheme's listed conditions. It is stored locally, not inferred, and is not your religion. "
        "Choose a category or skip."
    ),
}
MAIN_MENU = InlineKeyboardMarkup(
    [
        [InlineKeyboardButton("Discover schemes", callback_data="nav:discover"),
         InlineKeyboardButton("Family benefits", callback_data="nav:family")],
        [InlineKeyboardButton("Life event", callback_data="nav:event"),
         InlineKeyboardButton("Check eligibility", callback_data="nav:eligibility")],
        [InlineKeyboardButton("Document checklist", callback_data="nav:documents"),
         InlineKeyboardButton("Alternatives", callback_data="nav:alternatives")],
        [InlineKeyboardButton("Compare schemes", callback_data="nav:compare"),
         InlineKeyboardButton("Ask about a scheme", callback_data="nav:ask")],
        [InlineKeyboardButton("Edit profile", callback_data="nav:profile"),
         InlineKeyboardButton("Delete my data", callback_data="nav:delete")],
    ]
)


def _clean_text(value: Any, limit: int = 900) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]*>", " ", text)
    for marker in ("**", "__", "~~", "`"):
        text = text.replace(marker, "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text).strip()
    return text[:limit].strip()


def _escape(value: Any, limit: int = 900) -> str:
    return html.escape(_clean_text(value, limit))


def _short_summary(value: Any, limit: int = 180) -> str:
    text = _clean_text(value, limit + 1)
    first_sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    return (first_sentence[:limit].rstrip() + ("…" if len(first_sentence) > limit else "")) or "Not provided."


def _valid_http_url(value: Any) -> str | None:
    text = str(value or "").strip()
    try:
        parsed = urlparse(text)
    except ValueError:
        return None
    return text if parsed.scheme in {"http", "https"} and parsed.netloc else None


def _profile_ready(profile: dict[str, Any]) -> bool:
    return bool(profile.get("state") and profile.get("occupation"))


def _check_label(result: EligibilityResult) -> str:
    return {
        "potentially_eligible": "Potential match",
        "not_matching": "Known condition mismatch",
        "needs_information": "More details needed",
    }[result.outcome]


def _documents_for(scheme: dict[str, Any]) -> list[str]:
    raw = scheme.get("documents_required")
    documents = [
        re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", item).strip()
        for item in re.split(r"[\n;]+", str(raw or ""))
        if item.strip()
    ]
    if scheme.get("eligibility_caste") and not any(
        re.search(r"caste|community|social category", item, re.IGNORECASE) for item in documents
    ):
        documents = documents[:11]
        documents.append("Caste/community certificate (confirm the exact scheme requirement)")
    return documents[:12]


def _scheme_card(item: dict[str, Any]) -> str:
    scheme = item["scheme"]
    eligibility: EligibilityResult = item["eligibility"]
    if eligibility.outcome == "not_matching":
        condition_values = eligibility.failed or eligibility.satisfied or eligibility.unknown
        why = "A known condition differs: " + eligibility.failed[0]
    elif eligibility.satisfied:
        condition_values = eligibility.satisfied
        why = "Your profile matches: " + eligibility.satisfied[0]
    else:
        condition_values = eligibility.unknown or eligibility.failed
        why = "Please confirm: " + (eligibility.unknown[0] if eligibility.unknown else "scheme conditions are incomplete.")
    lines = [
        f"<b>{_escape(scheme.get('name'), 180)}</b>",
        f"<i>{_escape(scheme.get('state') or 'All India')} · {_escape(scheme.get('category') or 'Scheme', 100)}</i>",
        f"Benefit: {_escape(_short_summary(scheme.get('benefits')), 220)}",
        f"Status: <b>{_escape(_check_label(eligibility), 120)}</b>",
    ]
    if condition_values:
        lines.append("Key conditions: " + _escape("; ".join(condition_values[:3]), 300))
    lines.append("Why it may fit: " + _escape(why, 200))
    return "\n".join(lines)


def _scheme_details(item: dict[str, Any]) -> str:
    scheme = item["scheme"]
    eligibility: EligibilityResult = item["eligibility"]
    documents = _documents_for(scheme)
    apply_url = _valid_http_url(scheme.get("apply_url"))
    info_url = _valid_http_url(scheme.get("official_url"))
    lines = [
        f"<b>{_escape(scheme.get('name'), 180)}</b>",
        f"<b>Benefits</b>\n{_escape(_clean_text(scheme.get('benefits'), 650) or 'Not listed in this dataset.')}",
        f"<b>Eligibility</b>\n{_escape(_clean_text(scheme.get('eligibility_text'), 700) or 'Not listed in this dataset.')}",
        f"<b>Your check</b>\n{_escape(_check_label(eligibility))}",
    ]
    if eligibility.unknown:
        lines.append("<b>Still to check</b>\n" + _escape("; ".join(eligibility.unknown[:4]), 500))
    if eligibility.failed:
        lines.append("<b>Conditions that do not match</b>\n" + _escape("; ".join(eligibility.failed[:4]), 500))
    lines.append("<b>Documents</b>\n" + (_escape("; ".join(documents), 550) if documents else "Not listed in this dataset."))
    lines.append(
        "<b>Application steps</b>\n"
        + (_escape(_clean_text(scheme.get("application_process"), 550)) if scheme.get("application_process") else "Not listed in this dataset.")
    )
    if apply_url:
        lines.append(f'<a href="{html.escape(apply_url, quote=True)}">Open the application page</a>')
    elif info_url:
        lines.append("An application link was not supplied.")
        lines.append(f'<a href="{html.escape(info_url, quote=True)}">Open the scheme information page</a>')
    else:
        lines.append("Application link: not supplied in this dataset.")
    lines.append(
        "Source: supplied third-party dataset. Not independently verified"
        + (f"; dataset timestamp: {_escape(scheme.get('scraped_at'), 60)}." if scheme.get("scraped_at") else ".")
        + " Confirm current rules with the responsible department."
    )
    return "\n\n".join(lines)


def _scheme_keyboard(scheme: dict[str, Any], offset: int) -> InlineKeyboardMarkup:
    slug = scheme["slug"]
    apply_url = _valid_http_url(scheme.get("apply_url"))
    apply_button = (
        InlineKeyboardButton("Apply Officially", url=apply_url)
        if apply_url
        else InlineKeyboardButton("Apply Officially", callback_data=f"scheme_apply:{slug}:{offset}")
    )
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("View Details", callback_data=f"scheme_details:{slug}:{offset}"),
                InlineKeyboardButton("Check Documents", callback_data=f"document_checklist:{slug}:{offset}"),
            ],
            [
                apply_button,
                InlineKeyboardButton("Find Alternatives", callback_data=f"alternatives:{slug}:{offset}"),
            ],
            [InlineKeyboardButton("Back to Results", callback_data=f"recommendations_return:{offset}")],
        ]
    )


def _details_keyboard(
    scheme: dict[str, Any], offset: int, request_specific_community: bool = False
) -> InlineKeyboardMarkup:
    slug = scheme["slug"]
    apply_url = _valid_http_url(scheme.get("apply_url"))
    apply_button = (
        InlineKeyboardButton("Apply Officially", url=apply_url)
        if apply_url
        else InlineKeyboardButton("Apply Officially", callback_data=f"scheme_apply:{slug}:{offset}")
    )
    rows = [
        [
            InlineKeyboardButton("Check Documents", callback_data=f"document_checklist:{slug}:{offset}"),
            InlineKeyboardButton("Find Alternatives", callback_data=f"alternatives:{slug}:{offset}"),
        ],
        [apply_button],
    ]
    if request_specific_community:
        rows.append(
            [
                InlineKeyboardButton(
                    "Add detail for this check",
                    callback_data=f"profile_specific:{slug}:{offset}",
                )
            ]
        )
    rows.append(
        [
            InlineKeyboardButton("Back to Results", callback_data=f"recommendations_return:{offset}"),
            InlineKeyboardButton("Main Menu", callback_data="nav:home"),
        ]
    )
    return InlineKeyboardMarkup(rows)


class SchemeSathiBot:
    def __init__(self, store: Store | None = None, schemes: list[dict[str, Any]] | None = None) -> None:
        self.store = store or Store()
        self.schemes = schemes if schemes is not None else load_schemes()
        self.recommender = SchemeRecommender(self.schemes)
        self.by_slug = {item["slug"]: item for item in self.schemes}
        self._processed_callback_ids: OrderedDict[str, None] = OrderedDict()

    def _reserve_callback(self, callback_id: str) -> bool:
        if callback_id in self._processed_callback_ids:
            return False
        self._processed_callback_ids[callback_id] = None
        if len(self._processed_callback_ids) > 2048:
            self._processed_callback_ids.popitem(last=False)
        return True

    def _social_category_keyboard(self) -> InlineKeyboardMarkup:
        rows = [
            [
                InlineKeyboardButton(category, callback_data=f"profile_category:{category}")
                for category in SOCIAL_CATEGORIES[index : index + 2]
            ]
            for index in range(0, len(SOCIAL_CATEGORIES), 2)
        ]
        rows.append([InlineKeyboardButton("Skip", callback_data="profile_category:skip")])
        return InlineKeyboardMarkup(rows)

    async def _prompt_profile_field(self, message: Any, field: str) -> None:
        if field == "social_category":
            await message.reply_text(
                FIELD_PROMPTS[field],
                reply_markup=self._social_category_keyboard(),
            )
        elif field == "residence":
            await message.reply_text(
                "Is your home in a rural or urban area? You may skip.",
                reply_markup=InlineKeyboardMarkup(
                    [[
                        InlineKeyboardButton("Rural", callback_data="profile_choice:residence:rural"),
                        InlineKeyboardButton("Urban", callback_data="profile_choice:residence:urban"),
                    ]]
                ),
            )
        elif field == "gender":
            await message.reply_text(
                "Optional: what gender should I use when checking a scheme's conditions? You may skip.",
                reply_markup=InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton("Female", callback_data="profile_choice:gender:female"),
                            InlineKeyboardButton("Male", callback_data="profile_choice:gender:male"),
                        ],
                        [
                            InlineKeyboardButton("Other", callback_data="profile_choice:gender:other"),
                            InlineKeyboardButton("Skip", callback_data="profile_choice:gender:skip"),
                        ],
                    ]
                ),
            )
        else:
            await message.reply_text(FIELD_PROMPTS[field])

    def _save_social_category(self, user_id: int, value: str) -> None:
        if value == "skip":
            return
        profile = self.store.get_profile(user_id)
        profile.pop("caste", None)
        if value == "Prefer not to say":
            profile.pop("social_category", None)
        else:
            profile["social_category"] = value
        self.store.replace_profile(user_id, profile)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.effective_message.reply_text(
            "Welcome to <b>SchemeSathi AI</b>\n<i>Your benefits. Your eligibility. Your rights.</i>\n\n"
            "I help explore Indian Central and State scheme records. Dataset listings are not official verification or a guarantee of eligibility. "
            "I do not need Aadhaar, OTPs, passwords, or document numbers.\n\nChoose an option:",
            parse_mode=ParseMode.HTML,
            reply_markup=MAIN_MENU,
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.effective_message.reply_text(
            "/start — open the main menu\n/profile — view your saved profile\n/reset — stop the current conversation flow\n/delete_profile — permanently delete your saved profile, family and checklist data\n/about — service and data limitations\n\n"
            "Use the buttons for scheme discovery, eligibility, family mapping, life events, document readiness, alternatives and comparisons. "
            "Share only information you are comfortable storing locally."
        )

    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self.store.clear_conversation(update.effective_user.id)
        await update.effective_message.reply_text("Current conversation stopped. Your saved profile and family data were not changed.", reply_markup=MAIN_MENU)

    async def profile(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        profile = self.store.get_profile(update.effective_user.id)
        fields = [
            f"{key.replace('_', ' ').title()}: "
            f"{profile.get(key) if profile.get(key) is not None else 'not provided'}"
            for key in (*PROFILE_FIELDS, "disability", "bpl", "life_event")
        ]
        await update.effective_message.reply_text(
            "Your saved profile:\n" + "\n".join(fields),
            reply_markup=InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton("Edit profile", callback_data="nav:profile")],
                    [InlineKeyboardButton("Edit social category", callback_data="nav:edit_social_category")],
                    [InlineKeyboardButton("Main menu", callback_data="nav:home")],
                ]
            ),
        )

    async def delete_profile(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self.store.delete_profile(update.effective_user.id)
        await update.effective_message.reply_text("Your saved profile, family members, document checklist and conversation state have been deleted.", reply_markup=MAIN_MENU)

    async def about(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.effective_message.reply_text(
            "SchemeSathi AI — Bridging the gap between government benefits and the people who need them.\n\n"
            f"Loaded {len(self.schemes):,} records from the supplied third-party CSV. Records and links have not been independently verified against official sources. "
            f"Intent model: {'trained model available' if INTENT_MODEL_PATH.is_file() else 'not trained yet'}. "
            "Recommendations use text relevance; they are not eligibility probabilities. Eligibility checks are deterministic checks of populated structured fields only. "
            "Missing details remain unknown. No paid AI API is used."
        )

    async def callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if not query:
            return
        if not self._reserve_callback(str(query.id)):
            await query.answer("This action was already processed.")
            return
        data = query.data or ""
        action, separator, value = data.partition(":")
        route = LEGACY_CALLBACK_ACTIONS.get(action, action)
        parts = data.split(":")
        valid = separator and (
            (route == "nav" and value in NAVIGATION_ACTIONS)
            or (
                route == "document_checklist"
                and len(parts) in {2, 3}
                and parts[1] in self.by_slug
                and (len(parts) == 2 or parts[2].isdigit())
            )
            or (route == "eligibility_check" and value in self.by_slug)
            or (
                route == "alternatives"
                and len(parts) in {2, 3}
                and parts[1] in self.by_slug
                and (len(parts) == 2 or parts[2].isdigit())
            )
            or (route == "recommendations_page" and value.isdigit())
            or (route == "recommendations_return" and value.isdigit())
            or (
                route in {"scheme_details", "scheme_apply"}
                and len(parts) == 3
                and parts[1] in self.by_slug
                and parts[2].isdigit()
            )
            or (
                route == "profile_specific"
                and len(parts) == 3
                and parts[1] in self.by_slug
                and parts[2].isdigit()
            )
            or (route == "profile_category" and value in {*SOCIAL_CATEGORIES, "skip"})
            or (
                route == "profile_choice"
                and len(parts) == 3
                and (
                    (parts[1] == "residence" and parts[2] in {"rural", "urban"})
                    or (parts[1] == "gender" and parts[2] in {"female", "male", "other", "skip"})
                )
            )
            or (
                route == "family_benefits"
                and value.isdigit()
                and int(value) <= len(self.store.get_family(update.effective_user.id))
            )
            or (
                route == "family_edit"
                and value.isdigit()
                and any(
                    str(member["id"]) == value
                    for member in self.store.get_family(update.effective_user.id)
                )
            )
            or (
                route == "document_status"
                and len(parts) in {4, 5}
                and parts[1] in self.by_slug
                and parts[2].isdigit()
                and int(parts[2]) < len(_documents_for(self.by_slug[parts[1]]))
                and parts[3] in {"available", "missing", "unknown"}
                and (len(parts) == 4 or parts[4].isdigit())
            )
        )
        await query.answer(
            None if valid else "This button is no longer available.",
            show_alert=not valid,
        )
        if not valid:
            return
        user_id = update.effective_user.id
        if route == "nav":
            await self._menu_action(query, user_id, value)
        elif route == "document_checklist":
            self.store.clear_conversation(user_id)
            offset = int(parts[2]) if len(parts) == 3 else 0
            await self._document_checklist(query.message, user_id, parts[1], offset)
        elif route == "eligibility_check":
            self.store.clear_conversation(user_id)
            await self._show_scheme_eligibility(query.message, user_id, self.by_slug[value])
        elif route == "document_status":
            _, slug, index_text, status, *offset_value = parts
            self.store.clear_conversation(user_id)
            document = _documents_for(self.by_slug[slug])[int(index_text)]
            self.store.set_document_status(user_id, slug, document, status)
            await self._document_checklist(
                query.message,
                user_id,
                slug,
                int(offset_value[0]) if offset_value else 0,
            )
        elif route == "alternatives":
            self.store.clear_conversation(user_id)
            offset = int(parts[2]) if len(parts) == 3 else 0
            await self._alternatives(query.message, user_id, parts[1], offset)
        elif route == "scheme_details":
            await self._show_scheme_details(query.message, user_id, parts[1], int(parts[2]))
        elif route == "scheme_apply":
            await self._show_apply_information(query.message, parts[1], int(parts[2]))
        elif route == "profile_specific":
            self.store.set_conversation(
                user_id,
                "specific_community",
                {"slug": parts[1], "offset": int(parts[2])},
            )
            await query.message.reply_text(
                "This scheme lists a specific caste or community condition. If you want to check it, "
                "type the exact name you choose to provide. It will be used only for this check and "
                "will not be saved to your profile. Type skip to continue without it."
            )
        elif route == "recommendations_return":
            await self._recommendation_page(query, user_id, value)
        elif route == "profile_category":
            state, data = self.store.get_conversation(user_id)
            if state == "profile":
                await self._process_profile_value(
                    query.message,
                    user_id,
                    "skip" if value == "skip" else value,
                    data,
                )
            elif state == "family":
                await self._process_family_value(
                    query.message,
                    user_id,
                    "skip" if value == "skip" else value,
                    data,
                )
            elif state == "social_category_edit":
                self._save_social_category(user_id, value)
                self.store.clear_conversation(user_id)
                await query.message.reply_text(
                    "Social category preference updated. You can change or remove it at any time.",
                    reply_markup=MAIN_MENU,
                )
            else:
                await query.message.reply_text("That profile question has expired. Open Edit profile to try again.")
        elif route == "profile_choice":
            state, data = self.store.get_conversation(user_id)
            selected = parts[2]
            if state == "profile":
                await self._process_profile_value(
                    query.message,
                    user_id,
                    "skip" if selected == "skip" else selected,
                    data,
                )
            elif state == "family":
                await self._process_family_value(
                    query.message,
                    user_id,
                    "skip" if selected == "skip" else selected,
                    data,
                )
            else:
                await query.message.reply_text("That profile question has expired. Open Edit profile to try again.")
        elif route == "family_benefits":
            self.store.clear_conversation(user_id)
            await self._show_family_member_benefits(query.message, user_id, int(value))
        elif route == "family_edit":
            self.store.clear_conversation(user_id)
            await self._start_family_edit(query.message, user_id, int(value))
        elif route == "recommendations_page":
            await self._recommendation_page(query, user_id, value)

    async def _menu_action(self, message: Any, user_id: int, action: str) -> None:
        target = getattr(message, "message", message)
        self.store.clear_conversation(user_id)
        if action in {"home", ""}:
            await target.reply_text("Choose an option:", reply_markup=MAIN_MENU)
        elif action == "profile":
            self.store.set_conversation(user_id, "profile", {"index": 0, "profile": self.store.get_profile(user_id)})
            await self._prompt_profile_field(target, PROFILE_FIELDS[0])
        elif action == "edit_social_category":
            self.store.set_conversation(user_id, "social_category_edit")
            await target.reply_text(
                FIELD_PROMPTS["social_category"],
                reply_markup=self._social_category_keyboard(),
            )
        elif action == "add_family":
            self.store.set_conversation(user_id, "family", {"index": 0, "profile": {}})
            await target.reply_text("What should I call this family member?")
        elif action == "family_edit":
            await self._edit_household(target, user_id)
        elif action == "family":
            await self._family_summary(target, user_id, self.store.get_profile(user_id))
        elif action == "documents":
            self.store.set_conversation(user_id, "document_lookup")
            await target.reply_text(
                "Type a scheme name to open its document-readiness checklist. You can also choose “Mark document readiness” on a scheme result."
            )
        elif action == "eligibility":
            profile = self.store.get_profile(user_id)
            if not _profile_ready(profile):
                self.store.set_conversation(user_id, "profile", {"index": 0, "profile": profile, "return_to": action})
                await target.reply_text("I need your state and occupation to make useful, location-aware suggestions. Let's update your profile.\n\n" + FIELD_PROMPTS[PROFILE_FIELDS[0]])
                return
            await self._show_recommendations(
                target,
                user_id,
                profile,
                "government assistance benefits for my circumstances",
                eligibility_only=True,
            )
        elif action in {"discover", "alternatives"}:
            profile = self.store.get_profile(user_id)
            if not _profile_ready(profile):
                self.store.set_conversation(user_id, "profile", {"index": 0, "profile": profile, "return_to": action})
                await target.reply_text("I need your state and occupation to make useful, location-aware suggestions. Let's update your profile.\n\n" + FIELD_PROMPTS[PROFILE_FIELDS[0]])
                return
            if action == "discover":
                await self._show_recommendations(target, user_id, profile, "government assistance benefits for my circumstances")
            else:
                await self._show_recommendations(target, user_id, profile, "alternative assistance schemes", alternatives=True)
        elif action == "event":
            self.store.set_conversation(user_id, "event", {})
            await target.reply_text("What life event should I consider? Examples: marriage, childbirth, education, unemployment, disability, retirement, death of a breadwinner, starting a business or farming.")
        elif action == "compare":
            self.store.set_conversation(user_id, "compare", {})
            await target.reply_text("Type two or three scheme names (separated by commas) to compare.")
        elif action == "ask":
            self.store.set_conversation(user_id, "ask", {})
            await target.reply_text("Ask a question about a scheme, benefit, eligibility description, documents or application process.")
        elif action == "delete":
            await target.reply_text("Delete your saved profile, family members and document checklist?", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Yes, delete my data", callback_data="nav:confirm_delete"), InlineKeyboardButton("Cancel", callback_data="nav:home")]]))
        elif action == "confirm_delete":
            self.store.delete_profile(user_id)
            await target.reply_text("Your saved profile, family members, document checklist and conversation state have been deleted.", reply_markup=MAIN_MENU)

    async def text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        text = (update.effective_message.text or "").strip()
        state, data = self.store.get_conversation(user_id)
        if state == "profile":
            await self._collect_profile(update, user_id, text, data)
        elif state == "family":
            await self._collect_family(update, user_id, text, data)
        elif state == "social_category_edit":
            selected = next(
                (category for category in SOCIAL_CATEGORIES[:-1] if category.casefold() == text.casefold()),
                None,
            )
            if selected is None and text.casefold() not in {"skip", "prefer not to say"}:
                await update.effective_message.reply_text(
                    FIELD_PROMPTS["social_category"],
                    reply_markup=self._social_category_keyboard(),
                )
                return
            self._save_social_category(
                user_id,
                "skip" if selected is None else selected,
            )
            self.store.clear_conversation(user_id)
            await update.effective_message.reply_text(
                "Social category preference updated.",
                reply_markup=MAIN_MENU,
            )
        elif state == "specific_community":
            if text.casefold() in {"skip", "prefer not to say"}:
                self.store.clear_conversation(user_id)
                await update.effective_message.reply_text(
                    "No specific community detail was used. You can still review the scheme information."
                )
                await self._show_scheme_details(
                    update.effective_message,
                    user_id,
                    str(data["slug"]),
                    int(data["offset"]),
                )
                return
            community = _clean_text(text, 100)
            if not community:
                await update.effective_message.reply_text("Please enter a name or type skip.")
                return
            self.store.clear_conversation(user_id)
            profile = self._profile_for_scheme(user_id, str(data["slug"]))
            profile["specific_community"] = community
            await self._show_scheme_details(
                update.effective_message,
                user_id,
                str(data["slug"]),
                int(data["offset"]),
                profile_override=profile,
            )
        elif state == "confirm_extraction":
            await self._confirm_extraction(update, user_id, text, data)
        elif state == "document_lookup":
            await self._select_document_scheme(update, user_id, text)
        elif state == "eligibility_lookup":
            await self._select_eligibility_scheme(update, user_id, text)
        elif state == "event":
            self.store.clear_conversation(user_id)
            profile = self.store.get_profile(user_id)
            profile["life_event"] = text
            await self._show_recommendations(update.effective_message, user_id, profile, text)
        elif state == "compare":
            self.store.clear_conversation(user_id)
            await self._compare(update.effective_message, text)
        elif state == "ask":
            self.store.clear_conversation(user_id)
            await self._answer_question(update.effective_message, text)
        else:
            extracted = extract_profile_details(text)
            if extracted["values"]:
                self.store.update_profile(user_id, extracted["values"])
            if extracted["needs_confirmation"]:
                confirmation = extracted["needs_confirmation"][0]
                field = next(
                    (name for name in ("age", "state", "occupation") if name in confirmation.casefold()),
                    None,
                )
                if field:
                    prompts = {
                        "age": "Please send the single age to use, for example: I am 35.",
                        "state": "Please send your current state or union territory name.",
                        "occupation": "Please send the one occupation that applies to you.",
                    }
                    self.store.set_conversation(user_id, "confirm_extraction", {"field": field})
                    await update.effective_message.reply_text(prompts[field])
                    return
                await update.effective_message.reply_text("\n".join(extracted["needs_confirmation"]))
                return
            try:
                intent = classify_intent(text)
            except FileNotFoundError:
                intent = "general"
            intent_actions = {
                "scheme_discovery": ("Discover schemes", "discover"),
                "eligibility": ("Check eligibility", "eligibility"),
                "family_benefits": ("Family benefits", "family"),
                "life_events": ("Explore a life event", "event"),
                "documents": ("Document checklist", "documents"),
                "application": ("Ask about a scheme", "ask"),
                "scheme_information": ("Ask about a scheme", "ask"),
                "alternatives": ("Find alternatives", "alternatives"),
                "comparison": ("Compare schemes", "compare"),
                "general": ("Open the main menu", "home"),
            }
            action_label, action = intent_actions.get(intent, intent_actions["general"])
            await update.effective_message.reply_text(
                f"I think this is about {intent.replace('_', ' ')}. Choose how to continue:",
                reply_markup=InlineKeyboardMarkup(
                    [
                        [InlineKeyboardButton(action_label, callback_data=f"nav:{action}")],
                        [InlineKeyboardButton("Open full menu", callback_data="nav:home")],
                    ]
                ),
            )

    async def _confirm_extraction(self, update: Update, user_id: int, text: str, data: dict[str, Any]) -> None:
        field = data.get("field")
        value: Any = None
        if field == "age":
            numbers = re.findall(r"\d+", text)
            if len(numbers) == 1 and 1 <= int(numbers[0]) <= 120:
                value = int(numbers[0])
        elif field == "state":
            value = extract_profile_details(text)["values"].get("state")
        elif field == "occupation":
            value = extract_profile_details(text)["values"].get("occupation")
        if value is None:
            await update.effective_message.reply_text("I couldn't confirm that detail. Please try again or use /reset to stop.")
            return
        self.store.update_profile(user_id, {field: value})
        self.store.clear_conversation(user_id)
        await update.effective_message.reply_text(
            f"Confirmed {field.replace('_', ' ')}: {value}. Choose an option to continue.",
            reply_markup=MAIN_MENU,
        )

    async def _select_document_scheme(self, update: Update, user_id: int, text: str) -> None:
        matches = self.recommender.search_by_name(text, limit=6)
        if not matches:
            await update.effective_message.reply_text(
                "I couldn't find that scheme name. Try its exact name from a result, or use /reset to stop."
            )
            return
        self.store.clear_conversation(user_id)
        profile = self.store.get_profile(user_id)
        self.store.set_recommendation_page(
            user_id,
            text,
            profile,
            "documents",
            [item["slug"] for item in matches],
        )
        if len(matches) == 1:
            await self._document_checklist(update.effective_message, user_id, matches[0]["slug"])
            return
        buttons = [
            [InlineKeyboardButton(str(item.get("name") or item["slug"])[:60], callback_data=f"document_checklist:{item['slug']}")]
            for item in matches
        ]
        await update.effective_message.reply_text(
            "I found several matching schemes. Choose one for its checklist:",
            reply_markup=InlineKeyboardMarkup(buttons),
        )

    async def _select_eligibility_scheme(self, update: Update, user_id: int, text: str) -> None:
        matches = self.recommender.search_by_name(text, limit=6)
        if not matches:
            await update.effective_message.reply_text(
                "I couldn't find that scheme name. Try its exact name from a result, or use /reset to stop."
            )
            return
        self.store.clear_conversation(user_id)
        profile = self.store.get_profile(user_id)
        self.store.set_recommendation_page(
            user_id,
            text,
            profile,
            "eligibility",
            [item["slug"] for item in matches],
        )
        if len(matches) == 1:
            await self._show_scheme_eligibility(update.effective_message, user_id, matches[0])
            return
        buttons = [
            [InlineKeyboardButton(str(item.get("name") or item["slug"])[:60], callback_data=f"eligibility_check:{item['slug']}")]
            for item in matches
        ]
        await update.effective_message.reply_text(
            "I found several matching schemes. Choose one to check against your profile:",
            reply_markup=InlineKeyboardMarkup(buttons),
        )

    async def _show_scheme_eligibility(self, message: Any, user_id: int, scheme: dict[str, Any]) -> None:
        profile = self.store.get_profile(user_id)
        if scheme["slug"] not in self.store.get_recommendation_slugs(user_id):
            self.store.set_recommendation_page(
                user_id,
                str(scheme.get("name") or scheme["slug"]),
                profile,
                "eligibility",
                [scheme["slug"]],
            )
        result = evaluate_eligibility(scheme, profile)
        await message.reply_text(
            _scheme_card({"scheme": scheme, "eligibility": result}),
            parse_mode=ParseMode.HTML,
            reply_markup=_scheme_keyboard(scheme, 0),
            disable_web_page_preview=True,
        )

    async def _collect_profile(self, update: Update, user_id: int, text: str, data: dict[str, Any]) -> None:
        await self._process_profile_value(update.effective_message, user_id, text, data)

    async def _process_profile_value(
        self, message: Any, user_id: int, text: str, data: dict[str, Any]
    ) -> None:
        index = int(data.get("index", 0))
        field = PROFILE_FIELDS[index]
        value: Any = None
        if text.casefold() not in {"skip", "unknown", "prefer not to say"}:
            if field == "age":
                match = re.search(r"\d{1,3}", text)
                if not match or not 1 <= int(match.group()) <= 120:
                    await message.reply_text("Please enter an age from 1 to 120, or type skip.")
                    return
                value = int(match.group())
            elif field == "annual_income":
                digits = re.sub(r"[^\d]", "", text)
                if not digits:
                    await message.reply_text("Please enter annual income in INR, or type skip.")
                    return
                value = int(digits)
            elif field == "residence":
                if text.casefold() not in {"rural", "urban"}:
                    await message.reply_text("Please enter rural or urban, or type skip.")
                    return
                value = text.casefold()
            elif field == "state":
                value = text.strip().title()
            elif field == "social_category":
                selected = next(
                    (category for category in SOCIAL_CATEGORIES[:-1] if category.casefold() == text.casefold()),
                    None,
                )
                if selected is None:
                    await self._prompt_profile_field(message, field)
                    return
                value = selected
            else:
                value = text.strip()[:100]
        profile = data.get("profile", self.store.get_profile(user_id))
        if field == "social_category":
            if text.casefold() == "prefer not to say":
                profile.pop("caste", None)
                profile.pop(field, None)
            elif value is not None:
                profile.pop("caste", None)
                profile[field] = value
        elif value is not None:
            profile[field] = value
        index += 1
        if index < len(PROFILE_FIELDS):
            data.update({"index": index, "profile": profile})
            self.store.set_conversation(user_id, "profile", data)
            await self._prompt_profile_field(message, PROFILE_FIELDS[index])
            return
        self.store.replace_profile(user_id, profile)
        self.store.clear_conversation(user_id)
        return_to = data.get("return_to")
        await message.reply_text(
            "Profile saved. You can edit it or delete it at any time using /delete_profile.",
            reply_markup=MAIN_MENU,
        )
        if return_to in {"discover", "eligibility", "alternatives", "documents", "family"}:
            await self._menu_action(message, user_id, return_to)

    async def _collect_family(self, update: Update, user_id: int, text: str, data: dict[str, Any]) -> None:
        await self._process_family_value(update.effective_message, user_id, text, data)

    async def _process_family_value(
        self, message: Any, user_id: int, text: str, data: dict[str, Any]
    ) -> None:
        fields = ("name",) + PROFILE_FIELDS
        index = int(data.get("index", 0))
        field = fields[index]
        profile = data.get("profile", {})
        if field == "name":
            if not text.strip():
                await message.reply_text("Please enter a first name or family label.")
                return
            data["name"] = text[:80]
        elif text.casefold() not in {"skip", "unknown", "prefer not to say"}:
            if field == "age":
                match = re.search(r"\d{1,3}", text)
                if not match or not 1 <= int(match.group()) <= 120:
                    await message.reply_text("Please enter an age from 1 to 120, or type skip.")
                    return
                profile[field] = int(match.group())
            elif field == "annual_income":
                digits = re.sub(r"[^\d]", "", text)
                if not digits:
                    await message.reply_text("Please enter annual income in INR, or type skip.")
                    return
                profile[field] = int(digits)
            elif field == "residence":
                if text.casefold() not in {"rural", "urban"}:
                    await message.reply_text("Please enter rural or urban, or type skip.")
                    return
                profile[field] = text.casefold()
            elif field == "social_category":
                selected = next(
                    (category for category in SOCIAL_CATEGORIES[:-1] if category.casefold() == text.casefold()),
                    None,
                )
                if selected is None:
                    await self._prompt_profile_field(message, field)
                    return
                profile[field] = selected
            else:
                profile[field] = text.strip().title() if field == "state" else text.strip()[:100]
        elif field == "social_category" and (
            text.casefold() == "prefer not to say" or data.get("member_id") is None
        ):
            profile.pop("social_category", None)
            profile.pop("caste", None)
        index += 1
        if index < len(fields):
            data.update({"index": index, "profile": profile})
            self.store.set_conversation(user_id, "family", data)
            next_field = fields[index]
            if next_field == "name":
                await message.reply_text("What should I call this family member?")
            else:
                await self._prompt_profile_field(message, next_field)
            return
        member_id = data.get("member_id")
        if member_id is None:
            self.store.add_family_member(user_id, data["name"], profile)
        elif not self.store.update_family_member(user_id, int(member_id), data["name"], profile):
            self.store.clear_conversation(user_id)
            await message.reply_text("That household member is no longer available to edit.")
            return
        self.store.clear_conversation(user_id)
        await message.reply_text(
            f"Saved {data['name']}. Household details are updated.",
            reply_markup=MAIN_MENU,
        )
        await self._family_summary(message, user_id, self.store.get_profile(user_id))

    async def _show_recommendations(
        self,
        message: Any,
        user_id: int,
        profile: dict[str, Any],
        query: str,
        eligibility_only: bool = False,
        alternatives: bool = False,
        with_documents: bool = False,
        offset: int = 0,
    ) -> None:
        mode = "alternatives" if alternatives else "eligibility" if eligibility_only else "documents" if with_documents else "discover"
        outcomes = {"potentially_eligible", "needs_information"} if alternatives else None
        results = self.recommender.recommend(
            profile,
            query,
            limit=len(self.schemes),
            outcomes=outcomes,
        )
        if with_documents:
            results = [result for result in results if result["scheme"].get("documents_required")]
        self.store.set_recommendation_page(
            user_id,
            query,
            profile,
            mode,
            [result["scheme"]["slug"] for result in results],
        )
        if not results:
            await message.reply_text("No matching records were found for that request. Try a broader description or check the dataset later.", reply_markup=MAIN_MENU)
            return
        await self._render_recommendation_page(message, results, offset, mode)

    async def _render_recommendation_page(
        self,
        message: Any,
        results: list[dict[str, Any]],
        offset: int,
        mode: str,
    ) -> None:
        page_results = results[max(0, offset) : max(0, offset) + 3]
        if not page_results:
            await message.reply_text("That results page is no longer available.", reply_markup=MAIN_MENU)
            return
        if mode == "alternatives":
            title = "Alternatives to consider"
        elif mode == "documents":
            title = "Schemes with listed documents"
        else:
            title = "Scheme matches"
        await message.reply_text(
            f"<b>{title}</b>\nShowing {offset + 1}–{offset + len(page_results)} of {len(results)} matches.",
            parse_mode=ParseMode.HTML,
        )
        for result in page_results:
            await message.reply_text(
                _scheme_card(result),
                parse_mode=ParseMode.HTML,
                reply_markup=_scheme_keyboard(result["scheme"], offset),
                disable_web_page_preview=True,
            )
        page_buttons: list[InlineKeyboardButton] = []
        if offset:
            page_buttons.append(
                InlineKeyboardButton("Previous", callback_data=f"recommendations_page:{max(0, offset - 3)}")
            )
        if offset + 3 < len(results):
            page_buttons.append(
                InlineKeyboardButton("Next", callback_data=f"recommendations_page:{offset + 3}")
            )
        navigation_rows = [page_buttons] if page_buttons else []
        navigation_rows.append([InlineKeyboardButton("Main Menu", callback_data="nav:home")])
        await message.reply_text(
            "Choose a page or return to the menu.",
            reply_markup=InlineKeyboardMarkup(navigation_rows),
        )

    async def _recommendation_page(self, query: Any, user_id: int, value: str) -> None:
        try:
            offset = int(value)
        except ValueError:
            await query.message.reply_text("Invalid results page.")
            return
        page = self.store.get_recommendation_page(user_id)
        if not page:
            await query.message.reply_text("This results list has expired. Start a new search from the main menu.")
            return
        _, profile, mode = page
        slugs = self.store.get_recommendation_slugs(user_id)
        results = [
            {
                "scheme": self.by_slug[slug],
                "eligibility": evaluate_eligibility(self.by_slug[slug], profile),
                "relevance_score": 0.0,
            }
            for slug in slugs
            if slug in self.by_slug
        ]
        if offset < 0 or offset % 3 or offset >= len(results):
            await query.message.reply_text("That results page is no longer available.")
            return
        await self._render_recommendation_page(query.message, results, offset, mode)

    async def _family_summary(self, message: Any, user_id: int, profile: dict[str, Any]) -> None:
        family = self.store.get_family(user_id)
        members = [{"name": "You", **profile}, *family]
        lines = ["<b>Your Household</b>"]
        rows: list[list[InlineKeyboardButton]] = []
        for index, member in enumerate(members):
            details = [
                str(member.get("occupation") or "Occupation not provided"),
                f"Age {member['age']}" if member.get("age") is not None else "Age not provided",
            ]
            lines.append(f"• {_escape(member['name'], 80)} — {_escape(', '.join(details), 120)}")
            rows.append(
                [
                    InlineKeyboardButton(
                        f"View {member['name'][:35]}'s benefits",
                        callback_data=f"family_benefits:{index}",
                    )
                ]
            )
        rows.extend(
            [
                [InlineKeyboardButton("Add Family Member", callback_data="nav:add_family")],
                [InlineKeyboardButton("Edit Household", callback_data="nav:family_edit")],
                [InlineKeyboardButton("Main Menu", callback_data="nav:home")],
            ]
        )
        if not family:
            lines.append("\nYou are the only saved household member. Would you like to add others?")
        else:
            lines.append("\nChoose a member to view their benefits separately.")
        await message.reply_text(
            "\n".join(lines),
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(rows),
        )

    async def _show_family_member_benefits(
        self, message: Any, user_id: int, member_index: int
    ) -> None:
        profile = self.store.get_profile(user_id)
        members = [{"name": "You", **profile}, *self.store.get_family(user_id)]
        if not 0 <= member_index < len(members):
            await message.reply_text("That household member is no longer available.")
            return
        member = members[member_index]
        await message.reply_text(f"<b>Benefits for {_escape(member['name'], 80)}</b>", parse_mode=ParseMode.HTML)
        query = f"{member.get('occupation') or ''} {member.get('state') or ''} government benefits"
        await self._show_recommendations(
            message,
            user_id,
            member,
            query,
        )

    async def _edit_household(self, message: Any, user_id: int) -> None:
        family = self.store.get_family(user_id)
        rows = [
            [InlineKeyboardButton("Edit your profile", callback_data="nav:profile")]
        ]
        rows.extend(
            [
                InlineKeyboardButton(f"Edit {member['name']}", callback_data=f"family_edit:{member['id']}")
            ]
            for member in family
        )
        rows.extend(
            [
                [InlineKeyboardButton("Add Family Member", callback_data="nav:add_family")],
                [InlineKeyboardButton("Main Menu", callback_data="nav:home")],
            ]
        )
        await message.reply_text(
            "Choose a household profile to edit. Type “skip” to keep an existing answer.",
            reply_markup=InlineKeyboardMarkup(rows),
        )

    async def _start_family_edit(self, message: Any, user_id: int, member_id: int) -> None:
        member = next(
            (item for item in self.store.get_family(user_id) if item["id"] == member_id),
            None,
        )
        if not member:
            await message.reply_text("That household member is no longer available.")
            return
        profile = {key: value for key, value in member.items() if key not in {"id", "name"}}
        self.store.set_conversation(
            user_id,
            "family",
            {
                "index": 0,
                "member_id": member_id,
                "name": member["name"],
                "profile": profile,
            },
        )
        await message.reply_text("What name or family label should I use?")

    async def _document_checklist(
        self, message: Any, user_id: int, slug: str, offset: int = 0
    ) -> None:
        scheme = self.by_slug.get(slug)
        if not scheme:
            await message.reply_text("That scheme record is no longer available.")
            return
        documents = _documents_for(scheme)
        if not documents:
            await message.reply_text(
                "The dataset has no document list for this scheme. Check the responsible department's official guidance.",
                reply_markup=InlineKeyboardMarkup(
                    [
                        [InlineKeyboardButton("Back to Results", callback_data=f"recommendations_return:{offset}")],
                        [InlineKeyboardButton("Main Menu", callback_data="nav:home")],
                    ]
                ),
            )
            return
        statuses = self.store.get_document_statuses(user_id, slug)
        rows = []
        lines = [f"<b>Document readiness — {_escape(scheme['name'], 150)}</b>"]
        missing = []
        for index, document in enumerate(documents):
            status = statuses.get(document, "unknown")
            lines.append(f"{index + 1}. {_escape(document, 140)} — <b>{status}</b>")
            if status != "available":
                missing.append(document)
            rows.append(
                [
                    InlineKeyboardButton(f"{index + 1}: Available", callback_data=f"document_status:{slug}:{index}:available:{offset}"),
                    InlineKeyboardButton("Missing", callback_data=f"document_status:{slug}:{index}:missing:{offset}"),
                    InlineKeyboardButton("Unknown", callback_data=f"document_status:{slug}:{index}:unknown:{offset}"),
                ]
            )
        lines.append(f"\nReadiness: {len(documents) - len(missing)}/{len(documents)} marked available.")
        if missing:
            lines.append("Still to confirm or prepare: " + _escape(", ".join(missing), 400))
        rows.extend(
            [
                [InlineKeyboardButton("Back to Results", callback_data=f"recommendations_return:{offset}")],
                [InlineKeyboardButton("Main Menu", callback_data="nav:home")],
            ]
        )
        await message.reply_text(
            "\n".join(lines),
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(rows),
            disable_web_page_preview=True,
        )

    async def _show_scheme_details(
        self,
        message: Any,
        user_id: int,
        slug: str,
        offset: int,
        profile_override: dict[str, Any] | None = None,
    ) -> None:
        scheme = self.by_slug.get(slug)
        if not scheme:
            await message.reply_text("That scheme record is no longer available.")
            return
        profile = profile_override or self._profile_for_scheme(user_id, slug)
        item = {
            "scheme": scheme,
            "eligibility": evaluate_eligibility(
                scheme,
                profile,
            ),
            "relevance_score": None,
        }
        request_specific = any(
            "specific caste/community" in condition
            for condition in item["eligibility"].unknown
        )
        await message.reply_text(
            _scheme_details(item),
            parse_mode=ParseMode.HTML,
            reply_markup=_details_keyboard(scheme, offset, request_specific),
            disable_web_page_preview=True,
        )

    def _profile_for_scheme(self, user_id: int, slug: str) -> dict[str, Any]:
        page = self.store.get_recommendation_page(user_id)
        if page and slug in self.store.get_recommendation_slugs(user_id):
            return page[1]
        return self.store.get_profile(user_id)

    async def _show_apply_information(
        self, message: Any, slug: str, offset: int
    ) -> None:
        scheme = self.by_slug.get(slug)
        if not scheme:
            await message.reply_text("That scheme record is no longer available.")
            return
        info_url = _valid_http_url(scheme.get("official_url"))
        text = (
            "An official application link was not supplied in this dataset. "
            "Please confirm the application route with the responsible department."
        )
        if info_url:
            text += f'\n\n<a href="{html.escape(info_url, quote=True)}">Open scheme information</a>'
        await message.reply_text(
            text,
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton("Back to Results", callback_data=f"recommendations_return:{offset}")],
                    [InlineKeyboardButton("Main Menu", callback_data="nav:home")],
                ]
            ),
            disable_web_page_preview=True,
        )

    async def _alternatives(self, message: Any, user_id: int, slug: str, offset: int = 0) -> None:
        profile = self.store.get_profile(user_id)
        original = self.by_slug.get(slug)
        if not original:
            await message.reply_text("That scheme record is no longer available.")
            return
        results = self.recommender.recommend(
            profile,
            f"{original.get('category') or ''} {original.get('beneficiary_type') or ''} {original.get('description') or ''}",
            limit=5,
            outcomes={"potentially_eligible", "needs_information"},
        )
        results = [item for item in results if item["scheme"]["slug"] != slug][:3]
        if not results:
            await message.reply_text(
                "No alternatives with matching or unknown structured conditions were found in this dataset.",
                reply_markup=InlineKeyboardMarkup(
                    [
                        [InlineKeyboardButton("Back to Results", callback_data=f"recommendations_return:{offset}")],
                        [InlineKeyboardButton("Main Menu", callback_data="nav:home")],
                    ]
                ),
            )
            return
        await message.reply_text(
            "<b>Alternatives to consider</b>\nNo known structured condition failed; unknown conditions still need checking.",
            parse_mode=ParseMode.HTML,
        )
        for item in results:
            await message.reply_text(
                _scheme_card(item),
                parse_mode=ParseMode.HTML,
                reply_markup=_scheme_keyboard(item["scheme"], offset),
                disable_web_page_preview=True,
            )
        await message.reply_text(
            "Return to your original matches or choose another section.",
            reply_markup=InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton("Back to Results", callback_data=f"recommendations_return:{offset}")],
                    [InlineKeyboardButton("Main Menu", callback_data="nav:home")],
                ]
            ),
        )

    async def _compare(self, message: Any, text: str) -> None:
        queries = [item.strip() for item in text.split(",") if item.strip()][:3]
        if len(queries) < 2:
            await message.reply_text("Please provide at least two scheme names separated by commas.")
            return
        schemes: list[dict[str, Any]] = []
        for name in queries:
            normalized = name.casefold()
            match = next(
                (
                    scheme
                    for scheme in self.schemes
                    if normalized in str(scheme.get("name") or "").casefold()
                ),
                None,
            )
            if match:
                schemes.append(match)
        if len(schemes) < 2:
            await message.reply_text("I could not find two scheme records to compare. Try more distinctive names.")
            return
        for scheme in schemes:
            text = (
                f"<b>{_escape(scheme.get('name'))}</b>\n"
                f"Benefits: {_escape(scheme.get('benefits'), 600) or 'Not provided'}\n"
                f"Eligibility: {_escape(scheme.get('eligibility_text'), 600) or 'Not provided'}\n"
                f"Documents: {_escape(scheme.get('documents_required'), 500) or 'Not provided'}\n"
                f"Application: {_escape(scheme.get('application_process'), 500) or 'Not provided'}\n"
                f"Information URL: {_escape(scheme.get('official_url'), 450) or 'Not provided'}\n"
                "Source: third-party dataset record; not independently verified."
            )
            await message.reply_text(text, parse_mode=ParseMode.HTML, disable_web_page_preview=True)

    async def _answer_question(self, message: Any, text: str) -> None:
        matches = self.recommender.faq_matches(text, limit=1)
        if not matches or matches[0][1] <= 0:
            await message.reply_text("I couldn't find a relevant scheme description in this dataset. Try the scheme's exact name or check an official department source.", reply_markup=MAIN_MENU)
            return
        scheme, score = matches[0]
        answer = (
            f"<b>{_escape(scheme.get('name'))}</b>\n"
            f"Description: {_escape(scheme.get('description'), 900) or 'Not provided'}\n"
            f"Benefits: {_escape(scheme.get('benefits'), 650) or 'Not provided'}\n"
            f"Eligibility description: {_escape(scheme.get('eligibility_text'), 650) or 'Not provided'}\n"
            f"Documents: {_escape(scheme.get('documents_required'), 450) or 'Not provided'}\n"
            f"Application process: {_escape(scheme.get('application_process'), 700) or 'Not provided'}\n"
            f"Information: {_escape(scheme.get('official_url'), 450) or 'Not provided'}\n"
            f"Text match score: {score:.3f} (not an eligibility probability). Dataset record not independently verified."
        )
        await message.reply_text(answer, parse_mode=ParseMode.HTML, disable_web_page_preview=True)


def create_application() -> Application:
    # Keep this project's configured bot identity authoritative over any stale
    # TELEGRAM_BOT_TOKEN inherited from the shell that launched the process.
    load_dotenv(ROOT / ".env", override=True)
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing. Set it in the project-root .env file.")
    service = SchemeSathiBot()
    application = Application.builder().token(token).build()
    application.bot_data["service"] = service

    async def invoke(method: str, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await getattr(service, method)(update, context)

    application.add_handler(CommandHandler("start", lambda u, c: invoke("start", u, c)))
    application.add_handler(CommandHandler("help", lambda u, c: invoke("help_command", u, c)))
    application.add_handler(CommandHandler("reset", lambda u, c: invoke("reset", u, c)))
    application.add_handler(CommandHandler("profile", lambda u, c: invoke("profile", u, c)))
    application.add_handler(CommandHandler("delete_profile", lambda u, c: invoke("delete_profile", u, c)))
    application.add_handler(CommandHandler("about", lambda u, c: invoke("about", u, c)))
    application.add_handler(CallbackQueryHandler(service.callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, service.text_message))
    return application


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    application = create_application()
    application.run_polling(allowed_updates=Update.ALL_TYPES)
