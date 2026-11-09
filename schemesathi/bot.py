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
from .i18n import LANGUAGES, LANGUAGE_NAMES, translate
from .ml import SchemeRecommender, classify_intent
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
    "disability",
    "bpl",
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
DOCUMENTS_PER_PAGE = 4
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
        "language",
    }
)
LEGACY_CALLBACK_ACTIONS = {
    "menu": "nav",
    "docs": "document_checklist",
    "doc": "document_status",
    "alternative": "alternatives",
    "page": "recommendations_page",
}
def _main_menu(language: str = "en") -> InlineKeyboardMarkup:
    t = lambda key: translate(language, key)
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton(t("menu.discover"), callback_data="nav:discover"),
             InlineKeyboardButton(t("menu.family"), callback_data="nav:family")],
            [InlineKeyboardButton(t("menu.event"), callback_data="nav:event"),
             InlineKeyboardButton(t("menu.eligibility"), callback_data="nav:eligibility")],
            [InlineKeyboardButton(t("menu.documents"), callback_data="nav:documents"),
             InlineKeyboardButton(t("menu.alternatives"), callback_data="nav:alternatives")],
            [InlineKeyboardButton(t("menu.compare"), callback_data="nav:compare"),
             InlineKeyboardButton(t("menu.ask"), callback_data="nav:ask")],
            [InlineKeyboardButton(t("menu.profile"), callback_data="nav:profile"),
             InlineKeyboardButton(t("menu.language"), callback_data="nav:language")],
            [InlineKeyboardButton(t("menu.delete"), callback_data="nav:delete")],
        ]
    )


def _language_keyboard(language: str = "en") -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(("✅ " if code == language else "") + name, callback_data=f"language:{code}")]
        for code, name in LANGUAGE_NAMES.items()
    ]
    rows.append([InlineKeyboardButton(translate(language, "home"), callback_data="nav:home")])
    return InlineKeyboardMarkup(rows)


def _normalize_menu_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", value.casefold(), flags=re.UNICODE)).strip()


MAIN_MENU = _main_menu()


def _clean_text(value: Any, limit: int = 900) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"(?m)^\s*>\s?", "", text)
    for marker in ("**", "__", "~~", "`"):
        text = text.replace(marker, "")
    text = text.replace("*", "")
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


def _check_label(result: EligibilityResult, language: str = "en") -> str:
    key = {
        "potentially_eligible": "profile.all_pass",
        "not_matching": "profile.mismatch",
        "needs_information": "profile.unknown",
    }[result.outcome]
    return translate(language, key)


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


def _scheme_card(item: dict[str, Any], language: str = "en") -> str:
    t = lambda key, **values: translate(language, key, **values)
    scheme = item["scheme"]
    eligibility: EligibilityResult = item["eligibility"]
    source_url = _valid_http_url(scheme.get("official_url"))
    status = (
        f"{t('profile.all_pass')} · {t('label.count', count=str(len(eligibility.satisfied)))}"
        if eligibility.outcome == "potentially_eligible"
        else t("profile.mismatch") if eligibility.outcome == "not_matching"
        else t("profile.unknown")
    )
    lines = [
        f"<b>{_escape(scheme.get('name'), 180)}</b>",
        f"📍 <i>{_escape(scheme.get('state') or 'All India')} · {_escape(scheme.get('category') or 'Scheme', 100)}</i>",
        f"💰 {_escape(_short_summary(scheme.get('benefits'), 100), 130)}",
        status,
    ]
    if source_url:
        lines.append(f'<a href="{html.escape(source_url, quote=True)}">{t("source.link")}</a>')
    else:
        lines.append(t("source.missing"))
    return "\n".join(lines)


def _scheme_details(item: dict[str, Any], language: str = "en") -> str:
    t = lambda key: translate(language, key)
    scheme = item["scheme"]
    eligibility: EligibilityResult = item["eligibility"]
    documents = _documents_for(scheme)
    apply_url = _valid_http_url(scheme.get("apply_url"))
    info_url = _valid_http_url(scheme.get("official_url"))
    lines = [
        f"<b>{_escape(scheme.get('name'), 180)}</b>",
        f"<b>{t('label.benefits')}</b>\n{_escape(_clean_text(scheme.get('benefits'), 320) or t('label.not_listed'))}",
        f"<b>{t('label.eligibility')}</b>\n{_escape(_clean_text(scheme.get('eligibility_text'), 420) or t('label.not_listed'))}",
        f"<b>{t('label.your_check')}</b>\n{_escape(_check_label(eligibility, language))}",
    ]
    if eligibility.unknown:
        lines.append(f"<b>{t('label.still_check')}</b>\n" + _escape("; ".join(eligibility.unknown[:4]), 500))
    if eligibility.failed:
        lines.append(f"<b>{t('label.mismatch')}</b>\n" + _escape("; ".join(eligibility.failed[:4]), 500))
    brief_documents = documents[:6]
    if len(documents) > 6:
        brief_documents.append(f"… +{len(documents) - 6}")
    lines.append(f"<b>{t('label.documents')}</b>\n" + (_escape("; ".join(brief_documents), 300) if documents else t("label.not_listed")))
    lines.append(
        f"<b>{t('label.application')}</b>\n"
        + (_escape(_clean_text(scheme.get("application_process"), 300)) if scheme.get("application_process") else t("label.not_listed"))
    )
    if apply_url:
        lines.append(f'<a href="{html.escape(apply_url, quote=True)}">{t("label.open_application")}</a>')
    elif info_url:
        lines.append(t("label.application_missing"))
        lines.append(f'<a href="{html.escape(info_url, quote=True)}">{t("label.open_information")}</a>')
    else:
        lines.append(t("label.no_application"))
    if info_url:
        lines.append(f'<a href="{html.escape(info_url, quote=True)}">{t("source.link")}</a>')
    lines.append(
        t("label.source")
        + (f"; dataset timestamp: {_escape(scheme.get('scraped_at'), 60)}." if scheme.get("scraped_at") else ".")
        + " " + t("label.source_confirm")
    )
    return "\n\n".join(lines)


def _scheme_keyboard(scheme: dict[str, Any], offset: int, language: str = "en") -> InlineKeyboardMarkup:
    slug = scheme["slug"]
    apply_url = _valid_http_url(scheme.get("apply_url"))
    apply_button = (
        InlineKeyboardButton(translate(language, "apply"), url=apply_url)
        if apply_url
        else InlineKeyboardButton(translate(language, "apply"), callback_data=f"scheme_apply:{slug}:{offset}")
    )
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(translate(language, "details"), callback_data=f"scheme_details:{slug}:{offset}"),
                InlineKeyboardButton(translate(language, "check.documents"), callback_data=f"document_checklist:{slug}:{offset}"),
            ],
            [
                apply_button,
                InlineKeyboardButton(translate(language, "find.alternatives"), callback_data=f"alternatives:{slug}:{offset}"),
            ],
            [InlineKeyboardButton(translate(language, "back.results"), callback_data=f"recommendations_return:{offset}")],
        ]
    )


def _details_keyboard(
    scheme: dict[str, Any], offset: int, request_specific_community: bool = False, language: str = "en"
) -> InlineKeyboardMarkup:
    slug = scheme["slug"]
    apply_url = _valid_http_url(scheme.get("apply_url"))
    apply_button = (
        InlineKeyboardButton(translate(language, "apply"), url=apply_url)
        if apply_url
        else InlineKeyboardButton(translate(language, "apply"), callback_data=f"scheme_apply:{slug}:{offset}")
    )
    rows = [
        [
            InlineKeyboardButton(translate(language, "check.documents"), callback_data=f"document_checklist:{slug}:{offset}"),
            InlineKeyboardButton(translate(language, "find.alternatives"), callback_data=f"alternatives:{slug}:{offset}"),
        ],
        [apply_button],
    ]
    if request_specific_community:
        rows.append(
            [
                InlineKeyboardButton(
                    translate(language, "specific.detail"),
                    callback_data=f"profile_specific:{slug}:{offset}",
                )
            ]
        )
    rows.append(
        [
            InlineKeyboardButton(translate(language, "back.results"), callback_data=f"recommendations_return:{offset}"),
            InlineKeyboardButton(translate(language, "main"), callback_data="nav:home"),
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

    def _language(self, user_id: int) -> str:
        language = self.store.get_profile(user_id).get("language", "en")
        return language if language in LANGUAGES else "en"

    def _t(self, user_id: int, key: str, **values: str) -> str:
        return translate(self._language(user_id), key, **values)

    def _reserve_callback(self, callback_id: str) -> bool:
        if callback_id in self._processed_callback_ids:
            return False
        self._processed_callback_ids[callback_id] = None
        if len(self._processed_callback_ids) > 2048:
            self._processed_callback_ids.popitem(last=False)
        return True

    def _social_category_keyboard(self, language: str = "en") -> InlineKeyboardMarkup:
        rows = [
            [
                InlineKeyboardButton(category, callback_data=f"profile_category:{category}")
                for category in SOCIAL_CATEGORIES[index : index + 2]
            ]
            for index in range(0, len(SOCIAL_CATEGORIES), 2)
        ]
        rows.append([InlineKeyboardButton(translate(language, "skip"), callback_data="profile_category:skip")])
        return InlineKeyboardMarkup(rows)

    async def _prompt_profile_field(self, message: Any, user_id: int, field: str) -> None:
        language = self._language(user_id)
        if field == "social_category":
            await message.reply_text(
                self._t(user_id, "prompt.social_category"),
                reply_markup=self._social_category_keyboard(language),
            )
        elif field == "residence":
            await message.reply_text(
                self._t(user_id, "prompt.residence"),
                reply_markup=InlineKeyboardMarkup(
                    [[
                        InlineKeyboardButton(self._t(user_id, "rural"), callback_data="profile_choice:residence:rural"),
                        InlineKeyboardButton(self._t(user_id, "urban"), callback_data="profile_choice:residence:urban"),
                    ]]
                ),
            )
        elif field == "gender":
            await message.reply_text(
                self._t(user_id, "prompt.gender"),
                reply_markup=InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(self._t(user_id, "female"), callback_data="profile_choice:gender:female"),
                            InlineKeyboardButton(self._t(user_id, "male"), callback_data="profile_choice:gender:male"),
                        ],
                        [
                            InlineKeyboardButton(self._t(user_id, "other"), callback_data="profile_choice:gender:other"),
                            InlineKeyboardButton(self._t(user_id, "skip"), callback_data="profile_choice:gender:skip"),
                        ],
                    ]
                ),
            )
        elif field in {"disability", "bpl"}:
            await message.reply_text(
                self._t(user_id, f"prompt.{field}"),
                reply_markup=InlineKeyboardMarkup(
                    [[
                        InlineKeyboardButton(self._t(user_id, "yes"), callback_data=f"profile_choice:{field}:yes"),
                        InlineKeyboardButton(self._t(user_id, "no"), callback_data=f"profile_choice:{field}:no"),
                        InlineKeyboardButton(self._t(user_id, "skip"), callback_data=f"profile_choice:{field}:skip"),
                    ]]
                ),
            )
        else:
            await message.reply_text(self._t(user_id, f"prompt.{field}"))

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
        user_id = update.effective_user.id
        await update.effective_message.reply_text(
            self._t(user_id, "welcome"),
            parse_mode=ParseMode.HTML,
            reply_markup=_main_menu(self._language(user_id)),
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.effective_message.reply_text(
            self._t(update.effective_user.id, "help"),
            parse_mode=ParseMode.HTML,
        )

    async def language_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        language = self._language(user_id)
        await update.effective_message.reply_text(
            f"{self._t(user_id, 'language.current', language=LANGUAGE_NAMES[language])}\n\n{self._t(user_id, 'language.title')}",
            parse_mode=ParseMode.HTML,
            reply_markup=_language_keyboard(language),
        )

    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        self.store.clear_conversation(user_id)
        await update.effective_message.reply_text(
            self._t(user_id, "reset"),
            reply_markup=_main_menu(self._language(user_id)),
        )

    async def profile(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        profile = self.store.get_profile(user_id)
        fields = [
            f"{_escape(self._t(user_id, f'profile.field.{key}'), 50)}: "
            f"{_escape(profile.get(key) if profile.get(key) is not None else self._t(user_id, 'field.not_provided'), 140)}"
            for key in (*PROFILE_FIELDS, "life_event")
        ]
        fields.append(f"{self._t(user_id, 'language.short')}: {LANGUAGE_NAMES[language]}")
        await update.effective_message.reply_text(
            self._t(user_id, "profile.title") + "\n" + "\n".join(fields),
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton(self._t(user_id, "profile.edit"), callback_data="nav:profile")],
                    [InlineKeyboardButton(self._t(user_id, "profile.category"), callback_data="nav:edit_social_category")],
                    [InlineKeyboardButton(self._t(user_id, "language.button"), callback_data="nav:language")],
                    [InlineKeyboardButton(self._t(user_id, "home"), callback_data="nav:home")],
                ]
            ),
        )

    async def delete_profile(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        language = self._language(user_id)
        self.store.delete_profile(user_id)
        await update.effective_message.reply_text(
            translate(language, "profile.deleted"),
            reply_markup=_main_menu(language),
        )

    async def about(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        await update.effective_message.reply_text(
            f"{self._t(user_id, 'about')}\n\n📚 {len(self.schemes):,} {self._t(user_id, 'records.label')}."
        )

    async def callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if not query:
            return
        if not self._reserve_callback(str(query.id)):
            await query.answer(self._t(update.effective_user.id, "callback.duplicate"))
            return
        data = query.data or ""
        action, separator, value = data.partition(":")
        route = LEGACY_CALLBACK_ACTIONS.get(action, action)
        parts = data.split(":")
        valid = separator and (
            (route == "nav" and value in NAVIGATION_ACTIONS)
            or (route == "language" and value in LANGUAGES)
            or (
                route == "document_checklist"
                and len(parts) in {2, 3, 4}
                and parts[1] in self.by_slug
                and (len(parts) == 2 or parts[2].isdigit())
                and (len(parts) < 4 or (parts[3].isdigit() and int(parts[3]) * DOCUMENTS_PER_PAGE < max(1, len(_documents_for(self.by_slug[parts[1]])))))
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
                    or (parts[1] in {"disability", "bpl"} and parts[2] in {"yes", "no", "skip"})
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
                and len(parts) in {4, 5, 6}
                and parts[1] in self.by_slug
                and parts[2].isdigit()
                and int(parts[2]) < len(_documents_for(self.by_slug[parts[1]]))
                and parts[3] in {"available", "missing", "unknown"}
                and all(item.isdigit() for item in parts[4:])
            )
        )
        await query.answer(
            None if valid else self._t(update.effective_user.id, "callback.expired"),
            show_alert=not valid,
        )
        if not valid:
            return
        user_id = update.effective_user.id
        if route == "language":
            self.store.clear_conversation(user_id)
            profile = self.store.get_profile(user_id)
            profile["language"] = value
            self.store.replace_profile(user_id, profile)
            await query.edit_message_text(
                self._t(user_id, "language.saved", language=LANGUAGE_NAMES[value]) + "\n\n" + self._t(user_id, "welcome"),
                parse_mode=ParseMode.HTML,
                reply_markup=_main_menu(value),
            )
        elif route == "nav":
            await self._menu_action(query, user_id, value)
        elif route == "document_checklist":
            self.store.clear_conversation(user_id)
            offset = int(parts[2]) if len(parts) >= 3 else 0
            page = int(parts[3]) if len(parts) == 4 else 0
            await self._document_checklist(query.message, user_id, parts[1], offset, page, edit=True)
        elif route == "eligibility_check":
            self.store.clear_conversation(user_id)
            await self._show_scheme_eligibility(query.message, user_id, self.by_slug[value])
        elif route == "document_status":
            _, slug, index_text, status, *page_values = parts
            self.store.clear_conversation(user_id)
            document = _documents_for(self.by_slug[slug])[int(index_text)]
            self.store.set_document_status(user_id, slug, document, status)
            await self._document_checklist(
                query.message,
                user_id,
                slug,
                int(page_values[0]) if page_values else 0,
                int(page_values[1]) if len(page_values) > 1 else 0,
                edit=True,
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
                    self._t(user_id, "profile.updated"),
                    reply_markup=_main_menu(self._language(user_id)),
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
            await target.reply_text(self._t(user_id, "choose"), reply_markup=_main_menu(self._language(user_id)))
        elif action == "language":
            language = self._language(user_id)
            text = f"{self._t(user_id, 'language.current', language=LANGUAGE_NAMES[language])}\n\n{self._t(user_id, 'language.title')}"
            if hasattr(message, "edit_message_text"):
                await message.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=_language_keyboard(language))
            else:
                await target.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=_language_keyboard(language))
        elif action == "profile":
            self.store.set_conversation(user_id, "profile", {"index": 0, "profile": self.store.get_profile(user_id)})
            await self._prompt_profile_field(target, user_id, PROFILE_FIELDS[0])
        elif action == "edit_social_category":
            self.store.set_conversation(user_id, "social_category_edit")
            await target.reply_text(
                self._t(user_id, "prompt.social_category"),
                reply_markup=self._social_category_keyboard(self._language(user_id)),
            )
        elif action == "add_family":
            self.store.set_conversation(user_id, "family", {"index": 0, "profile": {}})
            await target.reply_text(self._t(user_id, "prompt.family_name"))
        elif action == "family_edit":
            await self._edit_household(target, user_id)
        elif action == "family":
            await self._family_summary(target, user_id, self.store.get_profile(user_id))
        elif action == "documents":
            self.store.set_conversation(user_id, "document_lookup")
            await target.reply_text(
                self._t(user_id, "prompt.documents")
            )
        elif action == "eligibility":
            profile = self.store.get_profile(user_id)
            if not _profile_ready(profile):
                self.store.set_conversation(user_id, "profile", {"index": 0, "profile": profile, "return_to": action})
                await target.reply_text(self._t(user_id, "need.profile") + self._t(user_id, "prompt.age"))
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
                await target.reply_text(self._t(user_id, "need.profile") + self._t(user_id, "prompt.age"))
                return
            if action == "discover":
                await self._show_recommendations(target, user_id, profile, "government assistance benefits for my circumstances")
            else:
                await self._show_recommendations(target, user_id, profile, "alternative assistance schemes", alternatives=True)
        elif action == "event":
            self.store.set_conversation(user_id, "event", {})
            await target.reply_text(self._t(user_id, "prompt.event"))
        elif action == "compare":
            self.store.set_conversation(user_id, "compare", {})
            await target.reply_text(self._t(user_id, "prompt.compare"))
        elif action == "ask":
            self.store.set_conversation(user_id, "ask", {})
            await target.reply_text(self._t(user_id, "prompt.ask"))
        elif action == "delete":
            await target.reply_text(self._t(user_id, "delete.confirm"), reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(self._t(user_id, "delete.yes"), callback_data="nav:confirm_delete"), InlineKeyboardButton(self._t(user_id, "cancel"), callback_data="nav:home")]]))
        elif action == "confirm_delete":
            language = self._language(user_id)
            self.store.delete_profile(user_id)
            await target.reply_text(translate(language, "profile.deleted"), reply_markup=_main_menu(language))

    async def text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        text = (update.effective_message.text or "").strip()
        quick_actions = {
            "home": "home", "menu": "home", "main menu": "home", "back": "home",
            "schemes": "discover", "discover": "discover", "find schemes": "discover",
            "family": "family", "family benefits": "family",
            "life event": "event", "eligibility": "eligibility", "check eligibility": "eligibility",
            "documents": "documents", "document checklist": "documents",
            "alternatives": "alternatives", "compare": "compare", "compare schemes": "compare",
            "ask": "ask", "ask a question": "ask", "language": "language", "change language": "language",
            "profile": "profile", "edit profile": "profile", "delete my data": "delete",
        }
        localized_actions = {
            "menu.discover": "discover", "menu.family": "family", "menu.event": "event",
            "menu.eligibility": "eligibility", "menu.documents": "documents",
            "menu.alternatives": "alternatives", "menu.compare": "compare", "menu.ask": "ask",
            "menu.profile": "profile", "menu.language": "language", "menu.delete": "delete",
        }
        normalized_text = _normalize_menu_text(text)
        action = quick_actions.get(normalized_text)
        if action is None:
            for language in LANGUAGES:
                action = next(
                    (target for key, target in localized_actions.items()
                     if _normalize_menu_text(translate(language, key)) == normalized_text),
                    None,
                )
                if action:
                    break
        if action:
            self.store.clear_conversation(user_id)
            await self._menu_action(update.effective_message, user_id, action)
            return
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
                    self._t(user_id, "prompt.social_category"),
                    reply_markup=self._social_category_keyboard(self._language(user_id)),
                )
                return
            self._save_social_category(
                user_id,
                "skip" if selected is None else selected,
            )
            self.store.clear_conversation(user_id)
            await update.effective_message.reply_text(
                self._t(user_id, "profile.updated"),
                reply_markup=_main_menu(self._language(user_id)),
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
            await self._compare(update.effective_message, user_id, text)
        elif state == "ask":
            self.store.clear_conversation(user_id)
            await self._answer_question(update.effective_message, user_id, text)
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
                        "age": self._t(user_id, "prompt.confirm.age"),
                        "state": self._t(user_id, "prompt.confirm.state"),
                        "occupation": self._t(user_id, "prompt.confirm.occupation"),
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
                "scheme_discovery": ("menu.discover", "discover"),
                "eligibility": ("menu.eligibility", "eligibility"),
                "family_benefits": ("menu.family", "family"),
                "life_events": ("menu.event", "event"),
                "documents": ("menu.documents", "documents"),
                "application": ("menu.ask", "ask"),
                "scheme_information": ("menu.ask", "ask"),
                "alternatives": ("menu.alternatives", "alternatives"),
                "comparison": ("menu.compare", "compare"),
                "general": ("home", "home"),
            }
            action_label_key, action = intent_actions.get(intent, intent_actions["general"])
            intent_label_key = f"intent.{intent}" if intent in intent_actions else "intent.general"
            await update.effective_message.reply_text(
                self._t(user_id, "intent.choose", intent=self._t(user_id, intent_label_key)),
                reply_markup=InlineKeyboardMarkup(
                    [
                        [InlineKeyboardButton(self._t(user_id, action_label_key), callback_data=f"nav:{action}")],
                        [InlineKeyboardButton(self._t(user_id, "menu.full"), callback_data="nav:home")],
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
            f"{self._t(user_id, 'profile.confirmed')} {field.replace('_', ' ')}: {value}. {self._t(user_id, 'choose')}",
            reply_markup=_main_menu(self._language(user_id)),
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
            _scheme_card({"scheme": scheme, "eligibility": result}, self._language(user_id)),
            parse_mode=ParseMode.HTML,
            reply_markup=_scheme_keyboard(scheme, 0, self._language(user_id)),
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
                    await message.reply_text(self._t(user_id, "error.age"))
                    return
                value = int(match.group())
            elif field == "annual_income":
                digits = re.sub(r"[^\d]", "", text)
                if not digits:
                    await message.reply_text(self._t(user_id, "error.income"))
                    return
                value = int(digits)
            elif field == "residence":
                if text.casefold() not in {"rural", "urban"}:
                    await message.reply_text(self._t(user_id, "error.residence"))
                    return
                value = text.casefold()
            elif field in {"disability", "bpl"}:
                normalized = text.strip().casefold()
                if normalized in {"yes", "y", "ಹೌದು", "हाँ"}:
                    value = True
                elif normalized in {"no", "n", "ಇಲ್ಲ", "नहीं"}:
                    value = False
                else:
                    await self._prompt_profile_field(message, user_id, field)
                    return
            elif field == "state":
                value = text.strip().title()
            elif field == "social_category":
                selected = next(
                    (category for category in SOCIAL_CATEGORIES[:-1] if category.casefold() == text.casefold()),
                    None,
                )
                if selected is None:
                    await self._prompt_profile_field(message, user_id, field)
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
            await self._prompt_profile_field(message, user_id, PROFILE_FIELDS[index])
            return
        self.store.replace_profile(user_id, profile)
        self.store.clear_conversation(user_id)
        return_to = data.get("return_to")
        await message.reply_text(
            self._t(user_id, "profile.saved"),
            reply_markup=_main_menu(self._language(user_id)),
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
                await message.reply_text(self._t(user_id, "error.name"))
                return
            data["name"] = text[:80]
        elif text.casefold() not in {"skip", "unknown", "prefer not to say"}:
            if field == "age":
                match = re.search(r"\d{1,3}", text)
                if not match or not 1 <= int(match.group()) <= 120:
                    await message.reply_text(self._t(user_id, "error.age"))
                    return
                profile[field] = int(match.group())
            elif field == "annual_income":
                digits = re.sub(r"[^\d]", "", text)
                if not digits:
                    await message.reply_text(self._t(user_id, "error.income"))
                    return
                profile[field] = int(digits)
            elif field == "residence":
                if text.casefold() not in {"rural", "urban"}:
                    await message.reply_text(self._t(user_id, "error.residence"))
                    return
                profile[field] = text.casefold()
            elif field in {"disability", "bpl"}:
                normalized = text.strip().casefold()
                if normalized in {"yes", "y", "ಹೌದು", "हाँ"}:
                    profile[field] = True
                elif normalized in {"no", "n", "ಇಲ್ಲ", "नहीं"}:
                    profile[field] = False
                else:
                    await self._prompt_profile_field(message, user_id, field)
                    return
            elif field == "social_category":
                selected = next(
                    (category for category in SOCIAL_CATEGORIES[:-1] if category.casefold() == text.casefold()),
                    None,
                )
                if selected is None:
                    await self._prompt_profile_field(message, user_id, field)
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
                await message.reply_text(self._t(user_id, "prompt.family_name"))
            else:
                await self._prompt_profile_field(message, user_id, next_field)
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
            self._t(user_id, "family.saved", name=data["name"]),
            reply_markup=_main_menu(self._language(user_id)),
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
        results = self.recommender.recommend(
            profile,
            query,
            limit=len(self.schemes),
            outcomes={"potentially_eligible"},
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
            await message.reply_text(
                f"<b>{self._t(user_id, 'no_eligible.title')}</b>\n{self._t(user_id, 'no_eligible.body')}",
                parse_mode=ParseMode.HTML,
                reply_markup=InlineKeyboardMarkup(
                    [
                        [InlineKeyboardButton(self._t(user_id, "profile.edit.button"), callback_data="nav:profile")],
                        [InlineKeyboardButton(self._t(user_id, "main"), callback_data="nav:home")],
                    ]
                ),
            )
            return
        await self._render_recommendation_page(message, user_id, results, offset, mode)

    async def _render_recommendation_page(
        self,
        message: Any,
        user_id: int,
        results: list[dict[str, Any]],
        offset: int,
        mode: str,
    ) -> None:
        page_results = results[max(0, offset) : max(0, offset) + 3]
        if not page_results:
            await message.reply_text(self._t(user_id, "callback.expired"), reply_markup=_main_menu(self._language(user_id)))
            return
        if mode == "alternatives":
            title = self._t(user_id, "title.alternatives")
        elif mode == "documents":
            title = self._t(user_id, "title.documents")
        else:
            title = self._t(user_id, "title.matches")
        await message.reply_text(
            f"<b>{title}</b>\n{self._t(user_id, 'label.showing', start=str(offset + 1), end=str(offset + len(page_results)), total=str(len(results)))}",
            parse_mode=ParseMode.HTML,
        )
        for result in page_results:
            await message.reply_text(
                _scheme_card(result, self._language(user_id)),
                parse_mode=ParseMode.HTML,
                reply_markup=_scheme_keyboard(result["scheme"], offset, self._language(user_id)),
                disable_web_page_preview=True,
            )
        page_buttons: list[InlineKeyboardButton] = []
        if offset:
            page_buttons.append(
                InlineKeyboardButton(self._t(user_id, "previous"), callback_data=f"recommendations_page:{max(0, offset - 3)}")
            )
        if offset + 3 < len(results):
            page_buttons.append(
                InlineKeyboardButton(self._t(user_id, "next"), callback_data=f"recommendations_page:{offset + 3}")
            )
        navigation_rows = [page_buttons] if page_buttons else []
        navigation_rows.append([InlineKeyboardButton(self._t(user_id, "main"), callback_data="nav:home")])
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
        await self._render_recommendation_page(query.message, user_id, results, offset, mode)

    async def _family_summary(self, message: Any, user_id: int, profile: dict[str, Any]) -> None:
        family = self.store.get_family(user_id)
        members = [{"name": self._t(user_id, "family.you"), **profile}, *family]
        lines = [self._t(user_id, "family.title")]
        rows: list[list[InlineKeyboardButton]] = []
        for index, member in enumerate(members):
            details = [
                str(member.get("occupation") or self._t(user_id, "family.occupation_missing")),
                self._t(user_id, "family.age", age=str(member["age"])) if member.get("age") is not None else self._t(user_id, "family.age_missing"),
            ]
            lines.append(f"• {_escape(member['name'], 80)} — {_escape(', '.join(details), 120)}")
            rows.append(
                [
                    InlineKeyboardButton(
                        self._t(user_id, "family.view", name=member["name"][:35]),
                        callback_data=f"family_benefits:{index}",
                    )
                ]
            )
        rows.extend(
            [
                [InlineKeyboardButton(self._t(user_id, "family.add"), callback_data="nav:add_family")],
                [InlineKeyboardButton(self._t(user_id, "family.edit"), callback_data="nav:family_edit")],
                [InlineKeyboardButton(self._t(user_id, "main"), callback_data="nav:home")],
            ]
        )
        if not family:
            lines.append(self._t(user_id, "family.only_you"))
        else:
            lines.append(self._t(user_id, "family.choose_member"))
        await message.reply_text(
            "\n".join(lines),
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(rows),
        )

    async def _show_family_member_benefits(
        self, message: Any, user_id: int, member_index: int
    ) -> None:
        profile = self.store.get_profile(user_id)
        members = [{"name": self._t(user_id, "family.you"), **profile}, *self.store.get_family(user_id)]
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
            [InlineKeyboardButton(self._t(user_id, "family.profile.edit"), callback_data="nav:profile")]
        ]
        rows.extend(
            [
                InlineKeyboardButton(f"✏️ {member['name']}", callback_data=f"family_edit:{member['id']}")
            ]
            for member in family
        )
        rows.extend(
            [
                [InlineKeyboardButton(self._t(user_id, "family.add"), callback_data="nav:add_family")],
                [InlineKeyboardButton(self._t(user_id, "main"), callback_data="nav:home")],
            ]
        )
        await message.reply_text(
            self._t(user_id, "family.choose_edit"),
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
        self, message: Any, user_id: int, slug: str, offset: int = 0, page: int = 0, edit: bool = False
    ) -> None:
        scheme = self.by_slug.get(slug)
        if not scheme:
            await message.reply_text("That scheme record is no longer available.")
            return
        documents = _documents_for(scheme)
        if not documents:
            text = self._t(user_id, "documents.none")
            markup = InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton(self._t(user_id, "back.results"), callback_data=f"recommendations_return:{offset}")],
                    [InlineKeyboardButton(self._t(user_id, "main"), callback_data="nav:home")],
                ]
            )
            send = message.edit_text if edit else message.reply_text
            await send(
                text,
                reply_markup=markup,
            )
            return
        statuses = self.store.get_document_statuses(user_id, slug)
        rows = []
        start = page * DOCUMENTS_PER_PAGE
        page_documents = documents[start : start + DOCUMENTS_PER_PAGE]
        available_count = sum(statuses.get(document) == "available" for document in documents)
        lines = [f"{self._t(user_id, 'documents.title')} — <b>{_escape(scheme['name'], 150)}</b>"]
        lines.append(self._t(
            user_id,
            "documents.summary",
            start=str(start + 1),
            end=str(start + len(page_documents)),
            total=str(len(documents)),
            done=str(available_count),
        ))
        for index, document in enumerate(page_documents, start=start):
            status = statuses.get(document, "unknown")
            lines.append(f"{index + 1}. {_escape(document, 140)} — <b>{self._t(user_id, f'documents.{status}')}</b>")
            rows.append(
                [
                    InlineKeyboardButton(f"{index + 1}: {self._t(user_id, 'documents.available.label')}", callback_data=f"document_status:{slug}:{index}:available:{offset}:{page}"),
                    InlineKeyboardButton(self._t(user_id, "documents.missing.label"), callback_data=f"document_status:{slug}:{index}:missing:{offset}:{page}"),
                    InlineKeyboardButton(self._t(user_id, "documents.unknown.label"), callback_data=f"document_status:{slug}:{index}:unknown:{offset}:{page}"),
                ]
            )
        page_count = max(1, (len(documents) + DOCUMENTS_PER_PAGE - 1) // DOCUMENTS_PER_PAGE)
        page_controls = []
        if page > 0:
            page_controls.append(InlineKeyboardButton(self._t(user_id, "previous"), callback_data=f"document_checklist:{slug}:{offset}:{page - 1}"))
        if page + 1 < page_count:
            page_controls.append(InlineKeyboardButton(self._t(user_id, "next"), callback_data=f"document_checklist:{slug}:{offset}:{page + 1}"))
        if page_controls:
            rows.append(page_controls)
        rows.extend(
            [
                [InlineKeyboardButton(self._t(user_id, "back.results"), callback_data=f"recommendations_return:{offset}")],
                [InlineKeyboardButton(self._t(user_id, "main"), callback_data="nav:home")],
            ]
        )
        send = message.edit_text if edit else message.reply_text
        await send(
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
            _scheme_details(item, self._language(user_id)),
            parse_mode=ParseMode.HTML,
            reply_markup=_details_keyboard(scheme, offset, request_specific, self._language(user_id)),
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
            outcomes={"potentially_eligible"},
        )
        results = [item for item in results if item["scheme"]["slug"] != slug][:3]
        if not results:
            await message.reply_text(
                self._t(user_id, "no_alternatives"),
                reply_markup=InlineKeyboardMarkup(
                    [
                        [InlineKeyboardButton(self._t(user_id, "back.results"), callback_data=f"recommendations_return:{offset}")],
                        [InlineKeyboardButton(self._t(user_id, "main"), callback_data="nav:home")],
                    ]
                ),
            )
            return
        await message.reply_text(
            f"<b>{self._t(user_id, 'title.alternatives')}</b>\n{self._t(user_id, 'alternatives.subtitle')}",
            parse_mode=ParseMode.HTML,
        )
        for item in results:
            await message.reply_text(
                _scheme_card(item, self._language(user_id)),
                parse_mode=ParseMode.HTML,
                reply_markup=_scheme_keyboard(item["scheme"], offset, self._language(user_id)),
                disable_web_page_preview=True,
            )
        await message.reply_text(
            self._t(user_id, "alternatives.return"),
            reply_markup=InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton(self._t(user_id, "back.results"), callback_data=f"recommendations_return:{offset}")],
                    [InlineKeyboardButton(self._t(user_id, "main"), callback_data="nav:home")],
                ]
            ),
        )

    async def _compare(self, message: Any, user_id: int, text: str) -> None:
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
            source_url = _valid_http_url(scheme.get("official_url"))
            text = (
                f"<b>{_escape(scheme.get('name'))}</b>\n"
                f"💰 {_escape(_short_summary(scheme.get('benefits'), 120), 140)}\n"
                f"✅ {_escape(_short_summary(scheme.get('eligibility_text'), 150), 170)}\n"
                + (f'<a href="{html.escape(source_url, quote=True)}">{self._t(user_id, "source.link")}</a>' if source_url else self._t(user_id, "source.missing"))
            )
            await message.reply_text(text, parse_mode=ParseMode.HTML, disable_web_page_preview=True)

    async def _answer_question(self, message: Any, user_id: int, text: str) -> None:
        matches = self.recommender.faq_matches(text, limit=1)
        if not matches or matches[0][1] < 0.12:
            await message.reply_text(self._t(user_id, "not_found"), reply_markup=_main_menu(self._language(user_id)))
            return
        scheme, score = matches[0]
        profile = self.store.get_profile(user_id)
        self.store.set_recommendation_page(user_id, text, profile, "ask", [scheme["slug"]])
        info_url = _valid_http_url(scheme.get("official_url"))
        answer = (
            f"<b>{_escape(scheme.get('name'))}</b>\n"
            f"📝 {_escape(_short_summary(scheme.get('description'), 180), 200)}\n"
            f"💰 {_escape(_short_summary(scheme.get('benefits'), 130), 150)}\n"
            + (f'<a href="{html.escape(info_url, quote=True)}">{self._t(user_id, "source.link")}</a>\n' if info_url else self._t(user_id, "source.missing") + "\n")
            + f"{self._t(user_id, 'label.score')}: {score:.2f} · {self._t(user_id, 'label.score_note')}."
        )
        await message.reply_text(
            answer,
            parse_mode=ParseMode.HTML,
            reply_markup=_scheme_keyboard(scheme, 0, self._language(user_id)),
            disable_web_page_preview=True,
        )


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
    application.add_handler(CommandHandler("language", lambda u, c: invoke("language_command", u, c)))
    application.add_handler(CommandHandler("delete_profile", lambda u, c: invoke("delete_profile", u, c)))
    application.add_handler(CommandHandler("about", lambda u, c: invoke("about", u, c)))
    application.add_handler(CallbackQueryHandler(service.callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, service.text_message))
    return application


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # HTTPX logs full Bot API URLs at INFO, which contain the bot token.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    application = create_application()
    application.run_polling(allowed_updates=Update.ALL_TYPES)
