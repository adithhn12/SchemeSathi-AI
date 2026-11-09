from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from schemesathi.bot import MAIN_MENU, SchemeSathiBot
from schemesathi.data import load_schemes
from schemesathi.eligibility import evaluate_eligibility
from schemesathi.extraction import extract_profile_details
from schemesathi.ml import SchemeRecommender
from schemesathi.storage import Store


def scheme(**overrides):
    base = {
        "slug": "test",
        "name": "Karnataka Farmer Support",
        "description": "Financial support for farmers in Karnataka.",
        "state": "Karnataka",
        "category": "Agriculture",
        "beneficiary_type": "Farmer",
        "benefits": "Financial support",
        "eligibility_text": "The applicant should be a farmer.",
        "documents_required": "Income certificate\nBank passbook",
        "apply_url": None,
        "official_url": None,
        "eligibility_age_min": 18,
        "eligibility_age_max": 60,
        "eligibility_income_max": 500000,
        "eligibility_gender": "all",
        "eligibility_residence": "both",
        "eligibility_state": ["Karnataka"],
        "eligibility_disability": False,
        "eligibility_bpl": False,
        "eligibility_caste": None,
    }
    base.update(overrides)
    return base


class EligibilityTests(unittest.TestCase):
    def test_age_and_income_boundaries_are_inclusive(self):
        result = evaluate_eligibility(
            scheme(), {"age": 18, "annual_income": 500000, "state": "Karnataka", "occupation": "farmer"}
        )
        self.assertEqual(result.outcome, "potentially_eligible")

    def test_outside_age_or_income_fails(self):
        result = evaluate_eligibility(
            scheme(), {"age": 61, "annual_income": 500001, "state": "Karnataka"}
        )
        self.assertEqual(result.outcome, "not_matching")
        self.assertEqual(len(result.failed), 2)

    def test_unknown_is_not_passed(self):
        result = evaluate_eligibility(scheme(), {})
        self.assertEqual(result.outcome, "needs_information")
        self.assertTrue(result.unknown)
        self.assertFalse(result.satisfied)

    def test_no_structured_eligibility_rules_need_more_information(self):
        record = scheme(
            state="Central",
            eligibility_age_min=None,
            eligibility_age_max=None,
            eligibility_income_max=None,
            eligibility_gender="all",
            eligibility_residence="both",
            eligibility_state=["All"],
            eligibility_disability=False,
            eligibility_bpl=False,
            eligibility_caste=None,
            beneficiary_type="Individual",
            eligibility_text=None,
        )
        result = evaluate_eligibility(record, {"state": "Karnataka"})
        self.assertEqual(result.outcome, "needs_information")
        self.assertTrue(any("No structured eligibility" in item for item in result.unknown))

    def test_other_state_fails_state_condition(self):
        result = evaluate_eligibility(scheme(), {"age": 30, "annual_income": 1, "state": "Kerala"})
        self.assertEqual(result.outcome, "not_matching")

    def test_gender_residence_and_disability_conditions(self):
        specific = scheme(
            eligibility_gender="female",
            eligibility_residence="rural",
            eligibility_disability=True,
        )
        result = evaluate_eligibility(
            specific,
            {"age": 30, "annual_income": 1, "state": "Karnataka", "gender": "female", "residence": "urban"},
        )
        self.assertEqual(result.outcome, "not_matching")
        self.assertTrue(any("disability" in item for item in result.unknown))

    def test_explicit_beneficiary_occupation_is_checked(self):
        farmer_scheme = scheme(beneficiary_type="Farmer")
        failed = evaluate_eligibility(
            farmer_scheme,
            {"age": 30, "annual_income": 1, "state": "Karnataka", "occupation": "student"},
        )
        unknown = evaluate_eligibility(
            farmer_scheme,
            {"age": 30, "annual_income": 1, "state": "Karnataka"},
        )
        self.assertIn("Occupation", failed.failed[0])
        self.assertTrue(any("Occupation" in item for item in unknown.unknown))

    def test_social_category_is_exact_and_missing_category_stays_unknown(self):
        category_scheme = scheme(eligibility_caste=["SC"])
        matched = evaluate_eligibility(
            category_scheme,
            {
                "age": 30,
                "annual_income": 1,
                "state": "Karnataka",
                "occupation": "farmer",
                "social_category": "SC",
            },
        )
        mismatch = evaluate_eligibility(
            category_scheme,
            {
                "age": 30,
                "annual_income": 1,
                "state": "Karnataka",
                "occupation": "farmer",
                "social_category": "OBC",
            },
        )
        unknown = evaluate_eligibility(
            category_scheme,
            {"age": 30, "annual_income": 1, "state": "Karnataka", "occupation": "farmer"},
        )
        self.assertEqual(matched.outcome, "potentially_eligible")
        self.assertEqual(mismatch.outcome, "not_matching")
        self.assertTrue(any("Social category" in item for item in unknown.unknown))

    def test_specific_community_requirement_is_unknown_without_extra_detail(self):
        result = evaluate_eligibility(
            scheme(eligibility_caste=["Kuruba"]),
            {
                "age": 30,
                "annual_income": 1,
                "state": "Karnataka",
                "occupation": "farmer",
                "social_category": "OBC",
            },
        )
        self.assertEqual(result.outcome, "needs_information")
        self.assertTrue(any("specific caste/community" in item for item in result.unknown))


class DataAndModelsTests(unittest.TestCase):
    def test_real_dataset_loads_multiline_fields_and_records(self):
        schemes = load_schemes()
        self.assertEqual(len(schemes), 4693)
        self.assertIn("\n", next(item for item in schemes if item.get("documents_required"))["documents_required"])

    def test_recommender_returns_ranked_scheme_with_separate_eligibility(self):
        engine = SchemeRecommender([scheme()])
        results = engine.recommend({"state": "Karnataka", "occupation": "farmer"}, "farmer support")
        self.assertEqual(results[0]["scheme"]["slug"], "test")
        self.assertIn("relevance_score", results[0])
        self.assertNotIn("probability", results[0])

    def test_extraction_confirms_multiple_ages(self):
        result = extract_profile_details("I am 28 years old, my mother is 61 years old, and I live in Karnataka")
        self.assertNotIn("age", result["values"])
        self.assertTrue(result["needs_confirmation"])
        self.assertEqual(result["values"]["state"], "Karnataka")

    def test_extraction_collects_education_and_optional_conditions(self):
        result = extract_profile_details(
            "I am a female student studying engineering, have a disability, and belong to OBC"
        )
        self.assertEqual(result["values"]["occupation"], "student")
        self.assertEqual(result["values"]["education"], "engineering")
        self.assertEqual(result["values"]["gender"], "female")
        self.assertEqual(result["values"]["disability"], True)
        self.assertNotIn("caste", result["values"])

    def test_profiles_family_conversations_and_document_status_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "test.sqlite3")
            store.update_profile(1, {"age": 30})
            store.update_profile(2, {"age": 40})
            store.add_family_member(1, "Parent", {"age": 65})
            store.set_conversation(1, "ask", {"query": "pension"})
            store.set_conversation(2, "event")
            store.set_document_status(1, "scheme", "Income certificate", "available")
            store.set_recommendation_page(1, "farmer schemes", {"state": "Karnataka"}, "discover")
            self.assertEqual(store.get_profile(1)["age"], 30)
            self.assertEqual(store.get_profile(2)["age"], 40)
            self.assertEqual(store.get_family(1)[0]["age"], 65)
            self.assertEqual(store.get_conversation(1)[0], "ask")
            self.assertEqual(store.get_conversation(2)[0], "event")
            self.assertEqual(store.get_document_statuses(1, "scheme")["Income certificate"], "available")
            self.assertEqual(store.get_recommendation_page(1)[0], "farmer schemes")
            store.set_recommendation_page(
                1, "farmer schemes", {"state": "Karnataka"}, "discover", ["scheme-a", "scheme-b"]
            )
            self.assertEqual(store.get_recommendation_slugs(1), ["scheme-a", "scheme-b"])
            store.delete_profile(1)
            self.assertEqual(store.get_family(1), [])
            self.assertEqual(store.get_document_statuses(1, "scheme"), {})
            self.assertIsNone(store.get_recommendation_page(1))

    def test_existing_recommendation_table_is_migrated_without_losing_page(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.sqlite3"
            connection = sqlite3.connect(path)
            try:
                connection.execute(
                    "CREATE TABLE recommendation_pages "
                    "(user_id INTEGER PRIMARY KEY, query TEXT NOT NULL, "
                    "profile_json TEXT NOT NULL, mode TEXT NOT NULL)"
                )
                connection.execute(
                    "INSERT INTO recommendation_pages VALUES(1, 'saved search', '{}', 'discover')"
                )
                connection.commit()
            finally:
                connection.close()

            store = Store(path)

            self.assertEqual(store.get_recommendation_page(1), ("saved search", {}, "discover"))
            self.assertEqual(store.get_recommendation_slugs(1), [])
            store.set_recommendation_page(1, "new search", {}, "discover", ["scheme"])
            self.assertEqual(store.get_recommendation_slugs(1), ["scheme"])


class TelegramFlowTests(unittest.IsolatedAsyncioTestCase):
    def make_callback(self, callback_id, data, user_id, message):
        query = SimpleNamespace(
            id=callback_id,
            data=data,
            message=message,
            answer=AsyncMock(),
        )
        return (
            SimpleNamespace(
                callback_query=query,
                effective_user=SimpleNamespace(id=user_id),
            ),
            query,
        )

    async def test_profile_collection_can_continue_from_a_text_message(self):
        with tempfile.TemporaryDirectory() as directory:
            service = SchemeSathiBot(
                Store(Path(directory) / "flow.sqlite3"),
                schemes=[scheme()],
            )
            message = SimpleNamespace(reply_text=AsyncMock())
            await service._menu_action(message, 9, "discover")
            state, data = service.store.get_conversation(9)
            self.assertEqual(state, "profile")
            self.assertEqual(data["return_to"], "discover")
            message.reply_text.assert_awaited_once()

    async def test_requested_navigation_sequence_routes_only_to_selected_features(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "navigation.sqlite3")
            service = SchemeSathiBot(store, schemes=[scheme()])
            user_id = 42
            message = SimpleNamespace(reply_text=AsyncMock())
            start_update = SimpleNamespace(effective_message=message)
            await service.start(start_update, SimpleNamespace())
            start_markup = message.reply_text.await_args.kwargs["reply_markup"]
            self.assertEqual(start_markup.inline_keyboard[0][0].callback_data, "nav:discover")

            await service._menu_action(message, user_id, "profile")
            for value in ("30", "farmer", "college", "300000", "Karnataka", "rural", "female", "OBC"):
                await service._collect_profile(
                    SimpleNamespace(effective_message=message),
                    user_id,
                    value,
                    store.get_conversation(user_id)[1],
                )
            saved_profile = store.get_profile(user_id)
            self.assertEqual(saved_profile["state"], "Karnataka")
            self.assertIsNone(store.get_conversation(user_id)[0])
            store.add_family_member(user_id, "Parent", {"age": 65, "state": "Karnataka"})
            saved_family = store.get_family(user_id)

            service._show_recommendations = AsyncMock()
            service._family_summary = AsyncMock()
            service._document_checklist = AsyncMock()
            service._alternatives = AsyncMock()
            service._compare = AsyncMock()
            callback_number = 0

            async def click(data):
                nonlocal callback_number
                callback_number += 1
                update, query = self.make_callback(
                    f"callback-{callback_number}", data, user_id, message
                )
                await service.callback(update, SimpleNamespace())
                query.answer.assert_awaited_once()

            await click("nav:discover")
            self.assertEqual(service._show_recommendations.await_count, 1)

            await click("nav:family")
            service._family_summary.assert_awaited_once()
            self.assertEqual(service._show_recommendations.await_count, 1)

            await click("nav:home")
            self.assertIsNone(store.get_conversation(user_id)[0])
            self.assertEqual(store.get_profile(user_id), saved_profile)
            home_markup = message.reply_text.await_args.kwargs["reply_markup"]
            self.assertEqual(len(home_markup.inline_keyboard), 5)
            self.assertEqual(service._show_recommendations.await_count, 1)

            await click("nav:eligibility")
            self.assertEqual(service._show_recommendations.await_count, 2)
            self.assertTrue(service._show_recommendations.await_args.kwargs["eligibility_only"])
            await click("nav:home")

            await click("nav:documents")
            self.assertEqual(store.get_conversation(user_id)[0], "document_lookup")
            self.assertEqual(service._show_recommendations.await_count, 2)
            await service.text_message(
                SimpleNamespace(
                    effective_user=SimpleNamespace(id=user_id),
                    effective_message=SimpleNamespace(text="Karnataka Farmer Support", reply_text=message.reply_text),
                ),
                SimpleNamespace(),
            )
            service._document_checklist.assert_awaited_once()
            self.assertIsNone(store.get_conversation(user_id)[0])
            self.assertEqual(service._show_recommendations.await_count, 2)

            await click("nav:alternatives")
            self.assertEqual(service._show_recommendations.await_count, 3)
            self.assertTrue(service._show_recommendations.await_args.kwargs["alternatives"])

            await click("document_checklist:test")
            self.assertEqual(service._document_checklist.await_count, 2)
            self.assertEqual(service._show_recommendations.await_count, 3)
            await click("alternatives:test")
            service._alternatives.assert_awaited_once()
            self.assertEqual(service._show_recommendations.await_count, 3)

            await click("nav:compare")
            self.assertEqual(store.get_conversation(user_id)[0], "compare")
            self.assertIn("scheme names", message.reply_text.await_args.args[0])
            service._compare.assert_not_awaited()

            await click("nav:home")
            self.assertIsNone(store.get_conversation(user_id)[0])
            await click("nav:event")
            self.assertEqual(store.get_conversation(user_id)[0], "event")
            await click("nav:ask")
            self.assertEqual(store.get_conversation(user_id)[0], "ask")
            self.assertEqual(store.get_profile(user_id), saved_profile)
            self.assertEqual(store.get_family(user_id), saved_family)

            callback_data = [
                button.callback_data
                for row in MAIN_MENU.inline_keyboard
                for button in row
            ]
            self.assertEqual(len(callback_data), len(set(callback_data)))

    async def test_duplicate_callback_id_is_answered_but_dispatched_once(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "duplicate.sqlite3")
            store.update_profile(8, {"state": "Karnataka", "occupation": "farmer"})
            service = SchemeSathiBot(store, schemes=[scheme()])
            service._show_recommendations = AsyncMock()
            message = SimpleNamespace(reply_text=AsyncMock())
            update, query = self.make_callback("duplicate-id", "nav:discover", 8, message)

            await service.callback(update, SimpleNamespace())
            await service.callback(update, SimpleNamespace())

            self.assertEqual(query.answer.await_count, 2)
            self.assertEqual(query.answer.await_args_list[1].args[0], "This action was already processed.")
            service._show_recommendations.assert_awaited_once()
            self.assertEqual(store.get_profile(8), {"state": "Karnataka", "occupation": "farmer"})

    async def test_family_menu_without_members_shows_mapping_not_individual_results(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "family-navigation.sqlite3")
            service = SchemeSathiBot(store, schemes=[scheme()])
            message = SimpleNamespace(reply_text=AsyncMock())
            service.recommender.recommend = Mock()
            profile = {"state": "Karnataka", "occupation": "farmer"}

            await service._family_summary(message, 7, profile)

            service.recommender.recommend.assert_not_called()
            self.assertIn("Would you like to add others?", message.reply_text.await_args.args[0])
            self.assertEqual(store.get_profile(7), {})
            self.assertEqual(store.get_family(7), [])

    async def test_family_mapping_lists_members_without_auto_recommending(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "household.sqlite3")
            store.update_profile(12, {"occupation": "student", "age": 21})
            store.add_family_member(12, "Father", {"occupation": "farmer", "age": 52})
            store.add_family_member(12, "Mother", {"occupation": "homemaker", "age": 47})
            service = SchemeSathiBot(store, schemes=[scheme()])
            service.recommender.recommend = Mock()
            message = SimpleNamespace(reply_text=AsyncMock())

            await service._family_summary(message, 12, store.get_profile(12))

            text = message.reply_text.await_args.args[0]
            self.assertIn("You — student, Age 21", text)
            self.assertIn("Father — farmer, Age 52", text)
            self.assertIn("Mother — homemaker, Age 47", text)
            service.recommender.recommend.assert_not_called()
            markup = message.reply_text.await_args.kwargs["reply_markup"]
            callbacks = [button.callback_data for row in markup.inline_keyboard for button in row]
            self.assertIn("family_benefits:1", callbacks)
            self.assertIn("nav:family_edit", callbacks)

    async def test_recommendations_page_three_results_and_preserves_order(self):
        with tempfile.TemporaryDirectory() as directory:
            schemes = [
                scheme(slug=f"scheme-{index}", name=f"Karnataka Farmer Support {index}")
                for index in range(8)
            ]
            store = Store(Path(directory) / "paging.sqlite3")
            service = SchemeSathiBot(store, schemes=schemes)
            recommend = Mock(wraps=service.recommender.recommend)
            service.recommender.recommend = recommend
            message = SimpleNamespace(reply_text=AsyncMock())
            profile = {
                "age": 35,
                "occupation": "farmer",
                "state": "Karnataka",
                "annual_income": 100000,
            }

            await service._show_recommendations(message, 15, profile, "farmer support")

            slugs = store.get_recommendation_slugs(15)
            self.assertEqual(len(slugs), 8)
            first_page_cards = [
                call for call in message.reply_text.await_args_list
                if call.args and "Why it may fit:" in call.args[0]
            ]
            self.assertEqual(len(first_page_cards), 3)
            self.assertIn("Showing 1–3 of 8 matches.", message.reply_text.await_args_list[0].args[0])

            await service._recommendation_page(SimpleNamespace(message=message), 15, "3")

            self.assertEqual(recommend.call_count, 1)
            self.assertEqual(store.get_recommendation_slugs(15), slugs)
            all_cards = [
                call for call in message.reply_text.await_args_list
                if call.args and "Why it may fit:" in call.args[0]
            ]
            self.assertEqual(len(all_cards), 6)
            self.assertIn("Showing 4–6 of 8 matches.", message.reply_text.await_args_list[5].args[0])

    async def test_scheme_details_are_clean_and_return_to_saved_results(self):
        with tempfile.TemporaryDirectory() as directory:
            detailed_scheme = scheme(
                benefits="Benefit &amp; **support** is provided.",
                eligibility_text="The applicant **must** be a farmer.",
                application_process="1. Visit the **portal**.\n2. Submit the form.",
                apply_url="https://example.gov.in/apply",
            )
            store = Store(Path(directory) / "details.sqlite3")
            store.update_profile(19, {"occupation": "farmer", "state": "Karnataka"})
            store.set_recommendation_page(19, "farmer support", store.get_profile(19), "discover", ["test"])
            service = SchemeSathiBot(store, schemes=[detailed_scheme])
            message = SimpleNamespace(reply_text=AsyncMock())
            update, query = self.make_callback("details-callback", "scheme_details:test:0", 19, message)

            await service.callback(update, SimpleNamespace())

            query.answer.assert_awaited_once()
            details = message.reply_text.await_args.args[0]
            self.assertIn("<b>Benefits</b>", details)
            self.assertIn("Benefit &amp; support", details)
            self.assertNotIn("**", details)
            self.assertIn("Open the application page", details)
            buttons = message.reply_text.await_args.kwargs["reply_markup"].inline_keyboard
            self.assertTrue(any(button.callback_data == "recommendations_return:0" for row in buttons for button in row))

            back_update, back_query = self.make_callback("back-callback", "recommendations_return:0", 19, message)
            await service.callback(back_update, SimpleNamespace())
            back_query.answer.assert_awaited_once()
            self.assertIn("Why it may fit:", message.reply_text.await_args_list[-2].args[0])

    async def test_social_category_can_be_set_cleared_and_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "social-category.sqlite3")
            store.update_profile(20, {"social_category": "OBC"})
            store.update_profile(21, {"social_category": "SC"})
            service = SchemeSathiBot(store, schemes=[scheme()])
            message = SimpleNamespace(reply_text=AsyncMock())
            await service._menu_action(message, 20, "edit_social_category")
            update, query = self.make_callback(
                "category-set", "profile_category:SC", 20, message
            )

            await service.callback(update, SimpleNamespace())

            query.answer.assert_awaited_once()
            self.assertEqual(store.get_profile(20)["social_category"], "SC")
            clear_update, clear_query = self.make_callback(
                "category-clear", "profile_category:Prefer not to say", 20, message
            )
            await service._menu_action(message, 20, "edit_social_category")
            await service.callback(clear_update, SimpleNamespace())

            clear_query.answer.assert_awaited_once()
            self.assertNotIn("social_category", store.get_profile(20))
            self.assertEqual(store.get_profile(21)["social_category"], "SC")
            self.assertIsNone(store.get_conversation(20)[0])

    async def test_specific_community_is_requested_only_for_relevant_scheme_and_not_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "specific-community.sqlite3")
            store.update_profile(
                23,
                {
                    "age": 35,
                    "occupation": "farmer",
                    "state": "Karnataka",
                    "annual_income": 100000,
                    "social_category": "OBC",
                },
            )
            community_scheme = scheme(eligibility_caste=["Kuruba"])
            service = SchemeSathiBot(store, schemes=[community_scheme])
            message = SimpleNamespace(reply_text=AsyncMock())
            await service._show_scheme_details(message, 23, "test", 0)
            detail_markup = message.reply_text.await_args.kwargs["reply_markup"]
            detail_callbacks = [
                button.callback_data
                for row in detail_markup.inline_keyboard
                for button in row
            ]
            self.assertIn("profile_specific:test:0", detail_callbacks)

            update, query = self.make_callback(
                "specific-question", "profile_specific:test:0", 23, message
            )
            await service.callback(update, SimpleNamespace())
            self.assertEqual(store.get_conversation(23)[0], "specific_community")

            await service.text_message(
                SimpleNamespace(
                    effective_user=SimpleNamespace(id=23),
                    effective_message=SimpleNamespace(text="Kuruba", reply_text=message.reply_text),
                ),
                SimpleNamespace(),
            )

            self.assertIsNone(store.get_conversation(23)[0])
            self.assertNotIn("specific_community", store.get_profile(23))
            self.assertIn("Potential match", message.reply_text.await_args_list[-1].args[0])

    async def test_optional_profile_fields_can_be_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "profile-skip.sqlite3")
            service = SchemeSathiBot(store, schemes=[scheme()])
            message = SimpleNamespace(reply_text=AsyncMock())
            await service._menu_action(message, 22, "profile")
            for _ in range(8):
                _, data = store.get_conversation(22)
                await service._process_profile_value(message, 22, "skip", data)

            self.assertIsNone(store.get_conversation(22)[0])
            self.assertNotIn("social_category", store.get_profile(22))

    async def test_document_checklist_tracks_status_and_returns_to_saved_page(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "document-checklist.sqlite3")
            service = SchemeSathiBot(
                store,
                schemes=[scheme(eligibility_caste=["SC"])],
            )
            message = SimpleNamespace(reply_text=AsyncMock())
            await service._document_checklist(message, 24, "test", 3)

            checklist = message.reply_text.await_args.args[0]
            self.assertIn("Caste/community certificate", checklist)
            markup = message.reply_text.await_args.kwargs["reply_markup"]
            status_callback = markup.inline_keyboard[0][0].callback_data
            self.assertEqual(status_callback, "document_status:test:0:available:3")

            update, query = self.make_callback("document-status", status_callback, 24, message)
            await service.callback(update, SimpleNamespace())

            query.answer.assert_awaited_once()
            self.assertEqual(
                store.get_document_statuses(24, "test")["Income certificate"],
                "available",
            )
            self.assertTrue(
                any(
                    button.callback_data == "recommendations_return:3"
                    for row in message.reply_text.await_args.kwargs["reply_markup"].inline_keyboard
                    for button in row
                )
            )

    async def test_edit_household_updates_only_selected_member(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "family-edit.sqlite3")
            store.update_profile(25, {"occupation": "student", "age": 20})
            store.add_family_member(25, "Parent", {"occupation": "farmer", "age": 50})
            member = store.get_family(25)[0]
            service = SchemeSathiBot(store, schemes=[scheme()])
            message = SimpleNamespace(reply_text=AsyncMock())
            update, query = self.make_callback(
                "family-edit", f"family_edit:{member['id']}", 25, message
            )

            await service.callback(update, SimpleNamespace())
            self.assertEqual(store.get_conversation(25)[0], "family")
            for value in (
                "Father",
                "52",
                "farmer",
                "school",
                "100000",
                "Karnataka",
                "rural",
                "male",
                "skip",
            ):
                _, data = store.get_conversation(25)
                await service._process_family_value(message, 25, value, data)

            self.assertEqual(store.get_family(25)[0]["name"], "Father")
            self.assertEqual(store.get_family(25)[0]["age"], 52)
            self.assertEqual(store.get_profile(25), {"occupation": "student", "age": 20})
            query.answer.assert_awaited_once()

    async def test_legacy_menu_callback_remains_routed_after_deployment(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "legacy-navigation.sqlite3")
            store.update_profile(5, {"state": "Karnataka", "occupation": "farmer"})
            service = SchemeSathiBot(store, schemes=[scheme()])
            service._family_summary = AsyncMock()
            message = SimpleNamespace(reply_text=AsyncMock())
            update, query = self.make_callback("legacy-menu-callback", "menu:family", 5, message)

            await service.callback(update, SimpleNamespace())

            query.answer.assert_awaited_once()
            service._family_summary.assert_awaited_once()
            self.assertIsNone(store.get_conversation(5)[0])


if __name__ == "__main__":
    unittest.main()
