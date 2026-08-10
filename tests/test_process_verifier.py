"""CPU-only tests for deterministic public-state process verification."""

import unittest

from shopping_grpo.training.grpo.process_verifier import (
    PROCESS_VERIFIER_SCHEMA,
    canonical_action_key,
    canonical_public_anchor,
    exact_public_observation_hash,
    first_process_failure_step,
    verify_public_transition,
)

HOME = """[SHOPPING_OBSERVATION_V2]
page_type: search_home

搜索功能是否可用: True
可点击的按钮: []"""

SEARCH = """[SHOPPING_OBSERVATION_V2]
page_type: search_results
query: headphones
normalized_query: headphones
Page 1 of 2 (Total results: 21; ranks 1-20)
1|12345678|99|Acme|audio|wireless|First title
products_shown: 1

搜索功能是否可用: True
可点击的按钮: ["12345678", "back to search", "next >"]"""

DETAIL_INCOMPLETE = """[SHOPPING_OBSERVATION_V2]
page_type: product_detail
asin: 12345678
selected_options: {}
available_options: {"color": ["black", "white"]}

搜索功能是否可用: True
可点击的按钮: ["black", "white", "buy now", "back to search"]"""

DETAIL_READY = DETAIL_INCOMPLETE.replace(
    "selected_options: {}",
    'selected_options: {"color": "black"}',
)


class ProcessVerifierTest(unittest.TestCase):
    def test_exact_public_hash_preserves_actor_visible_bytes(self):
        self.assertNotEqual(
            exact_public_observation_hash(HOME),
            exact_public_observation_hash(HOME + "\n"),
        )

    def test_search_transition_is_legal_novel_evidence(self):
        result = verify_public_transition(
            HOME,
            SEARCH,
            tool_name="search_products",
            parameters={"query": "headphones"},
        )

        self.assertEqual(result["schema_version"], PROCESS_VERIFIER_SCHEMA)
        self.assertTrue(result["checks"]["legal_action"])
        self.assertTrue(result["checks"]["novel_evidence"])
        self.assertGreater(result["process_reward"], 0.0)
        self.assertFalse(any(result["failures"].values()))

    def test_real_structured_page_number_changes_public_anchor(self):
        next_page = SEARCH.replace("Page 1 of 2", "Page 2 of 2")

        result = verify_public_transition(
            SEARCH,
            next_page,
            tool_name="next_page",
        )

        self.assertTrue(result["checks"]["state_changed"])
        self.assertTrue(result["checks"]["novel_evidence"])

    def test_opening_unlisted_candidate_is_auditable_illegal_action(self):
        result = verify_public_transition(
            SEARCH,
            SEARCH,
            tool_name="open_product",
            parameters={"asin": "99999999"},
        )

        self.assertFalse(result["checks"]["legal_action"])
        self.assertTrue(result["failures"]["illegal_action"])
        self.assertEqual(
            result["guard_reason"],
            "click_not_in_previous_observation",
        )

    def test_missing_required_tool_argument_is_illegal(self):
        result = verify_public_transition(
            HOME,
            HOME,
            tool_name="search_products",
            parameters={},
        )

        self.assertTrue(result["failures"]["illegal_action"])
        self.assertEqual(
            result["guard_reason"],
            "schema_missing_arguments:query",
        )

    def test_option_selection_and_purchase_readiness_are_public_checks(self):
        selected = verify_public_transition(
            DETAIL_INCOMPLETE,
            DETAIL_READY,
            tool_name="select_option",
            parameters={"value": "black"},
        )
        purchase = verify_public_transition(
            DETAIL_READY,
            DETAIL_READY,
            tool_name="buy_now",
            terminal=True,
        )

        self.assertTrue(selected["checks"]["option_selection_progress"])
        self.assertTrue(purchase["checks"]["purchase_ready_before_action"])
        self.assertFalse(purchase["failures"]["premature_purchase"])
        self.assertEqual(purchase["process_reward"], 0.0)

    def test_premature_purchase_and_first_failure_step_are_reported(self):
        premature = verify_public_transition(
            DETAIL_INCOMPLETE,
            DETAIL_INCOMPLETE,
            tool_name="buy_now",
            terminal=True,
        )
        turn = {"verifier": premature}

        self.assertTrue(premature["failures"]["premature_purchase"])
        self.assertEqual(first_process_failure_step([turn]), 0)

    def test_same_public_decision_without_progress_is_repeat_failure(self):
        first = verify_public_transition(
            HOME,
            HOME,
            tool_name="think",
            parameters={"note": "wait"},
        )
        repeated = verify_public_transition(
            HOME,
            HOME,
            tool_name="think",
            parameters={"note": "wait"},
            prior_decisions=[
                (
                    canonical_public_anchor(HOME),
                    canonical_action_key("think", {"note": "wait"}),
                )
            ],
        )

        self.assertFalse(first["failures"]["repeated_no_progress"])
        self.assertTrue(first["failures"]["no_progress_action"])
        self.assertTrue(repeated["failures"]["repeated_no_progress"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
