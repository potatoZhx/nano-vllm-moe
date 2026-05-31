import unittest

from benchmarks.scripts.spec_verify_expert_count_stats import _acceptance_stats


class TestAcceptancePositionStats(unittest.TestCase):
    def test_acceptance_stats_include_per_draft_position_rates(self):
        profile = {
            "spec_step_traces": [
                {
                    "sequences": [
                        {"drafted_tokens": 4, "accepted_draft_tokens": 0, "rejected": True},
                        {"drafted_tokens": 4, "accepted_draft_tokens": 1, "rejected": True},
                    ]
                },
                {
                    "sequences": [
                        {"drafted_tokens": 4, "accepted_draft_tokens": 4, "rejected": False},
                        {"drafted_tokens": 2, "accepted_draft_tokens": 2, "rejected": False},
                    ]
                },
                {
                    "sequences": [
                        {"drafted_tokens": 0, "accepted_draft_tokens": 0, "rejected": False},
                    ]
                },
            ]
        }

        stats = _acceptance_stats(profile)

        self.assertEqual(
            stats["draft_position_acceptance"],
            [
                {"position": 1, "drafted_count": 4, "accepted_count": 3, "acceptance_rate": 0.75},
                {"position": 2, "drafted_count": 4, "accepted_count": 2, "acceptance_rate": 0.5},
                {
                    "position": 3,
                    "drafted_count": 3,
                    "accepted_count": 1,
                    "acceptance_rate": 1.0 / 3.0,
                },
                {
                    "position": 4,
                    "drafted_count": 3,
                    "accepted_count": 1,
                    "acceptance_rate": 1.0 / 3.0,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
