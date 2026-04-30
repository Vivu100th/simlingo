"""Unit tests for MTID core geometry, risk, and schema helpers."""

from __future__ import annotations

import math
import unittest

import numpy as np

from mtid.core import (
    actor_crossing_rollout,
    actor_summary,
    approximate_ttc,
    choose_best_candidate,
    classify_actor,
    collision_rectangles_overlap,
    equal_spacing_route,
    inverse_conversion_2d,
    min_distance,
    score_candidate,
    transform_waypoints,
    validate_dreamer_payload,
)


class CoreGeometryTests(unittest.TestCase):
    def test_inverse_conversion_rotates_global_point_to_ego_frame(self) -> None:
        local = inverse_conversion_2d([0.0, 1.0], [0.0, 0.0], math.pi / 2.0)
        np.testing.assert_allclose(local, [1.0, 0.0], atol=1e-6)

    def test_equal_spacing_route_returns_fixed_shape_from_empty_or_short_route(self) -> None:
        empty = equal_spacing_route([], num_points=4)
        np.testing.assert_allclose(empty, np.zeros((4, 2)))

        route = equal_spacing_route([[2.0, 0.0], [5.0, 0.0]], num_points=4)
        self.assertEqual(route.shape, (4, 2))
        np.testing.assert_allclose(route[:, 1], 0.0, atol=1e-6)
        self.assertGreater(route[-1, 0], route[0, 0])

    def test_transform_waypoints_creates_expected_counterfactuals(self) -> None:
        base = np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=float)
        slow = transform_waypoints(base, "slow")
        left = transform_waypoints(base, "nudge_left")

        self.assertLess(slow[-1, 0], base[-1, 0])
        self.assertGreater(left[-1, 1], base[-1, 1])
        with self.assertRaises(ValueError):
            transform_waypoints(base, "teleport")


class ActorAndRiskTests(unittest.TestCase):
    def test_actor_classification_supports_mixed_traffic_types(self) -> None:
        self.assertEqual(classify_actor({"class": "walker"}), "pedestrian")
        self.assertEqual(
            classify_actor({"class": "car", "type_id": "vehicle.yamaha.yzf", "number_of_wheels": 2}),
            "two_wheeler",
        )
        self.assertEqual(classify_actor({"class": "car", "number_of_wheels": 4}), "vehicle")
        self.assertIsNone(classify_actor({"class": "ego_car"}))

    def test_actor_summary_normalizes_required_fields(self) -> None:
        summary = actor_summary(
            {
                "id": 7,
                "class": "walker",
                "position": [4.0, -1.0, 0.0],
                "speed": 1.5,
                "yaw": 0.25,
                "distance": 4.2,
            }
        )

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["actor_type"], "pedestrian")
        self.assertEqual(summary["position"], [4.0, -1.0])
        self.assertEqual(summary["speed"], 1.5)

    def test_crossing_rollout_moves_actor_toward_road_center(self) -> None:
        actor = {"position": [8.0, 3.0], "speed": 1.0}
        rollout = actor_crossing_rollout(actor, steps=3, dt=0.5)
        self.assertLess(rollout[-1, 1], actor["position"][1])

    def test_ttc_ignores_far_parallel_paths_but_catches_same_lane_conflict(self) -> None:
        ego = np.asarray([[float(i), 0.0] for i in range(10)], dtype=float)
        actor = np.asarray([[10.0 - float(i), 0.0] for i in range(10)], dtype=float)
        far_actor = np.asarray([[20.0, 10.0] for _ in range(10)], dtype=float)

        self.assertLessEqual(min_distance(ego, actor), 1.0)
        self.assertLess(approximate_ttc(ego, actor, 0.25), 2.0)
        self.assertGreater(min_distance(ego, far_actor), 10.0)
        self.assertTrue(math.isinf(approximate_ttc(ego, far_actor, 0.25)))

    def test_score_candidate_marks_close_pedestrian_interaction_unsafe(self) -> None:
        ego = np.asarray([[float(i), 0.0] for i in range(10)], dtype=float)
        actor = np.asarray([[10.0 - float(i), 0.0] for i in range(10)], dtype=float)

        result = score_candidate(ego, [({"id": 1, "actor_type": "pedestrian"}, actor)], 0.25)
        self.assertFalse(result["safe"])
        self.assertGreater(result["risk_score"], 0.0)
        self.assertEqual(result["actor_id"], 1)

    def test_collision_rectangles_overlap_is_conservative(self) -> None:
        self.assertTrue(collision_rectangles_overlap([0, 0], [1, 1], [1.5, 0], [1, 1]))
        self.assertFalse(collision_rectangles_overlap([0, 0], [1, 1], [3.5, 0], [1, 1]))


class CandidateAndSchemaTests(unittest.TestCase):
    def test_choose_best_candidate_prefers_safe_option_before_risk_sorting(self) -> None:
        unsafe = (
            "keep",
            np.asarray([[1.0, 0.0], [2.0, 0.0]]),
            {"safe": False, "risk_score": 0.1, "comfort": 0.0, "progress": 2.0},
        )
        safe = (
            "slow",
            np.asarray([[0.5, 0.0], [1.0, 0.0]]),
            {"safe": True, "risk_score": 0.4, "comfort": 0.0, "progress": 1.0},
        )

        action, waypoints, score = choose_best_candidate([unsafe, safe])
        self.assertEqual(action, "slow")
        np.testing.assert_allclose(waypoints, safe[1])
        self.assertIs(score, safe[2])

    def test_validate_dreamer_payload_accepts_mtid_contract_and_reports_missing_keys(self) -> None:
        option = {
            "mode": "jaywalker_crossing",
            "waypoints": [[1.0, 0.0], [2.0, 0.0]],
            "route": [[1.0, 0.0], [2.0, 0.0]],
            "rgb_path": "rgb/0000.jpg",
            "allowed": True,
            "info": {"candidate_action": "slow"},
            "route_reasoning": "risk-aware slowdown",
            "dreamer_instruction": ["Slow down."],
            "instructions_templates": ["Slow down."],
            "templates_placeholders": [{}],
            "dreamer_answer_safety": "Following the given instruction. Waypoints:",
            "safe_to_execute": True,
        }
        self.assertEqual(validate_dreamer_payload({"jaywalker_crossing": [option]}), [])

        broken = dict(option)
        broken.pop("route")
        errors = validate_dreamer_payload({"jaywalker_crossing": [broken]})
        self.assertTrue(any("missing keys" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
