"""Tests for helicon.optimize.pareto."""

import numpy as np
import pytest

from helicon.optimize.pareto import (
    ParetoResult,
    hypervolume_indicator,
    is_dominated,
    pareto_front,
)


class TestIsDominated:
    def test_single_point_not_dominated(self):
        costs = np.array([[1.0, 2.0]])
        dominated = is_dominated(costs)
        assert not dominated[0]

    def test_two_points_one_dominated(self):
        # Point 0 dominates point 1: [1,1] vs [2,2]
        costs = np.array([[1.0, 1.0], [2.0, 2.0]])
        dominated = is_dominated(costs)
        assert not dominated[0]
        assert dominated[1]

    def test_two_nondominated_points(self):
        # Trade-off: [1,3] vs [3,1] — neither dominates
        costs = np.array([[1.0, 3.0], [3.0, 1.0]])
        dominated = is_dominated(costs)
        assert not dominated[0]
        assert not dominated[1]

    def test_three_points_one_dominated(self):
        costs = np.array([
            [1.0, 2.0],  # not dominated
            [3.0, 1.0],  # not dominated
            [2.0, 2.0],  # dominated by [1,2]
        ])
        dominated = is_dominated(costs)
        assert not dominated[0]
        assert not dominated[1]
        assert dominated[2]

    def test_identical_points_not_dominated(self):
        # Identical points do NOT dominate each other (requires strictly better in one)
        costs = np.array([[1.0, 1.0], [1.0, 1.0]])
        dominated = is_dominated(costs)
        assert not dominated[0]
        assert not dominated[1]

    def test_three_objective_dominance(self):
        costs = np.array([
            [1.0, 1.0, 1.0],  # dominates all others
            [2.0, 2.0, 2.0],  # dominated
            [1.5, 0.5, 1.5],  # not dominated by [1,1,1]: better in obj 1
        ])
        dominated = is_dominated(costs)
        assert not dominated[0]
        assert dominated[1]
        assert not dominated[2]

    def test_returns_bool_array(self):
        costs = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = is_dominated(costs)
        assert result.dtype == bool
        assert result.shape == (2,)


class TestParetoFront:
    def test_returns_pareto_result(self):
        costs = np.array([[1.0, 2.0], [3.0, 1.0]])
        result = pareto_front(costs)
        assert isinstance(result, ParetoResult)

    def test_both_tradeoff_points_on_front(self):
        costs = np.array([[1.0, 3.0], [3.0, 1.0], [2.0, 2.0]])
        result = pareto_front(costs)
        # [2,2] is dominated by both [1,3] and [3,1]... actually [1,3] doesn't dominate [2,2] since 3>2.
        # [1,3]: obj0=1 < 2, but obj1=3 > 2, so no dominance
        # [3,1]: obj0=3 > 2, obj1=1 < 2, no dominance
        # None dominates [2,2] — all three on front
        assert result.front_mask.sum() == 3

    def test_one_point_dominates_all(self):
        costs = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        result = pareto_front(costs)
        assert result.front_indices.tolist() == [0]
        assert result.front_mask[0]
        assert not result.front_mask[1]
        assert not result.front_mask[2]

    def test_front_costs_property(self):
        costs = np.array([[1.0, 3.0], [3.0, 1.0], [5.0, 5.0]])
        result = pareto_front(costs)
        # [5,5] is dominated
        assert result.front_costs.shape[0] == 2
        np.testing.assert_array_equal(result.front_costs, costs[result.front_indices])

    def test_maximization_via_negation(self):
        # Maximize thrust (negative = minimize)
        thrust = np.array([5.0, 3.0, 4.0, 3.0])
        eta = np.array([0.6, 0.8, 0.7, 0.9])
        costs = np.column_stack([-thrust, -eta])
        result = pareto_front(costs)
        # [5,0.6], [4,0.7], [3,0.9] on front; [3,0.8] dominated by [3,0.9]
        assert 0 in result.front_indices  # thrust=5
        assert 3 in result.front_indices  # eta=0.9
        assert 1 not in result.front_indices  # [3,0.8] dominated by [3,0.9]

    def test_single_point(self):
        costs = np.array([[2.5, 1.0]])
        result = pareto_front(costs)
        assert result.front_mask[0]
        assert len(result.front_indices) == 1

    def test_costs_preserved(self):
        costs = np.array([[1.0, 2.0], [3.0, 0.5]])
        result = pareto_front(costs)
        np.testing.assert_array_equal(result.costs, costs)


class TestHypervolumeIndicator:
    def test_single_point(self):
        front = np.array([[1.0, 1.0]])
        ref = np.array([2.0, 2.0])
        hv = hypervolume_indicator(front, ref)
        assert abs(hv - 1.0) < 1e-10

    def test_two_tradeoff_points(self):
        front = np.array([[1.0, 3.0], [3.0, 1.0]])
        ref = np.array([4.0, 4.0])
        # Sort by obj0: [1,3], [3,1]
        # width1 = 4-1=3, height1 = 4-3=1 → 3
        # width2 = 4-3=1, height2 = 3-1=2 → 2 (prev_y = min(4,3)=3)
        # total = 3 + 2 = 5
        hv = hypervolume_indicator(front, ref)
        assert abs(hv - 5.0) < 1e-10

    def test_dominated_point_excluded_hv(self):
        # [1,1] dominates [2,2] — compute HV for front containing only [1,1]
        front = np.array([[1.0, 1.0]])
        ref = np.array([3.0, 3.0])
        hv = hypervolume_indicator(front, ref)
        assert abs(hv - 4.0) < 1e-10  # (3-1)*(3-1) = 4

    def test_wrong_n_objectives_raises(self):
        front = np.array([[1.0, 2.0, 3.0]])
        ref = np.array([5.0, 5.0, 5.0])
        with pytest.raises(ValueError, match="2 objectives"):
            hypervolume_indicator(front, ref)

    def test_hv_increases_with_better_front(self):
        ref = np.array([10.0, 10.0])
        front_a = np.array([[3.0, 3.0]])
        front_b = np.array([[2.0, 2.0]])  # better (dominates a)
        hv_a = hypervolume_indicator(front_a, ref)
        hv_b = hypervolume_indicator(front_b, ref)
        assert hv_b > hv_a
