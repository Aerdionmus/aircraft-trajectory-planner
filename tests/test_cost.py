from __future__ import annotations

import math

import pytest

from atp.aircraft.performance import MEDIUM_TWIN_JET
from atp.core.geometry import Vec2
from atp.environment.airspace import Airspace, GridSpec, GridState
from atp.environment.restrictions import CircularRestriction, RestrictionSet
from atp.environment.risk import ConstantRisk, GaussianHazard
from atp.environment.wind import UniformWind
from atp.planning.cost import CostModel, CostWeights, SegmentMetrics

SPEC = GridSpec(
    cells_x=10, cells_y=10, cell_size_nm=10.0, flight_levels=(280, 300, 320)
)


def model(**kwargs) -> CostModel:
    airspace = Airspace(spec=SPEC, **kwargs.pop("airspace", {}))
    weights = kwargs.pop(
        "weights", CostWeights(time_cost_per_hour=3000.0, fuel_cost_per_kg=0.8)
    )
    return CostModel(airspace, MEDIUM_TWIN_JET, weights, **kwargs)


def test_negative_weights_rejected():
    with pytest.raises(ValueError):
        CostWeights(time_cost_per_hour=-1.0)


def test_still_air_time_matches_tas():
    m = model()
    metrics = m.evaluate(GridState(0, 0, 0), GridState(1, 0, 0))
    assert metrics.feasible
    assert metrics.ground_distance_nm == pytest.approx(10.0)
    assert metrics.time_h == pytest.approx(10.0 / 450.0)
    assert metrics.ground_speed_kt == pytest.approx(450.0)


def test_tailwind_is_cheaper_than_headwind():
    tail = model(airspace={"wind": UniformWind(Vec2(80.0, 0.0))})
    head = model(airspace={"wind": UniformWind(Vec2(-80.0, 0.0))})
    eastbound = (GridState(0, 0, 0), GridState(1, 0, 0))
    assert tail.transition_cost(*eastbound)[0] < head.transition_cost(*eastbound)[0]
    assert tail.evaluate(*eastbound).ground_speed_kt == pytest.approx(530.0)


def test_hard_restriction_makes_transition_infeasible():
    restrictions = RestrictionSet(
        regions=(
            CircularRestriction(
                region_id="P-1", centre_nm=Vec2(25.0, 5.0), radius_nm=8.0
            ),
        )
    )
    m = model(airspace={"restrictions": restrictions})
    cost, metrics = m.transition_cost(GridState(1, 0, 0), GridState(2, 0, 0))
    assert not metrics.feasible
    assert cost == math.inf
    assert "P-1" in metrics.infeasible_reason


def test_risk_exposure_scales_with_time_in_the_field():
    risky = model(
        airspace={"risk": ConstantRisk(2.0)},
        weights=CostWeights(time_cost_per_hour=0.0, risk_cost_per_exposure=100.0),
    )
    metrics = risky.evaluate(GridState(0, 0, 0), GridState(1, 0, 0))
    assert metrics.risk_exposure == pytest.approx(2.0 * metrics.time_h)
    assert risky.price(metrics).risk == pytest.approx(
        100.0 * 2.0 * metrics.time_h
    )


def test_localised_hazard_only_charges_where_it_is():
    hazard = GaussianHazard(centre_nm=Vec2(95.0, 95.0), peak=5.0, sigma_nm=5.0)
    m = model(
        airspace={"risk": hazard},
        weights=CostWeights(risk_cost_per_exposure=100.0),
    )
    far = m.evaluate(GridState(0, 0, 0), GridState(1, 0, 0))
    near = m.evaluate(GridState(8, 9, 0), GridState(9, 9, 0))
    assert near.risk_exposure > far.risk_exposure * 100


def test_climb_rate_limit_rejects_unflyable_level_change():
    # 10 NM at 450 kt is 80 s; 2000 ft in 80 s is 1500 fpm, inside the limit.
    m = model()
    assert m.evaluate(GridState(0, 0, 0), GridState(1, 0, 1)).feasible
    # Shrink the cell so the same level change would need > 1800 fpm.
    tight = CostModel(
        Airspace(
            spec=GridSpec(
                cells_x=10, cells_y=10, cell_size_nm=3.0, flight_levels=(280, 300)
            )
        ),
        MEDIUM_TWIN_JET,
        CostWeights(),
    )
    metrics = tight.evaluate(GridState(0, 0, 0), GridState(1, 0, 1))
    assert not metrics.feasible
    assert "fpm" in metrics.infeasible_reason


def test_descent_credit_never_produces_a_negative_edge_cost():
    aggressive = MEDIUM_TWIN_JET.__class__(
        name="test",
        cruise_tas_kt=450.0,
        cruise_fuel_flow_kg_per_h=100.0,
        descent_fuel_credit_kg_per_1000ft=10_000.0,
    )
    m = CostModel(
        Airspace(spec=SPEC), aggressive, CostWeights(fuel_cost_per_kg=1.0)
    )
    metrics = m.evaluate(GridState(0, 0, 2), GridState(1, 0, 1))
    assert metrics.fuel_kg >= 0.0
    assert m.price(metrics).total >= 0.0


def test_speed_cost_lower_bound_is_not_exceeded_by_any_transition():
    """The bound is per NM of *horizontal* progress and covers the time, fuel
    and risk terms only; the distance price is charged against 3D length."""
    m = model(airspace={"wind": UniformWind(Vec2(60.0, 30.0))})
    bound = m.speed_cost_lower_bound_per_nm()
    for dx, dy in SPEC.moves:
        for dl in (0, 1, -1):
            a = GridState(4, 4, 1)
            b = GridState(4 + dx, 4 + dy, 1 + dl)
            cost, metrics = m.transition_cost(a, b)
            if metrics.feasible:
                horizontal = math.hypot(dx, dy) * SPEC.cell_size_nm
                priced = m.price(metrics)
                # The bound holds only together with the constant offset that
                # gives back the descent fuel credit, which is not proportional
                # to distance.
                assert priced.time + priced.fuel + priced.risk >= (
                    bound * horizontal - m.constant_cost_offset() - 1e-9
                )
                assert cost >= bound * horizontal - m.constant_cost_offset() - 1e-9


def test_descent_credit_alone_can_break_a_naive_per_nm_fuel_bound():
    """Why :meth:`CostModel.constant_cost_offset` exists.

    A descending transition burns less than ``ff_min * time``, so a bound built
    purely from a per-NM fuel floor is not a lower bound on it.
    """
    m = model()
    descending = m.evaluate(GridState(4, 4, 2), GridState(5, 4, 1))
    assert descending.feasible
    ff_min = MEDIUM_TWIN_JET.min_fuel_flow_over_levels_kg_per_h(SPEC.flight_levels)
    assert descending.fuel_kg < ff_min * descending.time_h
    assert m.constant_cost_offset() > 0.0


def test_constant_offset_is_zero_in_the_common_configurations():
    single_level = CostModel(
        Airspace(spec=GridSpec(cells_x=4, cells_y=4, cell_size_nm=10.0, flight_levels=(300,))),
        MEDIUM_TWIN_JET,
        CostWeights(fuel_cost_per_kg=0.8),
    )
    assert single_level.constant_cost_offset() == 0.0
    unpriced_fuel = model(weights=CostWeights(time_cost_per_hour=3000.0))
    assert unpriced_fuel.constant_cost_offset() == 0.0


def test_segment_metrics_accumulate():
    a = SegmentMetrics(True, 10.0, 0.0, 0.1, 200.0, 1.0, 0.0, 100.0, 0.5)
    b = SegmentMetrics(True, 30.0, 1000.0, 0.2, 400.0, 2.0, 5.0, 150.0, 0.2)
    total = a + b
    assert total.ground_distance_nm == pytest.approx(40.0)
    assert total.fuel_kg == pytest.approx(600.0)
    assert total.restriction_penalty == pytest.approx(5.0)
    # Distance-weighted mean ground speed.
    assert total.ground_speed_kt == pytest.approx((100 * 10 + 150 * 30) / 40)
