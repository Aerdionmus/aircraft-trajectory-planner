"""M3-REQ-003 / M3-REQ-013: the speed envelope and its operating limits.

The envelope is the object that turns "the aircraft has a speed" into "the
planner has a decision".  Two things have to hold: the decision set must be
well-formed (ascending, non-empty, with a selectable reference speed), and the
operating limits must actually bind -- a speed outside them must be unavailable
rather than clamped, and must become unavailable at the altitude where the
limit crosses it rather than at an arbitrary one.
"""

from __future__ import annotations

import math

import pytest

from atp.aircraft.envelope import (
    SYNTHETIC_JET_ENVELOPE,
    SYNTHETIC_TURBOPROP_ENVELOPE,
    SpeedEnvelope,
)
from atp.core.speeds import cas_from_tas, mach_from_tas, tas_from_cas, tas_from_mach


# -- construction ------------------------------------------------------------
def test_a_fixed_envelope_offers_no_decision():
    envelope = SpeedEnvelope.fixed(450.0)
    assert envelope.num_options == 1
    assert envelope.is_fixed
    assert envelope.cruise_index == 0
    assert envelope.tas_kt(0) == 450.0
    assert envelope.min_planning_tas_kt == envelope.max_planning_tas_kt == 450.0


def test_a_multi_speed_envelope_reports_its_extremes_and_cruise_index():
    envelope = SpeedEnvelope((330.0, 390.0, 450.0), 390.0)
    assert not envelope.is_fixed
    assert envelope.num_options == 3
    assert envelope.cruise_index == 1
    assert envelope.min_planning_tas_kt == 330.0
    assert envelope.max_planning_tas_kt == 450.0
    assert envelope.indices() == (0, 1, 2)


@pytest.mark.parametrize(
    "speeds, cruise",
    [
        ((), 450.0),
        ((450.0, 330.0), 450.0),  # descending
        ((330.0, 330.0), 330.0),  # duplicate
        ((-10.0, 450.0), 450.0),  # non-positive
        ((330.0, 450.0), 400.0),  # cruise not selectable
    ],
)
def test_malformed_envelopes_are_rejected(speeds, cruise):
    with pytest.raises(ValueError):
        SpeedEnvelope(speeds, cruise)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_cas_kt": -1.0},
        {"max_cas_kt": 0.0},
        {"max_mach": -0.5},
        {"min_cas_kt": 300.0, "max_cas_kt": 200.0},
    ],
)
def test_malformed_limits_are_rejected(kwargs):
    with pytest.raises(ValueError):
        SpeedEnvelope((450.0,), 450.0, **kwargs)


def test_an_out_of_range_index_raises_rather_than_clamping():
    envelope = SpeedEnvelope((330.0, 450.0), 450.0)
    for bad in (-1, 2, 99):
        with pytest.raises(IndexError):
            envelope.tas_kt(bad)


# -- limits ------------------------------------------------------------------
def test_an_envelope_with_no_limits_admits_every_speed_everywhere():
    envelope = SpeedEnvelope((100.0, 450.0), 450.0)
    for altitude in (0.0, 20000.0, 41000.0):
        assert envelope.available_indices(altitude) == (0, 1)


def test_the_mach_limit_binds_at_the_altitude_the_conversion_predicts():
    """The limit must bite exactly where TAS crosses it, not approximately."""
    envelope = SpeedEnvelope((450.0, 480.0), 450.0, max_mach=0.82)
    altitude = 30000.0
    boundary = tas_from_mach(0.82, altitude)
    assert 480.0 < boundary  # both speeds legal here
    assert envelope.available_indices(altitude) == (0, 1)

    # Climb until 480 kt exceeds Mmo. The speed of sound falls with temperature,
    # so the Mach limit expressed in TAS falls with altitude up to the
    # tropopause and is constant above it.
    higher = 39000.0
    assert tas_from_mach(0.82, higher) < 480.0
    assert envelope.available_indices(higher) == (0,)
    assert not envelope.is_available(1, higher)


def test_the_low_speed_cas_limit_binds_at_altitude():
    envelope = SpeedEnvelope((300.0, 450.0), 450.0, min_cas_kt=200.0)
    # 200 kt CAS is about 319 kt TAS at FL300, so 300 kt TAS is below the limit.
    assert tas_from_cas(200.0, 30000.0) > 300.0
    assert envelope.available_indices(30000.0) == (1,)
    # Nearer sea level the same TAS is comfortably above the limit.
    assert envelope.available_indices(0.0) == (0, 1)


def test_the_high_speed_cas_limit_binds_at_low_altitude():
    envelope = SpeedEnvelope((450.0,), 450.0, max_cas_kt=340.0)
    assert cas_from_tas(450.0, 0.0) > 340.0
    assert envelope.available_indices(0.0) == ()
    assert envelope.available_indices(30000.0) == (0,)


def test_a_speed_whose_limits_cannot_be_evaluated_is_unavailable():
    """Outside the ISA range, or supersonic, the limits are not computable.  The
    conservative answer is 'not available', matching the package's rule that an
    unproven number blocks rather than permits."""
    envelope = SpeedEnvelope((450.0,), 450.0, max_mach=0.82)
    assert envelope.available_indices(90000.0) == ()
    # A `max_mach` of 2.0 is no longer constructible at all (see
    # test_max_mach_at_or_above_one_is_rejected below) -- the envelope's own
    # limit has to stay inside the domain the conversions are defined on. The
    # same "unevaluable, hence unavailable" behaviour is still reachable
    # through a *planning speed* that is itself supersonic at altitude, under
    # an ordinary subsonic limit: 700 kt is about Mach 1.22 at FL410.
    supersonic = SpeedEnvelope((700.0,), 700.0, max_mach=0.82)
    assert supersonic.available_indices(41000.0) == ()


def test_availability_is_cached_without_affecting_equality():
    a = SpeedEnvelope((450.0,), 450.0, max_mach=0.82)
    b = SpeedEnvelope((450.0,), 450.0, max_mach=0.82)
    a.available_indices(30000.0)
    assert a == b, "asking an envelope a question must not change what it is"
    assert a.available_indices(30000.0) == b.available_indices(30000.0)


# -- the shipped envelopes ---------------------------------------------------
def test_the_shipped_jet_envelope_contains_the_milestone_2_operating_point():
    """450 kt must stay selectable or the Milestone 2 comparison loses its
    anchor."""
    assert 450.0 in SYNTHETIC_JET_ENVELOPE.planning_tas_kt
    assert SYNTHETIC_JET_ENVELOPE.cruise_tas_kt == 450.0


def test_every_shipped_planning_speed_is_flyable_at_the_scenario_level():
    """FL300 is the level every Milestone 3 scenario uses; an envelope whose
    options were all illegal there would make the scenarios vacuous."""
    available = SYNTHETIC_JET_ENVELOPE.available_indices(30000.0)
    assert available == SYNTHETIC_JET_ENVELOPE.indices()


def test_the_jet_envelope_limits_are_not_vacuous():
    """If no reachable speed could ever violate a limit, the limits would be
    decoration.  A speed just outside the envelope must be refused."""
    envelope = SYNTHETIC_JET_ENVELOPE
    assert mach_from_tas(510.0, 30000.0) > envelope.max_mach
    assert not envelope.tas_is_within_limits(510.0, 30000.0)
    assert not envelope.tas_is_within_limits(300.0, 30000.0)


def test_the_turboprop_envelope_is_centred_on_its_own_cruise_speed():
    assert SYNTHETIC_TURBOPROP_ENVELOPE.cruise_tas_kt == 280.0
    assert SYNTHETIC_TURBOPROP_ENVELOPE.available_indices(20000.0)


def test_describe_is_explicit_about_provenance():
    described = SYNTHETIC_JET_ENVELOPE.describe()
    assert described["synthetic"] is True
    assert described["planning_tas_kt"] == list(
        SYNTHETIC_JET_ENVELOPE.planning_tas_kt
    )


# -- non-finite input validation (regression) --------------------------------
NON_FINITE = [math.nan, math.inf, -math.inf]


@pytest.mark.parametrize("bad", NON_FINITE)
def test_a_non_finite_planning_speed_is_rejected(bad):
    """NaN and infinity both slip past a bare ``<= 0.0`` check (`nan <= 0.0`
    and `inf <= 0.0` are both False in Python), so this is checked separately
    and first."""
    with pytest.raises(ValueError, match="finite"):
        SpeedEnvelope((330.0, bad, 450.0), 450.0)


@pytest.mark.parametrize("bad", NON_FINITE)
def test_a_non_finite_cruise_speed_is_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        SpeedEnvelope((330.0, 450.0), bad)


@pytest.mark.parametrize("limit_name", ["min_cas_kt", "max_cas_kt", "max_mach"])
@pytest.mark.parametrize("bad", NON_FINITE)
def test_a_non_finite_operating_limit_is_rejected(limit_name, bad):
    with pytest.raises(ValueError, match="finite"):
        SpeedEnvelope((330.0, 450.0), 450.0, **{limit_name: bad})


@pytest.mark.parametrize("bad_mach", [1.0, 1.2, 2.0])
def test_max_mach_at_or_above_one_is_rejected(bad_mach):
    """The subsonic stagnation relation `atp.core.speeds` implements is not
    defined at or above Mach 1, so a `max_mach` there would declare a limit
    the envelope could never actually evaluate."""
    with pytest.raises(ValueError, match="Mach 1"):
        SpeedEnvelope((330.0, 450.0), 450.0, max_mach=bad_mach)


def test_max_mach_just_below_one_is_accepted():
    envelope = SpeedEnvelope((330.0, 450.0), 450.0, max_mach=0.999)
    assert envelope.max_mach == 0.999


def test_fixed_parity_semantics_are_unchanged_by_the_stronger_validation():
    """The strengthened checks must not touch what ``.fixed()`` produces for
    an ordinary finite speed -- fixed-speed parity depends on this."""
    envelope = SpeedEnvelope.fixed(450.0)
    assert envelope.is_fixed
    assert envelope.planning_tas_kt == (450.0,)
    assert envelope.cruise_tas_kt == 450.0
    assert envelope.min_cas_kt is None
    assert envelope.max_cas_kt is None
    assert envelope.max_mach is None


# -- regression: ISA is load-bearing for the envelope, not decorative -------
def test_the_shipped_envelope_genuinely_narrows_between_fl300_and_fl390():
    """A compact, two-level demonstration that the atmosphere actually drives
    which speeds are legal, rather than merely being present as a utility
    library the envelope never really consults.

    At FL300 every shipped planning speed is legal. Climbing to FL390 removes
    480 kt because it now exceeds `max_mach = 0.82` -- the headline case this
    test is named for -- and, measured rather than assumed, it *also* removes
    330 and 360 kt because thinner air raises the CAS equivalent of a given TAS
    past `min_cas_kt = 200`. Both ends of the band move; this test asserts
    both rather than only the one the reviewer named, so the assertion cannot
    quietly pass while the other end silently regresses.
    """
    envelope = SYNTHETIC_JET_ENVELOPE
    speed_of_450 = envelope.planning_tas_kt.index(450.0)
    speed_of_480 = envelope.planning_tas_kt.index(480.0)

    at_fl300 = envelope.available_indices(30000.0)
    at_fl390 = envelope.available_indices(39000.0)

    assert at_fl300 == envelope.indices(), "every shipped speed must fly at FL300"
    assert speed_of_480 in at_fl300 and speed_of_480 not in at_fl390, (
        "480 kt must become illegal at FL390 under Mmo 0.82"
    )
    assert speed_of_450 in at_fl300 and speed_of_450 in at_fl390, (
        "450 kt, the Milestone 2 operating point, must remain legal at both"
    )
    assert len(at_fl390) < len(at_fl300), "the band must genuinely narrow"


# -- regression: a singleton envelope can still have limits to enforce ------
def test_has_limits_is_independent_of_is_fixed():
    """The two facts an envelope can carry -- "offers a decision" and "has
    constraints to enforce" -- are orthogonal, and this is the property that
    keeps them so."""
    plain_fixed = SpeedEnvelope.fixed(450.0)
    assert plain_fixed.is_fixed and not plain_fixed.has_limits

    limited_singleton = SpeedEnvelope((450.0,), 450.0, max_mach=0.60)
    assert limited_singleton.is_fixed and limited_singleton.has_limits

    unlimited_multi = SpeedEnvelope((330.0, 450.0), 450.0)
    assert not unlimited_multi.is_fixed and not unlimited_multi.has_limits

    limited_multi = SpeedEnvelope((330.0, 450.0), 450.0, max_mach=0.82)
    assert not limited_multi.is_fixed and limited_multi.has_limits


def test_a_singleton_envelope_with_a_limit_can_still_be_unavailable():
    """A one-option envelope is not automatically always flyable: declaring a
    restrictive limit must still bind, even though there is nothing to choose
    between."""
    envelope = SpeedEnvelope((450.0,), 450.0, max_mach=0.60)
    assert envelope.available_indices(30000.0) == ()
    assert not envelope.is_available(0, 30000.0)


def test_fixed_still_has_no_limits_by_construction():
    """.fixed() itself never declares a limit; only a hand-built singleton can."""
    assert not SpeedEnvelope.fixed(999.0).has_limits
