"""Coordinated level-turn geometry.

Scope of the model (stated here rather than implied by the code):

* Constant-bank, coordinated, **level** turn.  No roll-in/roll-out time, no
  bank scheduling, no load-factor limit beyond the bank limit itself, and no
  turn/climb coupling (a real aircraft loses climb performance in a turn).
* The turn is charged in time, fuel and risk, and **never** in distance.  See
  :func:`corner_cut_nm` for why the geometric shortening a fly-by produces is
  measured but not priced.

Derivation
----------
For a coordinated level turn at bank angle ``phi`` and true airspeed ``V``, the
horizontal component of lift supplies the centripetal acceleration::

    tan(phi) = V^2 / (g R)      =>      R = V^2 / (g tan phi)
    omega    = V / R            =>      omega = g tan(phi) / V

With ``V`` in knots and ``g`` in NM/h^2 (:data:`atp.core.units.G_NM_PER_H2`),
``R`` comes out in NM and ``omega`` in rad/h:

    kt^2 / (NM/h^2)  = (NM/h)^2 h^2 / NM = NM
    (NM/h^2) / (NM/h) = 1/h

*Wind translates, it does not rotate.*  In a spatially uniform wind the ground
velocity is ``v(t) = V psi_hat(t) + w`` with ``w`` constant, so the ground track
of a constant-bank turn is a trochoid, not a circle -- its radius of curvature
varies continuously through the turn.  An earlier version of this module claimed
that radius was simply ``GS / omega``, with ``GS`` taken as the larger of the
two legs' ground speeds.  That is **not** a sound bound and has been replaced;
see :func:`ground_curvature_radius_bound_nm` for the corrected derivation.  The
short version: writing ``a = w . psi_hat`` for the wind component along the
*instantaneous* heading, the radius of curvature at that instant is

    R(t) = GS(t)^3 / (V omega (V + a))

not ``GS(t) / omega``.  The two coincide only when ``w = 0``; in a tailwind
component (``a > 0``) the true radius is *larger* than ``GS / omega``, so the
old formula could under-estimate the space a turn needs -- the wrong direction
for a legality gate that is elsewhere documented to over-block rather than
under-block.  A short numerical example: at ``V = 450`` kt with a 100 kt wind
aligned with the heading at the point of tightest curvature, the true radius is
about 22% larger than ``GS / omega`` computed from either leg's endpoint speed.

Fly-by legality
---------------
An arc of radius ``R`` tangent to two legs meeting at an angle ``dpsi`` needs a
tangent length ``T = R tan(dpsi / 2)`` measured back from the corner along each
leg.  The turn fits only if that length is available on both legs, and only
half of each leg may be claimed because the neighbouring corners need their own
tangent lengths::

    R tan(dpsi / 2) <= 0.5 * min(L_in, L_out)
"""

from __future__ import annotations

import math

from ..core.units import G_NM_PER_H2

#: Turns below this are treated as straight flight: zero cost, always legal.
NEGLIGIBLE_TURN_RAD: float = math.radians(0.5)


def turn_radius_nm(tas_kt: float, bank_deg: float) -> float:
    """Air-mass turn radius [NM] for a coordinated level turn."""
    if tas_kt <= 0.0:
        raise ValueError("tas_kt must be positive")
    if not 0.0 < bank_deg < 90.0:
        raise ValueError("bank_deg must lie strictly between 0 and 90")
    return tas_kt * tas_kt / (G_NM_PER_H2 * math.tan(math.radians(bank_deg)))


def turn_rate_rad_per_h(tas_kt: float, bank_deg: float) -> float:
    """Turn rate [rad/h] for a coordinated level turn."""
    if tas_kt <= 0.0:
        raise ValueError("tas_kt must be positive")
    if not 0.0 < bank_deg < 90.0:
        raise ValueError("bank_deg must lie strictly between 0 and 90")
    return G_NM_PER_H2 * math.tan(math.radians(bank_deg)) / tas_kt


def tangent_length_nm(radius_nm: float, delta_heading_rad: float) -> float:
    """``R tan(dpsi / 2)``: the leg length a fly-by arc consumes on each side.

    Infinite for a course reversal (``dpsi -> pi``), which is the correct answer:
    no finite leg can contain a 180 degree fly-by.
    """
    half = 0.5 * abs(delta_heading_rad)
    if half >= 0.5 * math.pi - 1e-12:
        return math.inf
    return radius_nm * math.tan(half)


def turn_fits(
    radius_nm: float,
    delta_heading_rad: float,
    leg_in_nm: float,
    leg_out_nm: float,
) -> bool:
    """Fly-by tangent-fit test (gate G1)."""
    if abs(delta_heading_rad) < NEGLIGIBLE_TURN_RAD:
        return True
    available = 0.5 * min(leg_in_nm, leg_out_nm)
    return tangent_length_nm(radius_nm, delta_heading_rad) <= available + 1e-9


def ground_curvature_radius_bound_nm(
    tas_kt: float, wind_kt: float, bank_deg: float
) -> float:
    """Sound upper bound on the ground-path radius of curvature reached at any
    instant of a coordinated turn, under a locally uniform wind of magnitude
    ``wind_kt`` (the same node-centre sample :meth:`CostModel.turn_metrics`
    already uses for both legs of the corner).

    Derivation.  With ``psi(t)`` rotating at the constant air-frame rate
    ``omega`` and ``w`` a fixed wind vector, the ground velocity and
    acceleration are

        v(t) = V psi_hat(t) + w              a(t) = V omega psi_hat_perp(t)

    where ``psi_hat_perp`` is ``psi_hat`` rotated a quarter turn in the
    direction of ``omega``.  The radius of curvature of a plane curve is
    ``R = |v|^3 / |v x a|``.  Writing ``x = cos(theta)`` for the cosine of the
    angle between ``w`` and ``psi_hat(t)``, so that ``w . psi_hat = |w| x`` and
    ``|v|^2 = V^2 + 2 V |w| x + |w|^2``:

        v x a = V omega (V + |w| x)
        R(x)  = (V^2 + 2 V |w| x + |w|^2)^(3/2) / (V omega (V + |w| x))

    For ``|w| <= V`` this is non-decreasing in ``x`` on ``[-1, 1]`` (the sign of
    its derivative reduces to ``2V^2 + V|w|x - |w|^2 >= 0``, which holds
    throughout that range whenever ``|w| <= V``), so its maximum over every
    heading the aircraft could point during the turn -- not just the two leg
    endpoints -- is attained at ``x = 1``, the instant the heading is aligned
    with the wind:

        R_max = (V + |w|)^2 / (V omega)

    This reduces to the exact still-air radius ``V / omega`` when ``w = 0``, and
    it can exceed ``GS / omega`` evaluated at either leg's endpoint speed by a
    factor of up to ``(V + |w|) / V`` -- the earlier formula's error.

    The ``|w| <= V`` precondition is what makes the velocity-space circle of
    radius ``V`` centred on ``w`` contain the origin, which is what keeps
    ``V + |w| x`` positive (the ground track never doubles back on itself)
    throughout the sweep.  Outside that regime -- a wind stronger than the
    aircraft -- no sound finite bound is established here, and ``math.inf`` is
    returned: an unbounded radius makes the fly-by gate reject the corner
    outright, the conservative choice, rather than accept it on an unproven
    number.  Every wind field and aircraft shipped in this package stays well
    inside the ``|w| < V`` regime; this only guards a case that is not
    currently reachable through the built-in scenarios or aircraft.
    """
    if tas_kt <= 0.0:
        raise ValueError("tas_kt must be positive")
    if wind_kt < 0.0:
        raise ValueError("wind_kt must be non-negative")
    if wind_kt >= tas_kt:
        return math.inf
    omega = turn_rate_rad_per_h(tas_kt, bank_deg)
    gs_max = tas_kt + wind_kt
    return gs_max * gs_max / (tas_kt * omega)


def turn_time_h(delta_heading_rad: float, turn_rate_rad_per_h_: float) -> float:
    """Time [h] to change heading by ``dpsi`` at the given rate."""
    if turn_rate_rad_per_h_ <= 0.0:
        raise ValueError("turn rate must be positive")
    return abs(delta_heading_rad) / turn_rate_rad_per_h_


def corner_cut_nm(radius_nm: float, delta_heading_rad: float) -> float:
    """``2 R tan(dpsi/2) - R dpsi``: how much shorter the fly-by arc is than the
    cornered path it replaces.

    Reported as a diagnostic and **never** priced.  Crediting it would produce
    Milestone 2 edge costs below their Milestone 1 values, which would break the
    monotone-refinement invariant that the inherited-admissibility argument in
    :mod:`atp.planning.heuristics` rests on, and could make an edge cost
    negative outright.
    """
    dpsi = abs(delta_heading_rad)
    if dpsi < NEGLIGIBLE_TURN_RAD:
        return 0.0
    tangent = tangent_length_nm(radius_nm, dpsi)
    if not math.isfinite(tangent):
        return math.inf
    return max(0.0, 2.0 * tangent - radius_nm * dpsi)
