"""
execution_engine/regime_profile.py — Regime Execution Profiles for MARK III v2.4

Central configuration hub that maps each market regime to specific
execution parameters. Every module that needs regime-aware behavior
queries this module instead of using hardcoded values.

This is a PURE DATA module — no logic, no side effects, no state.
Just a parameter table and a lookup function.

Design principles:
  - The profile is frozen at entry time (immutable snapshot)
  - Management modules NEVER recalculate the profile mid-trade
  - All parameters are deterministic and auditable
  - The NORMAL profile matches exact v2.2 defaults (zero-risk upgrade)

Regime Parameter Summary:
  ┌─────────────┬───────┬──────┬──────┬──────┬──────┬──────┬───────┬──────┬──────┐
  │ Regime      │ Score │ Fact │ SL×  │ TP×  │ TrAc │ TrDi │ Grace │ Degr │ Crit │
  ├─────────────┼───────┼──────┼──────┼──────┼──────┼──────┼───────┼──────┼──────┤
  │ DEAD        │  999  │  —   │  —   │  —   │  —   │  —   │   60  │  —   │  —   │
  │ SLOW_TREND  │   45  │  3   │ 1.2  │ 2.0  │ 0.5  │ 0.8  │   90  │  55  │  35  │
  │ NORMAL      │   50  │  3   │ 1.5  │ 2.5  │ 0.8  │ 1.0  │  120  │  60  │  40  │
  │ FAST        │   65  │  3   │ 1.8  │ 3.0  │ 1.2  │ 1.5  │  180  │  50  │  30  │
  │ EXPLOSIVE   │   70  │  4   │ 2.0  │ 3.5  │ 1.5  │ 2.0  │  180  │  45  │  25  │
  └─────────────┴───────┴──────┴──────┴──────┴──────┴──────┴───────┴──────┴──────┘

  Score  = Minimum signal score to trade
  Fact   = Minimum independent factors required
  SL×    = Stop-loss distance as ATR multiplier
  TP×    = Take-profit distance as ATR multiplier
  TrAc   = Trailing activation (ATR distance in profit before trail starts)
  TrDi   = Trailing distance (how far behind price the trail follows, in ATR)
  Grace  = Health monitor grace period (seconds)
  Degr   = Health score below which state becomes DEGRADING
  Crit   = Health score below which state becomes CRITICAL

v2.4 Changes:
  - NORMAL min_score: 60 → 50 (unlock multi-factor setups that were
    structurally unreachable when Trend factor = 0)
  - SLOW_TREND min_score: 55 → 45 (same rationale for slow markets)
  - FAST / EXPLOSIVE unchanged (conservative approach)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import settings
from monitoring.logger import get_logger

logger = get_logger("RegimeProfile")


@dataclass(frozen=True)
class RegimeProfile:
    """
    Immutable execution profile for a specific market regime.

    Frozen at signal evaluation time. Once attached to a trade,
    the profile NEVER changes for the life of that trade.
    """
    # ── Identity ────────────────────────────────────────────────
    regime: str                    # "DEAD", "SLOW_TREND", "NORMAL", "FAST", "EXPLOSIVE"

    # ── Entry Quality ───────────────────────────────────────────
    min_score: int                 # Minimum signal score to trade
    min_factors: int               # Minimum independent factors required

    # ── SL/TP Behavior (ATR multipliers) ────────────────────────
    sl_atr_mult: float             # Stop-loss distance = ATR × this
    tp_atr_mult: float             # Take-profit distance = ATR × this

    # ── Trailing Stop (ATR-based) ───────────────────────────────
    trail_activation_atr: float    # How far in profit before trailing starts
    trail_distance_atr: float      # How far behind price the trail follows

    # ── Health Monitor ──────────────────────────────────────────
    grace_period_sec: int          # Seconds before health scoring begins
    health_degrading: int          # Score threshold → DEGRADING state
    health_critical: int           # Score threshold → CRITICAL state

    def summary(self) -> str:
        """One-line summary for log output."""
        return (
            f"Profile={self.regime} │ "
            f"MinScore={self.min_score} │ "
            f"SL={self.sl_atr_mult}×ATR │ "
            f"TP={self.tp_atr_mult}×ATR │ "
            f"Trail={self.trail_activation_atr}/{self.trail_distance_atr}×ATR │ "
            f"Grace={self.grace_period_sec}s │ "
            f"Degr={self.health_degrading} │ "
            f"Crit={self.health_critical}"
        )


# ═══════════════════════════════════════════════════════════════════
# REGIME PROFILE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════
# Each profile is calibrated to match the expected behavior of its
# regime. See module docstring for design rationale per column.

_PROFILES = {
    "DEAD": RegimeProfile(
        regime="DEAD",
        min_score=999,          # Effectively blocks all entries
        min_factors=6,
        sl_atr_mult=1.5,       # Irrelevant (never enters)
        tp_atr_mult=2.5,
        trail_activation_atr=0.0,
        trail_distance_atr=0.0,
        grace_period_sec=60,
        health_degrading=60,
        health_critical=40,
    ),

    "SLOW_TREND": RegimeProfile(
        regime="SLOW_TREND",
        min_score=50,           # v2.5: 45→50
        min_factors=3,
        sl_atr_mult=1.2,       # Small noise envelope → tighter SL
        tp_atr_mult=2.0,       # Limited travel → conservative TP
        trail_activation_atr=0.5,   # Lock gains quickly
        trail_distance_atr=0.8,     # Tight trail (orderly moves)
        grace_period_sec=90,
        health_degrading=35,   # Must be < min_score (45)
        health_critical=20,
    ),

    "NORMAL": RegimeProfile(
        regime="NORMAL",
        min_score=55,           # v2.5: 50→55
        min_factors=3,
        sl_atr_mult=1.5,       # v2.2 default
        tp_atr_mult=2.5,       # v2.2 default
        trail_activation_atr=0.8,
        trail_distance_atr=1.0,
        grace_period_sec=120,
        health_degrading=40,   # Must be < min_score (50)
        health_critical=25,    

    ),

    "FAST": RegimeProfile(
        regime="FAST",
        min_score=65,           # Higher bar — more fakeouts
        min_factors=3,
        sl_atr_mult=1.8,       # Wider breathing room
        tp_atr_mult=3.0,       # Larger expected moves
        trail_activation_atr=1.2,   # Let runners develop
        trail_distance_atr=1.5,     # Loose trail (violent pullbacks)
        grace_period_sec=180,
        health_degrading=50,   # More tolerant — drawdowns expected
        health_critical=30,
    ),

    "EXPLOSIVE": RegimeProfile(
        regime="EXPLOSIVE",
        min_score=70,           # Maximum confluence demanded
        min_factors=4,
        sl_atr_mult=2.0,       # Widest — extreme whipsaws
        tp_atr_mult=3.5,       # Widest — breakout extensions
        trail_activation_atr=1.5,   # Loosest activation
        trail_distance_atr=2.0,     # Loosest trail
        grace_period_sec=180,
        health_degrading=45,   # Most tolerant
        health_critical=25,
    ),
}

# The NORMAL profile serves as the universal fallback
_DEFAULT_PROFILE = _PROFILES["NORMAL"]


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════

def get_profile(regime: str) -> RegimeProfile:
    """
    Returns the execution profile for the given regime.

    If REGIME_AWARE_EXECUTION is disabled in settings, always returns
    the NORMAL profile (exact v2.2 behavior).

    Args:
        regime: Market regime string from MomentumSpeed.

    Returns:
        Frozen RegimeProfile with all execution parameters.
    """
    # ── Toggle check: disabled → always NORMAL ──────────────────
    if not getattr(settings, "REGIME_AWARE_EXECUTION", True):
        return _DEFAULT_PROFILE

    profile = _PROFILES.get(regime, _DEFAULT_PROFILE)
    return profile


def get_all_profiles() -> dict:
    """Returns all defined profiles (for diagnostics/logging)."""
    return dict(_PROFILES)
