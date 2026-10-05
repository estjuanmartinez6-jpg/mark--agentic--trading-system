"""
simulator/checkpoint.py — Snapshot-Based Save/Resume for MARK III Replay (Req 12)

Enables:
  - Pause and resume replay at any candle
  - Branching experiments (replay same checkpoint with different params)
  - Debugging: isolate behavior at specific timestamps
  - Reproducible comparisons: same checkpoint + different seed

Checkpoint format: JSON (human-readable, version-controlled friendly).
NumPy RNG state serialized as base64-encoded bytes.

Saved state includes:
  - replay_index         (current M5 candle position)
  - sim_timestamp        (simulated unix time)
  - broker_state         (balance, equity, open positions)
  - equity_curve         (full history up to checkpoint)
  - rng_state            (broker's RandomState for determinism)
  - scorer_cooldowns     (prevents cooldown reset on resume)
  - health_contexts      (frozen entry contexts for open trades)
  - config               (full replay configuration)
"""
from __future__ import annotations

import base64
import json
import logging
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("Sim.Checkpoint")

# Checkpoint file version — increment when schema changes
CHECKPOINT_VERSION = "1.0"


@dataclass
class ReplaySnapshot:
    """
    Complete, serializable state of a replay session at a point in time.

    All fields must be JSON-serializable (or wrapped in helpers below).
    """
    version: str = CHECKPOINT_VERSION

    # ── Replay position ──────────────────────────────────────────────
    replay_index: int = 0               # Current M5 candle index in source_df
    sim_timestamp: float = 0.0          # Simulated Unix timestamp
    sim_timestamp_str: str = ""         # Human-readable UTC string

    # ── Account state ────────────────────────────────────────────────
    balance: float = 0.0
    equity: float = 0.0
    initial_balance: float = 0.0

    # ── Open positions ───────────────────────────────────────────────
    open_positions: List[Dict] = field(default_factory=list)

    # ── Closed trades history ────────────────────────────────────────
    closed_trades: List[Dict] = field(default_factory=list)

    # ── Equity curve ─────────────────────────────────────────────────
    # List of [sim_timestamp, equity] pairs
    equity_curve: List[List[float]] = field(default_factory=list)

    # ── Randomness state (Req 3) ─────────────────────────────────────
    # np.RandomState.get_state() serialized as base64 string
    rng_state_b64: str = ""

    # ── Scorer cooldowns ─────────────────────────────────────────────
    # {symbol: last_signal_unix_timestamp}
    scorer_cooldowns: Dict[str, float] = field(default_factory=dict)

    # ── Health monitor contexts ──────────────────────────────────────
    # Per-ticket entry context dicts for open trades
    health_contexts: Dict[int, Dict] = field(default_factory=dict)

    # ── Session configuration ────────────────────────────────────────
    config: Dict[str, Any] = field(default_factory=dict)

    # ── Metadata ─────────────────────────────────────────────────────
    created_at: str = ""
    candle_count: int = 0


class Checkpointer:
    """
    Saves and loads ReplaySnapshot objects as JSON files.

    Usage:
        cp = Checkpointer(directory="simulator/results/run_001")
        cp.save(snapshot, label="step_0100")
        # ... later ...
        snap = cp.load("simulator/results/run_001/checkpoint_step_0100.json")
    """

    def __init__(self, directory: str) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── Save ─────────────────────────────────────────────────────────

    def save(
        self,
        snapshot: ReplaySnapshot,
        label: str = None,
    ) -> str:
        """
        Serialize and save a snapshot to disk.

        Args:
            snapshot: The ReplaySnapshot to save.
            label:    Optional label for the checkpoint filename.
                      Defaults to "step_{replay_index:04d}".

        Returns:
            Absolute path of the saved checkpoint file.
        """
        if label is None:
            label = f"step_{snapshot.replay_index:06d}"

        filename = f"checkpoint_{label}.json"
        path = self._dir / filename

        # Add metadata
        snapshot.created_at = datetime.now(timezone.utc).isoformat()

        payload = self._to_dict(snapshot)

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            logger.info(
                f"[Checkpoint] Saved → {path.name} "
                f"(index={snapshot.replay_index}, "
                f"balance=${snapshot.balance:.2f})"
            )
            return str(path)
        except Exception as exc:
            logger.error(f"[Checkpoint] Save failed: {exc}")
            raise

    # ── Load ─────────────────────────────────────────────────────────

    def load(self, path: str) -> ReplaySnapshot:
        """
        Load a checkpoint from a JSON file.

        Args:
            path: Path to the checkpoint JSON file.

        Returns:
            ReplaySnapshot with all state restored.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

        try:
            with open(p, "r", encoding="utf-8") as f:
                payload = json.load(f)

            version = payload.get("version", "unknown")
            if version != CHECKPOINT_VERSION:
                logger.warning(
                    f"[Checkpoint] Version mismatch: "
                    f"file={version}, current={CHECKPOINT_VERSION}. "
                    f"Some fields may not restore correctly."
                )

            snap = self._from_dict(payload)
            logger.info(
                f"[Checkpoint] Loaded ← {p.name} "
                f"(index={snap.replay_index}, "
                f"ts={snap.sim_timestamp_str})"
            )
            return snap
        except Exception as exc:
            logger.error(f"[Checkpoint] Load failed: {exc}")
            raise

    def list_checkpoints(self) -> List[str]:
        """Returns sorted list of checkpoint file paths in this directory."""
        paths = sorted(self._dir.glob("checkpoint_*.json"))
        return [str(p) for p in paths]

    def latest(self) -> Optional[str]:
        """Returns path to the most recent checkpoint, or None."""
        checkpoints = self.list_checkpoints()
        return checkpoints[-1] if checkpoints else None

    # ── RNG state serialization ──────────────────────────────────────

    @staticmethod
    def encode_rng_state(rng: np.random.RandomState) -> str:
        """
        Serialize numpy RandomState to a base64 string for JSON storage.
        Preserves the full RNG state for exact deterministic resume.
        """
        try:
            state_bytes = pickle.dumps(rng.get_state())
            return base64.b64encode(state_bytes).decode("ascii")
        except Exception as exc:
            logger.warning(f"[Checkpoint] Could not encode RNG state: {exc}")
            return ""

    @staticmethod
    def decode_rng_state(b64_str: str) -> Optional[tuple]:
        """
        Deserialize base64-encoded RNG state back to a tuple for
        np.random.RandomState.set_state().
        """
        if not b64_str:
            return None
        try:
            state_bytes = base64.b64decode(b64_str.encode("ascii"))
            return pickle.loads(state_bytes)
        except Exception as exc:
            logger.warning(f"[Checkpoint] Could not decode RNG state: {exc}")
            return None

    # ── Serialization helpers ────────────────────────────────────────

    @staticmethod
    def _to_dict(snap: ReplaySnapshot) -> dict:
        """Convert ReplaySnapshot to a JSON-safe dict."""
        d = {
            "version":           snap.version,
            "replay_index":      snap.replay_index,
            "sim_timestamp":     snap.sim_timestamp,
            "sim_timestamp_str": snap.sim_timestamp_str,
            "balance":           snap.balance,
            "equity":            snap.equity,
            "initial_balance":   snap.initial_balance,
            "open_positions":    snap.open_positions,
            "closed_trades":     snap.closed_trades,
            "equity_curve":      snap.equity_curve,
            "rng_state_b64":     snap.rng_state_b64,
            "scorer_cooldowns":  snap.scorer_cooldowns,
            "health_contexts":   {str(k): v for k, v in snap.health_contexts.items()},
            "config":            snap.config,
            "created_at":        snap.created_at,
            "candle_count":      snap.candle_count,
        }
        return d

    @staticmethod
    def _from_dict(d: dict) -> ReplaySnapshot:
        """Reconstruct a ReplaySnapshot from a loaded dict."""
        health_contexts = {
            int(k): v
            for k, v in d.get("health_contexts", {}).items()
        }
        return ReplaySnapshot(
            version=d.get("version", CHECKPOINT_VERSION),
            replay_index=d.get("replay_index", 0),
            sim_timestamp=d.get("sim_timestamp", 0.0),
            sim_timestamp_str=d.get("sim_timestamp_str", ""),
            balance=d.get("balance", 0.0),
            equity=d.get("equity", 0.0),
            initial_balance=d.get("initial_balance", 0.0),
            open_positions=d.get("open_positions", []),
            closed_trades=d.get("closed_trades", []),
            equity_curve=d.get("equity_curve", []),
            rng_state_b64=d.get("rng_state_b64", ""),
            scorer_cooldowns=d.get("scorer_cooldowns", {}),
            health_contexts=health_contexts,
            config=d.get("config", {}),
            created_at=d.get("created_at", ""),
            candle_count=d.get("candle_count", 0),
        )
