"""
monitoring/notifier.py — Sistema de notificaciones para MARK I

Canales disponibles:
  1. Consola (siempre activo, con color y sonido en Windows)
  2. Telegram (si TELEGRAM_ENABLED=True en settings)
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import requests

from config import settings
from monitoring.logger import get_logger
from utils.helpers import direction_emoji, fmt_pnl, fmt_price

logger = get_logger("Notifier")


class Notifier:
    """
    Envía notificaciones por consola y Telegram de forma
    no bloqueante (hilo separado para Telegram).
    """

    def __init__(self) -> None:
        self._telegram_enabled = settings.TELEGRAM_ENABLED
        self._bot_token = settings.TELEGRAM_BOT_TOKEN
        self._chat_id = settings.TELEGRAM_CHAT_ID
        self._timeout = settings.TELEGRAM_TIMEOUT

        if self._telegram_enabled:
            logger.info("✅ Telegram activado")
        else:
            logger.info("ℹ️  Telegram desactivado (configura BOT_TOKEN y CHAT_ID en .env)")

    # ─── Señal generada ─────────────────────────────────────────
    def signal(
        self,
        symbol: str,
        direction: str,
        score: int,
        entry: float,
        sl: float,
        tp1: float,
        tp2: float,
        reason: str,
    ) -> None:
        emoji = direction_emoji(direction)
        msg = (
            f"{emoji} SEÑAL {direction} | {symbol}\n"
            f"   Score:  {score}/100\n"
            f"   Entrada: {fmt_price(entry, 2)}\n"
            f"   SL:      {fmt_price(sl, 2)}\n"
            f"   TP1:     {fmt_price(tp1, 2)}\n"
            f"   TP2:     {fmt_price(tp2, 2)}\n"
            f"   Razón:  {reason}"
        )
        logger.info(msg)
        self._beep(1)
        self._telegram_async(f"⚡ <b>MARK I — {symbol}</b>\n{msg}")

    # ─── Orden ejecutada ────────────────────────────────────────
    def order_executed(
        self,
        symbol: str,
        direction: str,
        lot: float,
        entry: float,
        sl: float,
        tp1: float,
        ticket: int,
    ) -> None:
        emoji = direction_emoji(direction)
        msg = (
            f"{emoji} ORDEN EJECUTADA | {symbol}\n"
            f"   Dirección: {direction} @ {fmt_price(entry, 2)}\n"
            f"   Lote: {lot}  |  SL: {fmt_price(sl, 2)}  |  TP1: {fmt_price(tp1, 2)}\n"
            f"   Ticket: {ticket}"
        )
        logger.info(msg)
        self._beep(2)
        self._telegram_async(f"✅ <b>MARK I — Orden abierta</b>\n{msg}")

    # ─── Trade cerrado ──────────────────────────────────────────
    def trade_closed(
        self,
        symbol: str,
        direction: str,
        pnl: float,
        reason: str,
        ticket: int,
    ) -> None:
        icon = "🟢" if pnl >= 0 else "🔴"
        msg = (
            f"{icon} TRADE CERRADO | {symbol} ({direction})\n"
            f"   P&L: {fmt_pnl(pnl)}\n"
            f"   Razón: {reason}  |  Ticket: {ticket}"
        )
        logger.info(msg)
        self._beep(1 if pnl >= 0 else 3)
        self._telegram_async(f"{icon} <b>MARK I — Trade cerrado</b>\n{msg}")

    # ─── Alerta de riesgo ────────────────────────────────────────
    def risk_alert(self, message: str) -> None:
        msg = f"⚠️  RIESGO | {message}"
        logger.warning(msg)
        self._beep(5)
        self._telegram_async(f"⚠️ <b>MARK I — Alerta de riesgo</b>\n{message}")

    # ─── Error crítico ───────────────────────────────────────────
    def critical_error(self, message: str) -> None:
        msg = f"🚨 ERROR CRÍTICO | {message}"
        logger.critical(msg)
        self._beep(10)
        self._telegram_async(f"🚨 <b>MARK I — Error crítico</b>\n{message}")

    # ─── Info general ────────────────────────────────────────────
    def info(self, message: str) -> None:
        logger.info(message)
        self._telegram_async(f"ℹ️ <b>MARK I</b>\n{message}")

    # ─── Beep de Windows ────────────────────────────────────────
    @staticmethod
    def _beep(count: int = 1) -> None:
        try:
            import winsound
            for _ in range(count):
                winsound.Beep(880, 150)
                time.sleep(0.05)
        except Exception:
            pass  # No disponible fuera de Windows o sin hardware

    # ─── Telegram async ─────────────────────────────────────────
    def _telegram_async(self, text: str) -> None:
        if not self._telegram_enabled:
            return
        thread = threading.Thread(
            target=self._send_telegram,
            args=(text,),
            daemon=True,
        )
        thread.start()

    def _send_telegram(self, text: str) -> None:
        url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_notification": False,
        }
        for attempt in range(3):
            try:
                resp = requests.post(url, json=payload, timeout=self._timeout)
                if resp.status_code == 200:
                    return
                logger.warning(f"Telegram error {resp.status_code}: {resp.text[:200]}")
            except requests.RequestException as exc:
                logger.warning(f"Telegram intento {attempt+1} fallido: {exc}")
                time.sleep(2 ** attempt)
