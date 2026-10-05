"""
execution_engine/session_manager.py — Validates operational sessions for the LIVE bot.
"""
from datetime import datetime, timezone
import pytz

class SessionManager:
    @staticmethod
    def is_in_session() -> tuple[bool, str]:
        """
        Returns (True, session_name) if currently within an allowed trading window.
        Returns (False, "OUT_OF_SESSION") otherwise.
        
        Optimized Windows (Colombia Time / COT = UTC-5):
        - MORNING:   05:30 - 08:30 (European/London Open)
        - AFTERNOON: 11:00 - 14:00 (Post-US-Open trend continuation)
        """
        now_utc = datetime.now(timezone.utc)
        cot_tz = pytz.timezone("America/Bogota")
        now_cot = now_utc.astimezone(cot_tz)
        
        # Calculate time as a float (e.g. 5:30 = 5.5)
        h = now_cot.hour + now_cot.minute / 60.0
        
        if 5.5 <= h <= 8.5:
            return True, "MORNING"
        elif 11.0 <= h <= 14.0:
            return True, "AFTERNOON"
            
        return False, "OUT_OF_SESSION"
