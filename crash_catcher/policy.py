"""Alert policies: given how long the eyes have been off the road and the
motion state, should we alert right now?

FixedPolicy is the baseline (2.0 s everywhere). ContextPolicy is Novelty 1.
Both run on every frame so they can be compared on identical data.
"""


class FixedPolicy:
    name = "fixed"

    def __init__(self, limit_s: float):
        self.limit_s = limit_s

    def should_alert(self, off_road: bool, off_road_duration: float, motion_state: str) -> bool:
        return off_road and off_road_duration >= self.limit_s


class ContextPolicy:
    name = "context"

    def __init__(self, limits_s: dict):
        self.limits_s = limits_s
        self._prev_state = None
        self._pulled_away_distracted = False

    def should_alert(self, off_road: bool, off_road_duration: float, motion_state: str) -> bool:
        was_stopped = self._prev_state == "stopped"
        self._prev_state = motion_state

        if not off_road:
            self._pulled_away_distracted = False
            return False

        # Pulling away while still looking away: alert at once, and keep
        # alerting until the eyes come back.
        if was_stopped and motion_state != "stopped":
            self._pulled_away_distracted = True
        if self._pulled_away_distracted:
            return True

        limit = self.limits_s.get(motion_state)
        if limit is None:  # stopped: a glance at a red light is fine
            return False
        return off_road_duration >= limit
