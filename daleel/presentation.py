"""Display formatting without changing stored UTC times or video offsets."""
from datetime import timedelta, timezone


SAUDI_TIME = timezone(timedelta(hours=3), "AST")


def saudi_datetime(value):
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(SAUDI_TIME)


def format_mission_clock(value):
    """Twelve-hour AST wall clock; video timestamps must not use this formatter."""
    return saudi_datetime(value).strftime("%I:%M %p").lstrip("0")


def target_label(value):
    """Formal display names only; stored COCO classes remain person/car."""
    return {"person": "Person", "car": "Vehicle"}.get(value, str(value).title())
