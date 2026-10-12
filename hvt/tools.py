"""Tool schemas, the system prompt, and a schema validator that never sees the gold answer."""
from __future__ import annotations

import json
import re
from datetime import date

TODAY = date(2026, 3, 2)  # a Monday; stated in the system prompt so relative dates have one answer

AIRPORTS = {
    "Mumbai": "BOM", "Delhi": "DEL", "Bengaluru": "BLR", "London": "LHR", "Paris": "CDG",
    "Tokyo": "HND", "New York": "JFK", "Singapore": "SIN", "Dubai": "DXB", "San Francisco": "SFO",
}
CURRENCY_CODES = ["USD", "INR", "EUR", "GBP", "JPY"]

TOOLS: dict[str, dict] = {
    "get_weather": {
        "description": "Weather forecast for a city on a date.",
        "params": {
            "city": {"type": "string", "description": "City name, e.g. Paris"},
            "date": {"type": "string", "format": "date"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"],
                     "description": "Use celsius if the user does not say"},
        },
    },
    "search_flights": {
        "description": "Search flights between two airports.",
        "params": {
            "origin": {"type": "string", "enum": sorted(AIRPORTS.values()), "description": "Airport code"},
            "destination": {"type": "string", "enum": sorted(AIRPORTS.values()), "description": "Airport code"},
            "date": {"type": "string", "format": "date"},
            "passengers": {"type": "integer", "min": 1, "max": 9},
        },
    },
    "convert_currency": {
        "description": "Convert an amount between currencies.",
        "params": {
            "amount": {"type": "number", "min": 0, "exclusive_min": True},
            "from_currency": {"type": "string", "enum": CURRENCY_CODES},
            "to_currency": {"type": "string", "enum": CURRENCY_CODES},
        },
    },
    "schedule_meeting": {
        "description": "Put a meeting on the calendar.",
        "params": {
            "date": {"type": "string", "format": "date"},
            "start_time": {"type": "string", "format": "time"},
            "duration_minutes": {"type": "integer", "min": 1, "max": 600},
        },
    },
}

_FORMAT_HINTS = {"date": "YYYY-MM-DD", "time": "HH:MM, 24-hour clock"}


def system_prompt() -> str:
    lines = [f"Today is {TODAY:%A}, {TODAY.isoformat()}.", "You can call these tools:"]
    for name, spec in TOOLS.items():
        args = []
        for arg, p in spec["params"].items():
            detail = p["type"]
            if "format" in p:
                detail += f", {_FORMAT_HINTS[p['format']]}"
            if "enum" in p:
                detail += ", one of " + "|".join(p["enum"])
            if "description" in p:
                detail += f"; {p['description']}"
            args.append(f"{arg} ({detail})")
        lines.append(f"- {name}: {spec['description']} Arguments: " + "; ".join(args))
    lines.append("Airport codes: " + ", ".join(f"{c}={a}" for c, a in AIRPORTS.items()) + ".")
    lines.append('Reply with only one JSON object: {"name": <tool>, "arguments": {...}}. All arguments are required.')
    return "\n".join(lines)


def extract_json(text: str) -> tuple[dict | None, str | None]:
    """Return the first JSON object in text, tolerating code fences and surrounding prose."""
    decoder = json.JSONDecoder()
    starts = [m.start() for m in re.finditer(r"\{", text)]
    if not starts:
        return None, "no JSON object found"
    first_error = None
    for i in starts:
        try:
            obj, _ = decoder.raw_decode(text, i)
        except json.JSONDecodeError as e:
            first_error = first_error or e.msg
            continue
        if isinstance(obj, dict):
            return obj, None
    return None, f"invalid JSON ({first_error or 'not an object'})"


def _valid_date(s: str) -> bool:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return False
    try:
        date.fromisoformat(s)
    except ValueError:
        return False
    return True


def _valid_time(s: str) -> bool:
    m = re.fullmatch(r"(\d{2}):(\d{2})", s)
    return bool(m) and int(m[1]) < 24 and int(m[2]) < 60


def _check_value(arg: str, value, p: dict) -> str | None:
    t = p["type"]
    is_num = isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "string" and not isinstance(value, str):
        return f"'{arg}' must be a string, got {json.dumps(value)}."
    if t == "integer" and not (is_num and float(value).is_integer()):
        return f"'{arg}' must be an integer, got {json.dumps(value)}."
    if t == "number" and not is_num:
        return f"'{arg}' must be a number, got {json.dumps(value)}."
    fmt = p.get("format")
    if fmt == "date" and not _valid_date(value):
        return f"'{arg}' must be a date in YYYY-MM-DD format, got {json.dumps(value)}."
    if fmt == "time" and not _valid_time(value):
        return f"'{arg}' must be a time in HH:MM 24-hour format, got {json.dumps(value)}."
    if "enum" in p and value not in p["enum"]:
        return f"'{arg}' must be one of {', '.join(p['enum'])}, got {json.dumps(value)}."
    if "min" in p and (value < p["min"] or (p.get("exclusive_min") and value == p["min"])):
        return f"'{arg}' must be {'greater than' if p.get('exclusive_min') else 'at least'} {p['min']}, got {value}."
    if "max" in p and value > p["max"]:
        return f"'{arg}' must be at most {p['max']}, got {value}."
    return None


def validate(call) -> list[str]:
    """Schema errors for a parsed call, phrased so a model can act on them. Empty list means valid."""
    if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not isinstance(call.get("arguments"), dict):
        return ['The reply must be {"name": <tool>, "arguments": {...}} with "arguments" as an object.']
    name, args = call["name"], call["arguments"]
    if name not in TOOLS:
        return [f"Unknown tool '{name}'. Available tools: {', '.join(TOOLS)}."]
    params = TOOLS[name]["params"]
    errors = [f"Missing required argument '{a}' for {name}." for a in params if a not in args]
    errors += [f"Unexpected argument '{a}' for {name}." for a in args if a not in params]
    errors += [e for a, v in args.items() if a in params and (e := _check_value(a, v, params[a]))]
    return errors


def check(text: str) -> list[str]:
    """Everything a harness can know about a raw reply without the answer key."""
    call, err = extract_json(text)
    if call is None:
        return [f"Your reply was not a valid JSON object: {err}."]
    return validate(call)


if __name__ == "__main__":
    print(system_prompt())
    print(check('```json\n{"name": "schedule_meeting", "arguments": {"date": "2026-03-06", "start_time": "3pm"}}\n```'))
