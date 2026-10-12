"""Seeded generator of tool-calling requests with exact gold answers.

Every argument is written in one of several phrasing "kinds" (a date as ISO, as "March 5",
as "in 3 days", as "this coming Friday"). Kinds are the unit we measure errors on and the
knob the targeted data arm turns.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import timedelta

from hvt.tools import AIRPORTS, TODAY

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
                8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}
CURRENCY_NAMES = {"USD": ["dollars", "US dollars", "bucks"], "INR": ["rupees", "Indian rupees"],
                  "EUR": ["euros"], "GBP": ["pounds", "British pounds"], "JPY": ["yen", "Japanese yen"]}

KIND_OPTIONS: dict[str, list[str]] = {
    "city": ["name"],
    "date": ["iso", "month_day", "relative", "weekday"],
    "unit": ["explicit", "symbol", "unspecified"],
    "airport": ["city", "code"],
    "passengers": ["digits", "words", "implied"],
    "amount": ["digits", "commas", "k_suffix", "words"],
    "currency": ["code", "name"],
    "time": ["24h", "ampm", "words"],
    "duration": ["minutes", "hours", "words"],
}
ARG_TYPES: dict[str, dict[str, str]] = {
    "get_weather": {"city": "city", "date": "date", "unit": "unit"},
    "search_flights": {"origin": "airport", "destination": "airport", "date": "date", "passengers": "passengers"},
    "convert_currency": {"amount": "amount", "from_currency": "currency", "to_currency": "currency"},
    "schedule_meeting": {"date": "date", "start_time": "time", "duration_minutes": "duration"},
}
TEMPLATES = {
    "get_weather": [
        "What's the weather going to be in {city} {date}{unit}?",
        "{city} forecast {date}{unit}, please.",
        "Check the weather in {city} {date}{unit}.",
    ],
    "search_flights": [
        "Find flights from {origin} to {destination} {date} {passengers}.",
        "I need to fly {origin} to {destination} {date}, {passengers}.",
        "Search flights {origin} to {destination} {date} {passengers}.",
    ],
    "convert_currency": [
        "Convert {amount} {from_currency} to {to_currency}.",
        "How much is {amount} {from_currency} in {to_currency}?",
        "I have {amount} {from_currency}, what's that in {to_currency}?",
    ],
    "schedule_meeting": [
        "Schedule a meeting {date} at {start_time} for {duration_minutes}.",
        "Book {duration_minutes} with the team {date} at {start_time}.",
        "Put a meeting on my calendar {date} at {start_time}, it should last {duration_minutes}.",
    ],
}


@dataclass(frozen=True)
class Example:
    request: str
    tool: str
    args: dict
    kinds: dict  # arg name -> phrasing kind

    def to_dict(self) -> dict:
        return {"request": self.request, "tool": self.tool, "args": self.args, "kinds": self.kinds}


# Each phraser takes (rng, kind) and returns (surface text, gold value).

def _date(rng: random.Random, kind: str):
    if kind == "relative":
        k = rng.randint(1, 10)
        text = {1: "tomorrow", 2: "the day after tomorrow"}.get(k, f"in {k} days")
        return text, (TODAY + timedelta(days=k)).isoformat()
    if kind == "weekday":
        wd = rng.randrange(7)
        offset = (wd - TODAY.weekday()) % 7 or 7  # strictly after today
        return f"this coming {WEEKDAYS[wd]}", (TODAY + timedelta(days=offset)).isoformat()
    d = TODAY + timedelta(days=rng.randint(1, 60))
    if kind == "iso":
        return f"on {d.isoformat()}", d.isoformat()
    month = MONTHS[d.month - 1]
    return rng.choice([f"on {month} {d.day}", f"on {d.day} {month}"]), d.isoformat()


def _period(hour24: int) -> str:
    return "morning" if hour24 < 12 else "afternoon" if hour24 < 17 else "evening"


def _h12(hour24: int) -> int:
    return hour24 % 12 or 12


def _time(rng: random.Random, kind: str):
    h, m = rng.randint(7, 20), rng.choice([0, 15, 30, 45])
    gold = f"{h:02d}:{m:02d}"
    if kind == "24h":
        return gold, gold
    if kind == "ampm":
        suffix = "am" if h < 12 else "pm"
        return (f"{_h12(h)}{suffix}" if m == 0 else f"{_h12(h)}:{m:02d} {suffix}"), gold
    if h == 12 and m == 0:
        return "noon", gold
    spoken = h + 1 if m == 45 else h
    phrase = {0: f"{_h12(h)} o'clock", 15: f"quarter past {_h12(h)}",
              30: f"half past {_h12(h)}", 45: f"quarter to {_h12(spoken)}"}[m]
    return f"{phrase} in the {_period(h)}", gold  # 11:45 is "quarter to 12 in the morning"


def _duration(rng: random.Random, kind: str):
    if kind == "minutes":
        v = rng.choice([15, 20, 30, 45, 60, 90, 120])
        return f"{v} minutes", v
    if kind == "hours":
        v = rng.choice([30, 60, 90, 120, 150, 180])
        hours = v / 60
        return (f"{hours:g} hour" if hours == 1 else f"{hours:g} hours"), v
    words = {15: "a quarter of an hour", 30: "half an hour", 45: "forty-five minutes", 60: "an hour",
             75: "an hour and fifteen minutes", 90: "an hour and a half", 120: "two hours"}
    v = rng.choice(list(words))
    return words[v], v


def _amount(rng: random.Random, kind: str):
    if kind == "digits":
        v = rng.choice([rng.randint(5, 999), round(rng.uniform(5, 999), 2)])
        return f"{v:g}" if isinstance(v, float) else str(v), v
    if kind == "commas":
        v = rng.randint(1_000, 250_000)
        return f"{v:,}", v
    if kind == "k_suffix":
        k = rng.choice([1, 1.5, 2, 2.5, 5, 7.5, 10, 12, 20, 50])
        return f"{k:g}k", int(k * 1000)
    words = {50: "fifty", 200: "two hundred", 1000: "a thousand", 1500: "fifteen hundred",
             2000: "two thousand", 25: "twenty-five", 500: "five hundred"}
    v = rng.choice(list(words))
    return words[v], v


def _passengers(rng: random.Random, kind: str):
    if kind == "implied":
        return rng.choice([("just for me", 1), ("for me and my wife", 2), ("for me and my two kids", 3),
                           ("for the four of us", 4), ("for my parents and me", 3)])
    v = rng.randint(1, 9)
    noun = "passenger" if v == 1 else "passengers"
    return (f"for {v} {noun}" if kind == "digits" else f"for {NUMBER_WORDS[v]} {noun}"), v


PHRASERS = {"date": _date, "time": _time, "duration": _duration, "amount": _amount, "passengers": _passengers}


def _pick(rng: random.Random, arg_type: str, weights: dict | None) -> str:
    options = KIND_OPTIONS[arg_type]
    if not weights:
        return rng.choice(options)
    w = [weights.get(f"{arg_type}:{k}", 1.0) for k in options]
    return rng.choice(options) if sum(w) <= 0 else rng.choices(options, weights=w)[0]


def sample(rng: random.Random, tool: str, kind_weights: dict | None = None) -> Example:
    kinds = {a: _pick(rng, t, kind_weights) for a, t in ARG_TYPES[tool].items()}
    slots, args = {}, {}
    if tool == "get_weather":
        city = rng.choice(list(AIRPORTS))
        unit = rng.choice(["celsius", "fahrenheit"]) if kinds["unit"] != "unspecified" else "celsius"
        slots["city"], args["city"] = city, city
        slots["unit"] = {"explicit": f" in {unit}", "symbol": f" in °{unit[0].upper()}",
                         "unspecified": ""}[kinds["unit"]]
        args["unit"] = unit
    elif tool == "search_flights":
        origin, dest = rng.sample(list(AIRPORTS), 2)
        for arg, city in (("origin", origin), ("destination", dest)):
            slots[arg] = city if kinds[arg] == "city" else AIRPORTS[city]
            args[arg] = AIRPORTS[city]
    elif tool == "convert_currency":
        src, dst = rng.sample(list(CURRENCY_NAMES), 2)
        for arg, code in (("from_currency", src), ("to_currency", dst)):
            slots[arg] = code if kinds[arg] == "code" else rng.choice(CURRENCY_NAMES[code])
            args[arg] = code
    for arg, arg_type in ARG_TYPES[tool].items():
        if arg_type in PHRASERS:
            slots[arg], args[arg] = PHRASERS[arg_type](rng, kinds[arg])
    text = rng.choice(TEMPLATES[tool]).format(**slots)
    return Example(request=text[0].upper() + text[1:], tool=tool, args=args, kinds=kinds)


def generate(n: int, seed: int, tool_weights: dict | None = None, kind_weights: dict | None = None,
             exclude: frozenset[str] | set[str] = frozenset()) -> list[Example]:
    """n examples with unique requests, none of which appear in `exclude`."""
    if n < 0:
        raise ValueError("n must be non-negative")
    rng = random.Random(seed)
    tools = list(ARG_TYPES)
    tw = [max(0.0, (tool_weights or {}).get(t, 1.0)) for t in tools]
    if sum(tw) <= 0:
        raise ValueError("tool_weights must give some tool positive weight")
    seen, out = set(exclude), []
    for _ in range(n * 50):
        if len(out) == n:
            break
        ex = sample(rng, rng.choices(tools, weights=tw)[0], kind_weights)
        if ex.request not in seen:
            seen.add(ex.request)
            out.append(ex)
    if len(out) < n:
        raise RuntimeError(f"could only generate {len(out)} unique examples out of {n}")
    return out


if __name__ == "__main__":
    for ex in generate(8, seed=0):
        print(f"{ex.request}\n  -> {ex.tool} {ex.args}  {ex.kinds}")
