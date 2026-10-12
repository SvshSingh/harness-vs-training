import json
import random
from datetime import date

import pytest

from hvt import data, harness, taskgen
from hvt.evaluate import summarize
from hvt.scoring import score
from hvt.targeting import weights_from_records
from hvt.tools import TODAY, check, extract_json, validate


# ---------- generator ----------

def test_today_is_monday():
    assert TODAY.weekday() == 0


def test_generation_is_deterministic_and_unique():
    a, b = taskgen.generate(300, seed=7), taskgen.generate(300, seed=7)
    assert [e.request for e in a] == [e.request for e in b]
    assert len({e.request for e in a}) == 300


def test_every_gold_call_passes_the_schema():
    for ex in taskgen.generate(4000, seed=3):
        assert validate({"name": ex.tool, "arguments": ex.args}) == [], ex


@pytest.mark.parametrize("kind", taskgen.KIND_OPTIONS["date"])
def test_dates_are_in_the_future(kind):
    rng = random.Random(0)
    for _ in range(200):
        _, gold = taskgen._date(rng, kind)
        assert date.fromisoformat(gold) > TODAY


def test_weekday_means_strictly_after_today():
    rng = random.Random(0)
    seen = {}
    for _ in range(300):
        text, gold = taskgen._date(rng, "weekday")
        seen[text] = gold
    assert seen["this coming Monday"] == "2026-03-09"
    assert seen["this coming Tuesday"] == "2026-03-03"


def test_spoken_times_match_gold():
    rng = random.Random(1)
    pairs = dict(taskgen._time(rng, "words") for _ in range(2000))
    assert pairs["quarter to 12 in the morning"] == "11:45"
    assert pairs["quarter to 9 in the evening"] == "20:45"
    assert pairs["half past 3 in the afternoon"] == "15:30"
    assert pairs["noon"] == "12:00"


def test_kind_weights_steer_sampling():
    w = {"date:weekday": 1.0, "date:iso": 0.0, "date:month_day": 0.0, "date:relative": 0.0}
    exs = taskgen.generate(200, seed=1, tool_weights={"get_weather": 1, "search_flights": 0,
                                                      "convert_currency": 0, "schedule_meeting": 0},
                           kind_weights=w)
    assert {e.tool for e in exs} == {"get_weather"}
    assert {e.kinds["date"] for e in exs} == {"weekday"}


def test_generate_rejects_bad_input():
    with pytest.raises(ValueError):
        taskgen.generate(-1, seed=0)
    with pytest.raises(ValueError):
        taskgen.generate(5, seed=0, tool_weights={t: 0 for t in taskgen.ARG_TYPES})


def test_training_data_never_overlaps_eval_or_probe():
    rows = data.train_set(500, seed=11)
    assert not {r["request"] for r in rows} & data.reserved()
    assert not {e.request for e in data.eval_set()} & {e.request for e in data.probe_set()}


# ---------- parsing, validation, scoring ----------

def test_extract_json_handles_fences_and_prose():
    obj, err = extract_json('Sure!\n```json\n{"name": "x", "arguments": {}}\n```')
    assert err is None and obj["name"] == "x"
    assert extract_json("no braces here")[0] is None
    assert extract_json('{"name": "x", ')[0] is None
    assert extract_json('{bad} then {"ok": 1}')[0] == {"ok": 1}


def test_validate_gives_actionable_errors():
    errs = validate({"name": "schedule_meeting",
                     "arguments": {"date": "March 6", "start_time": "3pm", "extra": 1}})
    joined = " ".join(errs)
    assert "Missing required argument 'duration_minutes'" in joined
    assert "Unexpected argument 'extra'" in joined
    assert "YYYY-MM-DD" in joined and "HH:MM" in joined
    assert validate({"name": "nope", "arguments": {}})[0].startswith("Unknown tool")
    assert validate([1, 2]) and validate({"name": "x"})


def test_validate_types_and_ranges():
    base = {"origin": "BOM", "destination": "DEL", "date": "2026-03-05"}
    assert validate({"name": "search_flights", "arguments": {**base, "passengers": 2}}) == []
    assert validate({"name": "search_flights", "arguments": {**base, "passengers": 2.0}}) == []
    assert validate({"name": "search_flights", "arguments": {**base, "passengers": True}})
    assert validate({"name": "search_flights", "arguments": {**base, "passengers": 0}})
    assert validate({"name": "search_flights", "arguments": {**base, "passengers": "2"}})
    assert validate({"name": "get_weather", "arguments": {"city": "Paris", "date": "2026-02-30", "unit": "celsius"}})
    assert validate({"name": "convert_currency", "arguments": {"amount": 0, "from_currency": "USD", "to_currency": "INR"}})


def test_check_flags_non_json():
    assert "not a valid JSON object" in check("I think you want the weather.")[0]


GOLD_ARGS = {"date": "2026-03-06", "start_time": "15:00", "duration_minutes": 90}


@pytest.mark.parametrize("text,failure,wrong", [
    (json.dumps({"name": "schedule_meeting", "arguments": GOLD_ARGS}), None, []),
    (json.dumps({"name": "schedule_meeting", "arguments": {**GOLD_ARGS, "duration_minutes": 90.0}}), None, []),
    ("nothing", "no_json", list(GOLD_ARGS)),
    ('{"tool": "schedule_meeting"}', "bad_shape", list(GOLD_ARGS)),
    (json.dumps({"name": "get_weather", "arguments": {}}), "wrong_tool", list(GOLD_ARGS)),
    (json.dumps({"name": "schedule_meeting", "arguments": {**GOLD_ARGS, "start_time": "3pm"}}), "schema", ["start_time"]),
    (json.dumps({"name": "schedule_meeting", "arguments": {**GOLD_ARGS, "start_time": "03:00"}}), "wrong_value", ["start_time"]),
])
def test_score_categories(text, failure, wrong):
    s = score(text, "schedule_meeting", GOLD_ARGS)
    assert (s.correct, s.failure, s.wrong_args) == (failure is None, failure, wrong)


def test_city_match_ignores_case():
    gold = {"city": "New York", "date": "2026-03-03", "unit": "celsius"}
    text = json.dumps({"name": "get_weather", "arguments": {**gold, "city": "new york"}})
    assert score(text, "get_weather", gold).correct


# ---------- harness ----------

class ScriptedModel:
    """Replies from a fixed script per request; records every conversation it is shown."""

    def __init__(self, script):
        self.script, self.calls, self.seen = script, {}, []

    def generate(self, conversations):
        out = []
        for conv in conversations:
            self.seen.append(conv)
            req = conv[1]["content"]
            k = self.calls.get(req, 0)
            self.calls[req] = k + 1
            replies = self.script[req]
            out.append((replies[min(k, len(replies) - 1)], 100, 20))
        return out


def _example():
    return taskgen.Example("Schedule it", "schedule_meeting", GOLD_ARGS, {
        "date": "iso", "start_time": "ampm", "duration_minutes": "words"})


def test_harness_repairs_schema_errors_and_counts_tokens():
    ex = _example()
    bad = json.dumps({"name": "schedule_meeting", "arguments": {**GOLD_ARGS, "start_time": "3pm"}})
    good = json.dumps({"name": "schedule_meeting", "arguments": GOLD_ARGS})
    plain = harness.run(ScriptedModel({ex.request: [bad, good]}), [ex], "plain")[0]
    assert not plain["correct"] and plain["attempts"] == 1

    m = ScriptedModel({ex.request: [bad, good]})
    rec = harness.run(m, [ex], "harness", max_retries=1)[0]
    assert rec["correct"] and not rec["first_correct"] and rec["attempts"] == 2
    assert (rec["prompt_tokens"], rec["completion_tokens"]) == (200, 40)
    retry_msg = m.seen[1][-1]["content"]
    assert "HH:MM" in retry_msg and "15:00" not in retry_msg  # actionable, but no answer leak


def test_harness_cannot_fix_valid_but_wrong_values():
    ex = _example()
    wrong = json.dumps({"name": "schedule_meeting", "arguments": {**GOLD_ARGS, "start_time": "03:00"}})
    rec = harness.run(ScriptedModel({ex.request: [wrong]}), [ex], "harness")[0]
    assert rec["attempts"] == 1 and rec["failure"] == "wrong_value"


def test_harness_respects_retry_budget():
    ex = _example()
    rec = harness.run(ScriptedModel({ex.request: ["junk"]}), [ex], "harness", max_retries=3)[0]
    assert rec["attempts"] == 4 and rec["failure"] == "no_json"
    with pytest.raises(ValueError):
        harness.run(ScriptedModel({}), [ex], "other")


# ---------- targeting and summary ----------

def test_targeting_upweights_failed_kinds():
    recs = [
        {"tool": "schedule_meeting", "kinds": {"date": "iso", "start_time": "words", "duration_minutes": "words"},
         "correct": False, "failure": "wrong_value", "wrong_args": ["start_time"]},
        {"tool": "schedule_meeting", "kinds": {"date": "iso", "start_time": "24h", "duration_minutes": "words"},
         "correct": True, "failure": None, "wrong_args": []},
        {"tool": "get_weather", "kinds": {"city": "name", "date": "weekday", "unit": "explicit"},
         "correct": False, "failure": "no_json", "wrong_args": ["city", "date", "unit"]},
    ]
    w = weights_from_records(recs, floor=0.05)
    k = w["kind_weights"]
    assert k["time:words"] == pytest.approx(1.05) and k["time:24h"] == pytest.approx(0.05)
    assert k["date:weekday"] == pytest.approx(0.05)   # whole-call failure is charged to the tool
    assert k["amount:words"] == 1.0                    # never probed
    assert w["tool_weights"]["get_weather"] == pytest.approx(1.05)
    with pytest.raises(ValueError):
        weights_from_records(recs, floor=-1)


def test_summarize():
    ex = _example()
    good = json.dumps({"name": "schedule_meeting", "arguments": GOLD_ARGS})
    s = summarize(harness.run(ScriptedModel({ex.request: [good]}), [ex], "plain"))
    assert s["accuracy"] == 1.0 and s["failures"] == {} and s["arg_accuracy_by_kind"]["time:ampm"]["acc"] == 1.0
    with pytest.raises(ValueError):
        summarize([])


def test_summarize_counts_repaired_and_masked():
    ex = _example()
    good = json.dumps({"name": "schedule_meeting", "arguments": GOLD_ARGS})
    schema_bad = json.dumps({"name": "schedule_meeting", "arguments": {**GOLD_ARGS, "start_time": "3pm"}})
    valid_wrong = json.dumps({"name": "schedule_meeting", "arguments": {**GOLD_ARGS, "start_time": "03:00"}})
    ex2 = taskgen.Example("Other", ex.tool, ex.args, ex.kinds)
    recs = harness.run(ScriptedModel({ex.request: [schema_bad, good], ex2.request: [schema_bad, valid_wrong]}),
                       [ex, ex2], "harness")
    s = summarize(recs)
    assert (s["repaired"], s["masked"], s["accuracy"]) == (1, 1, 0.5)
