import json
import sqlite3
from pathlib import Path

import pytest

import data_saver
from data_saver.core import MB, DataSaver, fmt_bytes
from data_saver.forecast import Forecast, Item
from data_saver.ledger import Ledger
from data_saver.meter import Meter


class Counters:
    def __init__(self):
        self.rx = self.tx = 0

    def __call__(self):
        return self.rx, self.tx


def make(config=None, forecasts=None, counters=None, now=1_800_000_000.0):
    settings = dict(config or {})
    clock = {"now": now}
    ledger = Ledger(lambda: sqlite3.connect(":memory:", check_same_thread=False), clock=lambda: clock["now"])
    calls = []

    def forecaster(command):
        calls.append(command)
        return (forecasts or {}).get(command, Forecast())

    saver = DataSaver(ledger, Meter(counters or Counters()), lambda key, default=None: settings.get(key, default),
                      settings.__setitem__, forecaster, clock=lambda: clock["now"])
    saver.test_settings, saver.test_clock, saver.test_forecast_calls = settings, clock, calls
    return saver


def sized(megabytes, **kw):
    return Forecast(items=[Item("pip", "torch", int(megabytes * MB), **kw)], overhead_bytes=1500)


def gate(saver, command):
    return saver.pre_tool_call(tool_name="terminal", args={"command": command}, session_id="s1")


# ── forecast gate ─────────────────────────────────────────────────────────

def test_small_download_passes_silently_and_charges_the_lookup():
    saver = make(forecasts={"pip install torch": sized(3)})
    assert gate(saver, "pip install torch") is None
    assert saver.ledger.by_category()["preflight"] == 1500


def test_non_download_commands_are_never_forecast():
    saver = make()
    assert gate(saver, "ls -la && git status") is None
    assert saver.test_forecast_calls == []


def test_large_download_goes_to_approval_with_sizes_in_the_message():
    saver = make(forecasts={"pip install torch": sized(800)})
    result = gate(saver, "pip install torch")
    assert result["action"] == "approve"
    assert "800 MB" in result["message"] and "500 MB" in result["message"]
    assert "Only" in result["message"]  # 800 MB does not fit the 500 MB budget


def test_over_remaining_budget_asks_even_when_below_the_ask_threshold():
    saver = make({"daily_budget_mb": 100}, forecasts={"pip install torch": sized(30)})
    saver.ledger.record("tool", "terminal", rx=90 * MB)
    assert gate(saver, "pip install torch")["action"] == "approve"


def test_enforce_mode_blocks_and_tells_the_agent_not_to_work_around_it():
    saver = make({"mode": "enforce", "daily_budget_mb": 100}, forecasts={"pip install torch": sized(300)})
    result = gate(saver, "pip install torch")
    assert result["action"] == "block"
    assert "/datasaver budget" in result["message"] and "Do not retry" in result["message"]


def test_enforce_mode_still_asks_for_large_downloads_that_fit():
    saver = make({"mode": "enforce", "daily_budget_mb": 5000}, forecasts={"pip install torch": sized(300)})
    assert gate(saver, "pip install torch")["action"] == "approve"


def test_monthly_budget_is_the_tightest():
    saver = make({"daily_budget_mb": 500, "monthly_budget_mb": 1000}, forecasts={"pip install torch": sized(40)})
    saver.ledger.record("tool", "terminal", rx=20 * MB)
    saver.test_clock["now"] += 86400  # next day, same month: daily resets, monthly does not
    saver.ledger.record("tool", "terminal", rx=950 * MB)
    saver.test_clock["now"] += 86400
    result = gate(saver, "pip install torch")
    assert result["action"] == "approve" and "this month" in result["message"]


def test_unknown_size_asks_only_once_the_budget_is_spent():
    unknown = Forecast(items=[Item("npm", "npm packages", None, "size cannot be looked up in advance")])
    saver = make({"daily_budget_mb": 10}, forecasts={"npm install": unknown})
    assert gate(saver, "npm install") is None
    saver.ledger.record("tool", "terminal", rx=11 * MB)
    result = gate(saver, "npm install")
    assert result["action"] == "approve" and "unknown size" in result["message"]


def test_lower_bound_forecasts_say_at_least():
    saver = make(forecasts={"pip install torch": sized(80, lower_bound=True)})
    assert "at least 80.0 MB" in gate(saver, "pip install torch")["message"]


def test_observe_mode_and_pause_never_gate():
    saver = make({"mode": "observe"}, forecasts={"pip install torch": sized(9000)})
    assert gate(saver, "pip install torch") is None
    saver = make(forecasts={"pip install torch": sized(9000)})
    assert "paused for 30" in saver.command("pause 30")
    assert gate(saver, "pip install torch") is None
    saver.test_clock["now"] += 31 * 60
    assert gate(saver, "pip install torch")["action"] == "approve"
    saver.command("pause 30")
    saver.command("resume")
    assert gate(saver, "pip install torch")["action"] == "approve"


def test_no_budget_set_still_asks_for_large_downloads():
    saver = make({"daily_budget_mb": 0}, forecasts={"pip install torch": sized(800)})
    result = gate(saver, "pip install torch")
    assert result["action"] == "approve" and "budget" not in result["message"].split(".")[1]


def test_network_tools_are_gated_only_when_the_budget_is_spent():
    saver = make({"daily_budget_mb": 10})
    assert saver.pre_tool_call(tool_name="web_extract", args={"urls": ["https://x.com"]}) is None
    saver.ledger.record("tool", "terminal", rx=12 * MB)
    asked = saver.pre_tool_call(tool_name="browser_exec", args={})
    assert asked["action"] == "approve" and asked["rule_key"].startswith("data-saver:over-budget:")
    assert saver.pre_tool_call(tool_name="read_file", args={"path": "x"}) is None
    saver.test_settings["mode"] = "enforce"
    assert saver.pre_tool_call(tool_name="web_search", args={"query": "x"})["action"] == "block"


def test_gate_fails_open_when_something_breaks():
    saver = make()
    saver._forecast = lambda command: 1 / 0
    assert gate(saver, "pip install torch") is None
    assert saver.pre_tool_call(tool_name="terminal", args="not a dict") is None


# ── lean mode ─────────────────────────────────────────────────────────────

def test_lean_mode_trims_web_results_when_budget_is_low():
    saver = make({"daily_budget_mb": 100})
    assert saver.tool_request(tool_name="web_extract", args={"urls": ["u"]}) is None
    saver.ledger.record("tool", "terminal", rx=80 * MB)
    extract = saver.tool_request(tool_name="web_extract", args={"urls": ["u"]})
    assert extract["args"] == {"urls": ["u"], "char_limit": 6000} and extract["source"] == "data-saver"
    assert saver.tool_request(tool_name="web_search", args={"query": "q", "limit": 10})["args"]["limit"] == 3
    # An explicit choice by the model, other tools and switched-off lean mode are left alone.
    assert saver.tool_request(tool_name="web_extract", args={"urls": ["u"], "char_limit": 20000}) is None
    assert saver.tool_request(tool_name="web_search", args={"query": "q", "limit": 2}) is None
    assert saver.tool_request(tool_name="terminal", args={"command": "ls"}) is None
    saver.test_settings["lean_below_pct"] = 0
    assert saver.tool_request(tool_name="web_extract", args={"urls": ["u"]}) is None


# ── metering ──────────────────────────────────────────────────────────────

def test_tool_execution_records_real_bytes_and_returns_the_result():
    counters = Counters()
    saver = make(counters=counters)

    def tool(args):
        counters.rx += 5 * MB
        counters.tx += 100_000
        return '{"ok": true}'

    out = saver.tool_execution(tool_name="terminal", args={"command": "curl x"}, next_call=tool, session_id="s1")
    assert out == '{"ok": true}'
    assert saver.ledger.total(session_id="s1") == 5 * MB + 100_000
    assert saver.ledger.biggest_calls()[0]["detail"] == "curl x"


def test_background_noise_below_the_floor_and_local_tools_are_not_recorded():
    counters = Counters()
    saver = make(counters=counters)

    def quiet(args):
        counters.rx += 500
        return "x"

    def busy(args):
        counters.rx += 50 * MB
        return "x"

    saver.tool_execution(tool_name="terminal", args={}, next_call=quiet)
    saver.tool_execution(tool_name="read_file", args={}, next_call=busy)
    assert saver.ledger.total() == 0


def test_tool_errors_propagate_and_are_still_metered():
    counters = Counters()
    saver = make(counters=counters)

    def failing(args):
        counters.rx += MB
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        saver.tool_execution(tool_name="terminal", args={}, next_call=failing)
    assert saver.ledger.total() == MB


def test_model_calls_are_metered_under_their_own_category():
    counters = Counters()
    saver = make(counters=counters)

    def call(request):
        counters.tx += 400_000
        counters.rx += 20_000
        return {"choices": []}

    saver.llm_execution(request={"messages": []}, next_call=call, model="hermes-4", provider="nous")
    assert saver.ledger.by_category()["model"] == 420_000
    assert saver.ledger.top_sources()[0]["source"] == "nous/hermes-4"


def test_metering_unavailable_still_runs_the_tool():
    saver = make(counters=lambda: None)
    assert saver.tool_execution(tool_name="terminal", args={}, next_call=lambda a: "ran") == "ran"
    assert "unavailable" in saver.status_text()


def test_overlapping_calls_split_traffic_instead_of_double_counting():
    counters = Counters()
    meter = Meter(counters)
    meter.begin("a")
    counters.rx += 100
    meter.begin("b")
    counters.rx += 200  # shared by a and b
    a = meter.end("a")
    counters.rx += 50
    b = meter.end("b")
    assert a == (200, 0) and b == (150, 0)
    assert a[0] + b[0] == 350
    counters.rx += 999  # nothing running: not attributed to anyone
    meter.begin("c")
    assert meter.end("c") == (0, 0)
    assert meter.end("never-started") is None


def test_counter_reset_does_not_produce_negative_usage():
    counters = Counters()
    counters.rx = 10_000
    meter = Meter(counters)
    meter.begin("a")
    counters.rx = 5  # interface restarted
    assert meter.end("a") == (0, 0)


# ── status, command, tools ────────────────────────────────────────────────

def test_status_text_and_tools():
    saver = make({"daily_budget_mb": 100}, forecasts={"pip install torch": sized(800)})
    saver.ledger.record("tool", "terminal", rx=30 * MB, detail="pip install x")
    saver.ledger.record("model", "nous/hermes-4", tx=10 * MB)
    text = saver.command("")
    assert "40.0 MB of 100 MB (40%)" in text and "terminal" in text and "model 10.0 MB" in text
    status = json.loads(saver.tool_data_usage({}))
    assert status["today_bytes"] == 40 * MB and status["budgets"][0]["left"] == 60 * MB
    forecast = json.loads(saver.tool_data_forecast({"command": "pip install torch"}))
    assert forecast["total_bytes"] == 800 * MB and forecast["fits_budget"] is False
    assert "error" in json.loads(saver.tool_data_forecast({}))


def test_command_changes_settings_and_rejects_bad_input():
    saver = make()
    assert "set to 300 MB" in saver.command("budget 300")
    assert saver.test_settings["daily_budget_mb"] == 300
    saver.command("mode enforce")
    assert saver.test_settings["mode"] == "enforce"
    assert saver.command("mode nonsense").startswith("Usage")
    assert saver.command("budget lots").startswith("Usage")
    assert saver.command("frobnicate").startswith("Usage")
    assert "nothing in this command" in saver.command("check ls -la")


def test_bad_config_values_fall_back_to_defaults():
    saver = make({"daily_budget_mb": "abc", "mode": "yolo", "ask_above_mb": None})
    assert saver.setting("daily_budget_mb") == 500
    assert saver.setting("mode") == "ask" and saver.setting("ask_above_mb") == 50


@pytest.mark.parametrize("count,text", [(0, "0 B"), (900, "900 B"), (2048, "2 KB"), (5 * MB, "5.0 MB"),
                                        (250 * MB, "250 MB"), (3 * 1024 * MB, "3.00 GB")])
def test_fmt_bytes(count, text):
    assert fmt_bytes(count) == text


# ── registration matches the manifest ─────────────────────────────────────

class FakeCtx:
    def __init__(self):
        self.tools, self.hooks, self.middleware, self.commands = [], [], [], []

    def get_config(self, key, default=None):
        return default

    def set_config(self, key, value):
        pass

    def register_tool(self, name, **kw):
        assert kw["schema"]["name"] == name
        self.tools.append(name)

    def register_hook(self, name, callback):
        self.hooks.append(name)

    def register_middleware(self, kind, callback):
        self.middleware.append(kind)

    def register_command(self, name, handler, **kw):
        self.commands.append(name)


def test_register_matches_plugin_yaml_and_touches_no_disk_or_network():
    ctx = FakeCtx()
    data_saver.register(ctx)  # would raise if it opened the database: Hermes is not importable here
    manifest = (Path(data_saver.__file__).parent / "plugin.yaml").read_text()

    def declared(key):
        block = manifest.split(f"{key}:\n", 1)[1]
        items = []
        for line in block.splitlines():
            if not line.startswith("  - "):
                break
            items.append(line[4:].strip())
        return items

    assert sorted(ctx.tools) == sorted(declared("provides_tools"))
    assert sorted(ctx.hooks) == sorted(declared("provides_hooks"))
    assert sorted(ctx.middleware) == sorted(declared("provides_middleware"))
    assert ctx.commands == ["datasaver"]


# ── background downloads ──────────────────────────────────────────────────

BG_RESULT = '{"status": "running", "session_id": "proc_abc123", "notify_on_complete": true}'


def start_background(saver, counters, command="pip install torch"):
    saver.pre_tool_call(tool_name="terminal", args={"command": command}, session_id="s1")
    return saver.tool_execution(tool_name="terminal", args={"command": command},
                                next_call=lambda a: BG_RESULT, session_id="s1")


def test_background_download_absorbs_traffic_instead_of_the_model():
    counters = Counters()
    saver = make({"daily_budget_mb": 0, "ask_above_mb": 100000}, counters=counters,
                 forecasts={"pip install torch": sized(200)})
    assert start_background(saver, counters) == BG_RESULT

    def model_call(request):
        counters.rx += 40 * MB  # the download, not the model
        return {}

    saver.llm_execution(request={}, next_call=model_call, model="m", provider="p")
    counters.rx += 60 * MB  # between calls: nothing else active, still the download
    saver.test_clock["now"] += 5
    status = saver.status()
    assert status["today_by_category"]["model"] == 0
    top = {row["source"]: row["bytes"] for row in status["top_today"]}
    assert top["terminal (background)"] == 100 * MB
    assert saver.ledger.biggest_calls()[0]["detail"] == "pip install torch"


def test_background_following_stops_when_quiet_and_model_is_metered_again():
    counters = Counters()
    saver = make({"daily_budget_mb": 0, "ask_above_mb": 100000}, counters=counters,
                 forecasts={"pip install torch": sized(200)})
    start_background(saver, counters)
    counters.rx += 10 * MB
    saver.test_clock["now"] += 30
    saver.status()                      # 10 MB moved: still following
    saver.test_clock["now"] += 30
    saver.status()                      # nothing moved for 30 s: stop
    assert saver._background == {}

    def model_call(request):
        counters.rx += MB
        return {}

    saver.llm_execution(request={}, next_call=model_call, model="m", provider="p")
    assert saver.ledger.by_category()["model"] == MB


def test_background_following_stops_at_one_and_a_half_times_the_forecast_and_on_timeout():
    counters = Counters()
    saver = make({"daily_budget_mb": 0, "ask_above_mb": 100000}, counters=counters,
                 forecasts={"pip install torch": sized(10)})
    start_background(saver, counters)
    counters.rx += 16 * MB
    saver.status()
    assert saver._background == {}
    start_background(saver, counters)
    counters.rx += MB
    saver.test_clock["now"] += 21 * 60
    saver.status()
    assert saver._background == {}


def test_background_servers_and_foreground_commands_are_not_followed():
    counters = Counters()
    saver = make(counters=counters)
    saver.tool_execution(tool_name="terminal", args={"command": "npm run dev", "background": True},
                         next_call=lambda a: BG_RESULT)
    saver.tool_execution(tool_name="terminal", args={"command": "pip install requests"},
                         next_call=lambda a: '{"output": "ok", "exit_code": 0}')
    assert saver._background == {}


def test_meter_take_keeps_the_call_running():
    counters = Counters()
    meter = Meter(counters)
    meter.begin("bg", absorb=True)
    meter.begin("call")
    counters.rx += 1000
    assert meter.take("bg") == (1000, 0)
    assert meter.end("call") == (0, 0)
    counters.rx += 500
    assert meter.end("bg") == (500, 0)
    assert meter.take("gone") is None
