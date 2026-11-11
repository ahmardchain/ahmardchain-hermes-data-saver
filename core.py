"""Budget policy, metering callbacks and the status report."""

from __future__ import annotations

import inspect
import json
import time
from typing import Any, Callable, Optional

import threading

from .forecast import Forecast, forecast_command, looks_like_download
from .ledger import Ledger, day_key, month_key
from .meter import Meter

MB = 1024 * 1024
MODES = ("ask", "enforce", "observe")
DEFAULTS = {"daily_budget_mb": 500, "monthly_budget_mb": 0, "ask_above_mb": 50, "mode": "ask", "lean_below_pct": 25}

# Tools that never touch the network: metering them would only record other apps' traffic.
LOCAL_TOOLS = frozenset({
    "read_file", "write_file", "patch", "search_files", "todo", "memory", "clarify", "skill_view",
    "skills_list", "skill_manage", "session_search", "data_usage", "data_forecast",
})
# Tools that always use the network; these are gated once the budget is spent.
NETWORK_TOOLS = frozenset({
    "web_search", "web_extract", "image_generate", "video_generate", "video_analyze", "vision_analyze",
    "text_to_speech",
})
NETWORK_PREFIXES = ("browser_", "web_")
NOISE_FLOOR = 2048  # bytes; smaller deltas are background chatter, not the tool call
LEAN_EXTRACT_CHARS = 6000
BACKGROUND_SOURCE = "terminal (background)"
BACKGROUND_MAX_SECONDS = 20 * 60  # stop following a background download after this long
BACKGROUND_IDLE_SECONDS = 20      # ...or once it has been quiet for this long
BACKGROUND_IDLE_BYTES = 64 * 1024
LEAN_SEARCH_RESULTS = 3


def fmt_bytes(count: float) -> str:
    count = float(max(0, count))
    if count < 1024:
        return f"{int(count)} B"
    if count < MB:
        return f"{count / 1024:.0f} KB"
    if count < 1024 * MB:
        value = count / MB
        return f"{value:.1f} MB" if value < 100 else f"{value:.0f} MB"
    return f"{count / (1024 * MB):.2f} GB"


def _as_int(value: Any, default: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def is_network_tool(tool_name: str) -> bool:
    return tool_name in NETWORK_TOOLS or tool_name.startswith(NETWORK_PREFIXES)


class DataSaver:
    def __init__(self, ledger: Ledger, meter: Meter, get_config: Callable[[str, Any], Any],
                 set_config: Optional[Callable[[str, Any], None]] = None,
                 forecaster: Callable[[str], Forecast] = forecast_command,
                 clock: Callable[[], float] = time.time):
        self.ledger = ledger
        self.meter = meter
        self._get = get_config
        self._set = set_config
        self._forecast = forecaster
        self._clock = clock
        self._lock = threading.Lock()
        self._expected: dict[str, int] = {}   # command -> forecast bytes, from pre_tool_call
        self._background: dict[object, dict[str, Any]] = {}

    # ── settings ──────────────────────────────────────────────────────────
    def setting(self, key: str) -> Any:
        try:
            value = self._get(key, DEFAULTS[key])
        except Exception:
            value = DEFAULTS[key]
        if key == "mode":
            return value if value in MODES else DEFAULTS["mode"]
        return _as_int(value, DEFAULTS[key])

    def paused_until(self) -> float:
        try:
            until = float(self.ledger.get_kv("paused_until", "0") or 0)
        except (TypeError, ValueError):
            return 0.0
        return until if until > self._clock() else 0.0

    def active(self) -> bool:
        return self.setting("mode") != "observe" and not self.paused_until()

    # ── budget arithmetic ─────────────────────────────────────────────────
    def budgets(self) -> list[dict[str, Any]]:
        """Every budget that is switched on, with what is used and left."""
        now = self._clock()
        out = []
        for label, key, used in (
            ("today", "daily_budget_mb", lambda: self.ledger.total(day=day_key(now))),
            ("this month", "monthly_budget_mb", lambda: self.ledger.total(month=month_key(now))),
        ):
            limit = self.setting(key) * MB
            if limit:
                spent = used()
                out.append({"label": label, "limit": limit, "used": spent, "left": limit - spent})
        return out

    def tightest(self) -> Optional[dict[str, Any]]:
        budgets = self.budgets()
        return min(budgets, key=lambda b: b["left"]) if budgets else None

    def left_pct(self) -> Optional[float]:
        tight = self.tightest()
        if not tight:
            return None
        return max(0.0, 100.0 * tight["left"] / tight["limit"])

    # ── pre_tool_call: forecast and gate ──────────────────────────────────
    def pre_tool_call(self, tool_name: str = "", args: Any = None, session_id: str = "", **_: Any) -> Optional[dict]:
        try:
            return self._gate(tool_name or "", args if isinstance(args, dict) else {}, session_id or "")
        except Exception:
            return None  # a broken meter must never break a tool call

    def _gate(self, tool_name: str, args: dict, session_id: str) -> Optional[dict]:
        if not self.active():
            return None
        mode = self.setting("mode")
        tight = self.tightest()
        spent_out = tight is not None and tight["left"] <= 0
        if tool_name == "terminal":
            command = str(args.get("command") or "")
            if not command or not looks_like_download(command):
                return None
            forecast = self._forecast(command)
            if forecast.overhead_bytes:
                self.ledger.record("preflight", "size lookup", rx=forecast.overhead_bytes, session_id=session_id,
                                   measured=False, detail=command[:200])
            if not forecast.items:
                return None
            total = forecast.total_bytes
            with self._lock:
                if len(self._expected) > 100:
                    self._expected.clear()
                self._expected[command] = total
            over = tight is not None and total > max(0, tight["left"])
            large = total > self.setting("ask_above_mb") * MB
            if not (over or large or spent_out):
                return None
            message = self._forecast_message(forecast, tight, over or spent_out)
            if mode == "enforce" and (over or spent_out):
                return {"action": "block", "message": message + " " + self._how_to_proceed()}
            return {"action": "approve", "message": message}
        if spent_out and is_network_tool(tool_name):
            message = (f"Data Saver: the data budget for {tight['label']} is spent "
                       f"({fmt_bytes(tight['used'])} of {fmt_bytes(tight['limit'])}). "
                       f"{tool_name} uses the network.")
            if mode == "enforce":
                return {"action": "block", "message": message + " " + self._how_to_proceed()}
            # One approval covers the rest of the day's network calls, not each one.
            return {"action": "approve", "message": message,
                    "rule_key": f"data-saver:over-budget:{day_key(self._clock())}"}
        return None

    @staticmethod
    def _how_to_proceed() -> str:
        return ("Do not retry or work around this. Tell the user what you wanted to download and its size; "
                "they can raise the budget with /datasaver budget <MB> or pause it with /datasaver pause <minutes>.")

    @staticmethod
    def _forecast_message(forecast: Forecast, tight: Optional[dict], over: bool) -> str:
        parts = []
        for item in forecast.sized[:4]:
            parts.append(f"{item.kind} {item.target}: {fmt_bytes(item.bytes or 0)}" + (f" ({item.note})" if item.note else ""))
        for item in forecast.unsized[:3]:
            parts.append(f"{item.kind} {item.target}: size unknown" + (f" ({item.note})" if item.note else ""))
        qualifier = "at least " if forecast.lower_bound else "about "
        head = f"Data Saver: this command will download {qualifier}{fmt_bytes(forecast.total_bytes)}."
        if not forecast.sized:
            head = "Data Saver: this command downloads data of unknown size."
        if tight is not None:
            left = max(0, tight["left"])
            head += (f" Only {fmt_bytes(left)} of the {fmt_bytes(tight['limit'])} budget for {tight['label']} is left."
                     if over else f" {fmt_bytes(left)} of the budget for {tight['label']} is left.")
        return head + " " + "; ".join(parts) + "."

    # ── tool_request: lean mode ───────────────────────────────────────────
    def tool_request(self, tool_name: str = "", args: Any = None, **_: Any) -> Optional[dict]:
        try:
            if tool_name not in ("web_extract", "web_search") or not isinstance(args, dict) or not self.active():
                return None
            threshold = self.setting("lean_below_pct")
            left = self.left_pct()
            if not threshold or left is None or left >= threshold:
                return None
            new_args = dict(args)
            if tool_name == "web_extract":
                if "char_limit" in args:
                    return None  # the model asked for a size on purpose
                new_args["char_limit"] = LEAN_EXTRACT_CHARS
            else:
                if _as_int(args.get("limit", 5), 5) <= LEAN_SEARCH_RESULTS:
                    return None
                new_args["limit"] = LEAN_SEARCH_RESULTS
            return {"args": new_args, "source": "data-saver",
                    "reason": f"lean mode: {left:.0f}% of the data budget left"}
        except Exception:
            return None

    # ── execution middleware: real metering ───────────────────────────────
    def tool_execution(self, tool_name: str = "", args: Any = None, next_call: Any = None,
                       session_id: str = "", **_: Any) -> Any:
        if tool_name in LOCAL_TOOLS:
            return next_call(args)
        detail = ""
        if isinstance(args, dict):
            detail = str(args.get("command") or args.get("urls") or args.get("url") or args.get("query") or "")
        return self._metered("tool", tool_name or "tool", detail, session_id, next_call, args)

    def llm_execution(self, request: Any = None, next_call: Any = None, session_id: str = "",
                      model: str = "", provider: str = "", **_: Any) -> Any:
        source = "/".join(part for part in (str(provider or ""), str(model or "")) if part) or "model"
        return self._metered("model", source, "", session_id, next_call, request)

    def _metered(self, category: str, source: str, detail: str, session_id: str, next_call: Any, payload: Any) -> Any:
        key = object()
        started = False
        self._sweep_background()
        try:
            started = self.meter.begin(key)
        except Exception:
            started = False
        try:
            result = next_call(payload)
        except BaseException:
            if started:
                self._finish(key, category, source, detail, session_id)
            raise
        if started:
            if inspect.isawaitable(result):
                self.meter.end(key)  # the work has not happened yet; nothing honest to record
            else:
                self._finish(key, category, source, detail, session_id)
                if category == "tool" and source == "terminal":
                    self._follow_background(detail, payload, result, session_id)
        return result

    # ── background downloads ──────────────────────────────────────────────
    def _follow_background(self, command: str, args: Any, result: Any, session_id: str) -> None:
        """A download that Hermes moved to the background keeps using data after its tool call
        returned. Keep attributing traffic to it until it goes quiet."""
        try:
            went_background = (isinstance(args, dict) and bool(args.get("background"))) or (
                isinstance(result, str) and '"session_id"' in result and "proc_" in result)
            if not went_background or not command or not looks_like_download(command):
                return
            key = object()
            if not self.meter.begin(key, absorb=True):
                return
            now = self._clock()
            with self._lock:
                self._background[key] = {"command": command[:200], "session_id": session_id or "",
                                         "expected": self._expected.get(command, 0), "got": 0,
                                         "started": now, "checked": now}
        except Exception:
            pass

    def _sweep_background(self) -> None:
        """Write what each background download has used so far; stop following finished ones."""
        try:
            with self._lock:
                following = list(self._background.items())
            now = self._clock()
            for key, info in following:
                usage = self.meter.take(key) or (0, 0)
                moved = usage[0] + usage[1]
                if moved:
                    self.ledger.record("tool", BACKGROUND_SOURCE, rx=usage[0], tx=usage[1],
                                       session_id=info["session_id"], measured=True, detail=info["command"])
                    info["got"] += moved
                quiet = now - info["checked"] >= BACKGROUND_IDLE_SECONDS and moved < BACKGROUND_IDLE_BYTES
                done = info["expected"] and info["got"] >= info["expected"] * 1.5
                if quiet or done or now - info["started"] > BACKGROUND_MAX_SECONDS:
                    self.meter.end(key)
                    with self._lock:
                        self._background.pop(key, None)
                info["checked"] = now
        except Exception:
            pass

    def _finish(self, key: object, category: str, source: str, detail: str, session_id: str) -> None:
        try:
            usage = self.meter.end(key)
            if usage and usage[0] + usage[1] >= NOISE_FLOOR:
                self.ledger.record(category, source, rx=usage[0], tx=usage[1], session_id=session_id or "",
                                   measured=True, detail=detail)
        except Exception:
            pass

    # ── reporting ─────────────────────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        self._sweep_background()
        now = self._clock()
        today, month = day_key(now), month_key(now)
        paused = self.paused_until()
        return {
            "mode": self.setting("mode"),
            "paused_minutes_left": int((paused - now) / 60) + 1 if paused else 0,
            "metering": "os network counters" if self.meter.available else "unavailable on this machine",
            "today_bytes": self.ledger.total(day=today),
            "month_bytes": self.ledger.total(month=month),
            "today_by_category": self.ledger.by_category(day=today),
            "budgets": self.budgets(),
            "ask_above_mb": self.setting("ask_above_mb"),
            "lean_mode": self._lean_on(),
            "top_today": self.ledger.top_sources(day=today, limit=5),
            "biggest_calls_today": self.ledger.biggest_calls(day=today, limit=3),
        }

    def _lean_on(self) -> bool:
        left, threshold = self.left_pct(), self.setting("lean_below_pct")
        return bool(self.active() and threshold and left is not None and left < threshold)

    def status_text(self) -> str:
        info = self.status()
        lines = [f"Data Saver — mode: {info['mode']}"
                 + (f" (paused, {info['paused_minutes_left']} min left)" if info["paused_minutes_left"] else "")]
        for budget in info["budgets"]:
            pct = 100.0 * budget["used"] / budget["limit"]
            bar = "█" * min(10, int(pct / 10)) + "░" * max(0, 10 - int(pct / 10))
            lines.append(f"{budget['label']:<10} {bar} {fmt_bytes(budget['used'])} of {fmt_bytes(budget['limit'])}"
                         f" ({pct:.0f}%), {fmt_bytes(max(0, budget['left']))} left")
        if not info["budgets"]:
            lines.append(f"today      {fmt_bytes(info['today_bytes'])} (no budget set)")
        cats = info["today_by_category"]
        lines.append(f"today: tools {fmt_bytes(cats['tool'])} · model {fmt_bytes(cats['model'])}"
                     f" · size lookups {fmt_bytes(cats['preflight'])}")
        if info["top_today"]:
            lines.append("biggest users today:")
            for row in info["top_today"]:
                lines.append(f"  {row['source']:<28} {fmt_bytes(row['bytes']):>9}  ({row['calls']} calls)")
        if info["lean_mode"]:
            lines.append("lean mode is on: web results are trimmed.")
        if info["metering"] != "os network counters":
            lines.append("metering is unavailable on this machine; forecasts still work.")
        lines.append(f"asks before any download over {info['ask_above_mb']} MB.")
        return "\n".join(lines)

    # ── /datasaver ────────────────────────────────────────────────────────
    def command(self, raw_args: str = "") -> str:
        parts = (raw_args or "").split()
        verb = parts[0].lower() if parts else "status"
        value = parts[1] if len(parts) > 1 else ""
        try:
            if verb in ("status", "top"):
                return self.status_text()
            if verb in ("budget", "daily", "monthly", "ask"):
                key = {"budget": "daily_budget_mb", "daily": "daily_budget_mb", "monthly": "monthly_budget_mb",
                       "ask": "ask_above_mb"}[verb]
                if not value.isdigit():
                    return f"Usage: /datasaver {verb} <megabytes>"
                if self._set is None:
                    return "Settings cannot be changed from here."
                self._set(key, int(value))
                return f"{key} set to {int(value)} MB.\n\n" + self.status_text()
            if verb == "mode":
                if value not in MODES:
                    return "Usage: /datasaver mode ask|enforce|observe"
                if self._set is None:
                    return "Settings cannot be changed from here."
                self._set("mode", value)
                return f"Mode set to {value}."
            if verb == "pause":
                minutes = int(value) if value.isdigit() else 60
                self.ledger.set_kv("paused_until", self._clock() + minutes * 60)
                return f"Data Saver paused for {minutes} minutes. Metering continues; nothing is gated."
            if verb == "resume":
                self.ledger.delete_kv("paused_until")
                return "Data Saver resumed."
            if verb == "check":
                command = (raw_args or "").split(None, 1)[1] if len(parts) > 1 else ""
                if not command:
                    return "Usage: /datasaver check <shell command>"
                return self.forecast_text(command)
        except Exception as exc:
            return f"Data Saver: {exc}"
        return ("Usage: /datasaver [status | budget <MB> | monthly <MB> | ask <MB> | mode ask|enforce|observe"
                " | pause <minutes> | resume | check <command>]")

    def forecast_text(self, command: str) -> str:
        forecast = self._forecast(command)
        if forecast.overhead_bytes:
            self.ledger.record("preflight", "size lookup", rx=forecast.overhead_bytes, measured=False,
                               detail=command[:200])
        if not forecast.items:
            return "Data Saver: nothing in this command looks like a download."
        return self._forecast_message(forecast, self.tightest(), False)

    # ── tools ─────────────────────────────────────────────────────────────
    def tool_data_usage(self, args: Any = None, **_: Any) -> str:
        try:
            return json.dumps(self.status())
        except Exception as exc:
            return json.dumps({"error": f"data_usage failed: {exc}"})

    def tool_data_forecast(self, args: Any = None, **_: Any) -> str:
        try:
            command = str((args or {}).get("command") or "").strip()
            if not command:
                return json.dumps({"error": "command is required"})
            forecast = self._forecast(command)
            if forecast.overhead_bytes:
                self.ledger.record("preflight", "size lookup", rx=forecast.overhead_bytes, measured=False,
                                   detail=command[:200])
            tight = self.tightest()
            out = forecast.as_dict()
            out["total"] = fmt_bytes(forecast.total_bytes)
            out["budget_left_bytes"] = max(0, tight["left"]) if tight else None
            out["fits_budget"] = (forecast.total_bytes <= max(0, tight["left"])) if tight else True
            return json.dumps(out)
        except Exception as exc:
            return json.dumps({"error": f"data_forecast failed: {exc}"})
