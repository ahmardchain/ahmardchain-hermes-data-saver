"""Data Saver: a mobile-data budget for Hermes Agent."""

from __future__ import annotations

from . import forecast
from .core import DataSaver
from .ledger import Ledger
from .meter import Meter

PLUGIN_NAME = "data-saver"

DATA_USAGE_SCHEMA = {
    "name": "data_usage",
    "description": (
        "Report how much network data Hermes has used today and this month, the data budget and what is "
        "left, and which tools and model calls used the most. Use it when the user asks about data or "
        "bandwidth use, or before planning work that downloads a lot."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

DATA_FORECAST_SCHEMA = {
    "name": "data_forecast",
    "description": (
        "Forecast how many bytes a shell command would download, without running it. Understands pip / uv "
        "installs (with dependencies), git clone from GitHub, docker pull from Docker Hub, Hugging Face "
        "downloads, ollama pull and curl / wget URLs. Use it to compare options (for example a full clone "
        "against --depth 1) before spending the user's data."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to size, exactly as it would be run."},
        },
        "required": ["command"],
    },
}


def _open_db():
    from plugins.plugin_storage import plugin_db

    return plugin_db(PLUGIN_NAME)


def register(ctx) -> None:
    ledger = Ledger(_open_db)
    forecast.lookup_store = ledger  # remember package sizes between sessions
    saver = DataSaver(ledger, Meter(), ctx.get_config, ctx.set_config)

    ctx.register_tool(name="data_usage", toolset="data_saver", schema=DATA_USAGE_SCHEMA,
                      handler=saver.tool_data_usage, emoji="📶")
    ctx.register_tool(name="data_forecast", toolset="data_saver", schema=DATA_FORECAST_SCHEMA,
                      handler=saver.tool_data_forecast, emoji="📶")

    ctx.register_hook("pre_tool_call", saver.pre_tool_call)
    ctx.register_middleware("tool_request", saver.tool_request)
    ctx.register_middleware("tool_execution", saver.tool_execution)
    ctx.register_middleware("llm_execution", saver.llm_execution)

    ctx.register_command(
        "datasaver", saver.command,
        description="Data budget: usage, budget, mode, pause, and size checks",
        args_hint="[status | budget <MB> | mode ask|enforce|observe | pause <min> | resume | check <command>]",
    )
