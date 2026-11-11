"""Load the plugin directory as a package named ``data_saver`` (its folder name has a hyphen)."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if "data_saver" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "data_saver", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["data_saver"] = module
    spec.loader.exec_module(module)


import pytest


@pytest.fixture(autouse=True)
def _fresh_lookup_store():
    from data_saver import forecast

    forecast.lookup_store = forecast.MemoryStore()
    forecast.clear_cache()
    saved = forecast.resolve_host
    forecast.resolve_host = lambda host: FAKE_DNS[host]  # KeyError = does not resolve
    yield
    forecast.resolve_host = saved


FAKE_DNS = {
    "example.com": ["93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"],
    "8.8.8.8": ["8.8.8.8"],
    "cdn.example.net": ["151.101.1.69"],
    "rebind.example.org": ["127.0.0.1"],
    "metadata.example.org": ["169.254.169.254"],
    "mixed.example.org": ["93.184.216.34", "10.0.0.7"],
    "intranet.corp.example": ["10.20.30.40"],
    "v6local.example.org": ["::1"],
    "mapped.example.org": ["::ffff:192.168.1.5"],
}
