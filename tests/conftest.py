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
    yield
