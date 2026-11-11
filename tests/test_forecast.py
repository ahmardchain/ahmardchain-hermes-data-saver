import gzip
import json
import urllib.error

import pytest

from data_saver import forecast as fc

MB = 1024 * 1024


class FakeResponse:
    def __init__(self, body=b"", headers=None, status=200):
        self._body, self.headers, self.status = body, headers or {}, status

    def read(self, limit=-1):
        return self._body if limit < 0 else self._body[:limit]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeWeb:
    """Maps URL -> JSON object, or URL -> headers for size probes. Records every request."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, request, timeout=None):
        url = request.full_url
        self.calls.append(request)
        if url not in self.routes:
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)
        value = self.routes[url]
        if isinstance(value, FakeResponse):
            return value
        body = gzip.compress(json.dumps(value).encode())
        return FakeResponse(body, {"Content-Encoding": "gzip"})


def run(command, routes):
    web = FakeWeb(routes)
    result = fc.forecast_command(command, fetcher=fc.Fetcher(web), use_cache=False)
    return result, web


def wheel(name, size, tag="py3-none-any"):
    return {"filename": f"{name}-1.0-{tag}.whl", "packagetype": "bdist_wheel", "size": size}


def pypi(name, size, requires=None, version="1.0"):
    return {"info": {"version": version, "requires_dist": requires or []}, "urls": [wheel(name, size)]}


# ── parsing ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("command,expected", [
    ("pip install requests", True),
    ("sudo -E pip3 install -U torch", True),
    ("python -m pip install numpy", True),
    ("uv pip install pandas", True),
    ("uv add httpx", True),
    ("cd /tmp && git clone https://github.com/a/b", True),
    ("docker pull ubuntu:24.04", True),
    ("huggingface-cli download org/model", True),
    ("ollama pull llama3", True),
    ("curl -LO https://example.com/big.iso", True),
    ("FOO=1 wget https://example.com/x.tar.gz", True),
    ("npm install", True),
    ("apt-get install -y ffmpeg", True),
    ("ls -la", False),
    ("pip list", False),
    ("git status", False),
    ("curl -I https://example.com", False),
    ("echo pip install torch", False),
    ("docker ps", False),
    ("", False),
])
def test_looks_like_download(command, expected):
    assert fc.looks_like_download(command) is expected


def test_split_commands_handles_operators_and_bad_quotes():
    assert fc.split_commands("cd x && sudo pip install a; ls | wc") == [
        ["cd", "x"], ["pip", "install", "a"], ["ls"], ["wc"]]
    assert fc.split_commands("echo 'unterminated")  # falls back instead of raising


@pytest.mark.parametrize("url,ok", [
    ("https://example.com/file", True),
    ("http://8.8.8.8/x", True),
    ("http://localhost:8000/x", False),
    ("http://127.0.0.1/x", False),
    ("http://169.254.169.254/latest/meta-data", False),
    ("http://10.0.0.5/x", False),
    ("http://192.168.1.1/x", False),
    ("http://[::1]/x", False),
    ("https://user:pw@example.com/x", False),
    ("ftp://example.com/x", False),
    ("http://printer.local/x", False),
    ("http://intranet/x", False),
])
def test_is_public_http_url(url, ok):
    assert fc.is_public_http_url(url) is ok


# ── pip ───────────────────────────────────────────────────────────────────

def test_pip_sums_package_and_dependencies():
    result, _ = run("pip install alpha", {
        "https://pypi.org/pypi/alpha/json": pypi("alpha", 10 * MB, ["beta>=1", "gamma; extra == 'dev'"]),
        "https://pypi.org/pypi/beta/json": pypi("beta", 5 * MB, ["alpha"]),  # cycle must not loop
    })
    assert result.total_bytes == 15 * MB
    assert not result.lower_bound
    assert "2 packages" in result.items[0].note


def test_pip_no_deps_and_pinned_version():
    result, web = run("pip install --no-deps alpha==2.0", {
        "https://pypi.org/pypi/alpha/2.0/json": pypi("alpha", 7 * MB, ["beta"], version="2.0"),
    })
    assert result.total_bytes == 7 * MB
    assert len(web.calls) == 1


def test_pip_extras_pull_in_marked_dependencies():
    result, _ = run("pip install 'alpha[dev]'", {
        "https://pypi.org/pypi/alpha/json": pypi("alpha", MB, ["gamma; extra == 'dev'"]),
        "https://pypi.org/pypi/gamma/json": pypi("gamma", 2 * MB),
    })
    assert result.total_bytes == 3 * MB


def test_pip_unknown_package_is_unsized_not_zero():
    result, _ = run("pip install doesnotexist", {})
    assert result.total_bytes == 0
    assert result.unsized and result.lower_bound


def test_pip_missing_dependency_marks_lower_bound():
    result, _ = run("pip install alpha", {
        "https://pypi.org/pypi/alpha/json": pypi("alpha", MB, ["missing"]),
    })
    assert result.total_bytes == MB and result.lower_bound


def test_pip_requirements_file_and_custom_index_are_flagged():
    result, _ = run("pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu", {})
    notes = " ".join(item.note for item in result.items)
    assert "requirements file" in notes and "custom index" in notes
    assert all(item.bytes is None for item in result.items)


def test_pip_skips_local_paths_and_editable():
    result, web = run("pip install -e . ./dist/pkg.whl", {})
    assert result.items == [] and web.calls == []


def test_pick_distribution_prefers_a_wheel_this_machine_can_use():
    files = [
        {"filename": "pkg-1.0.tar.gz", "packagetype": "sdist", "size": 111},
        {"filename": "pkg-1.0-cp27-cp27m-win32.whl", "packagetype": "bdist_wheel", "size": 999},
        wheel("pkg", 222),
    ]
    assert fc._pick_distribution(files) == 222
    assert fc._pick_distribution(files[:2]) == 111  # no usable wheel -> sdist
    assert fc._pick_distribution([]) is None


# ── git, docker, hugging face, ollama, urls ───────────────────────────────

def test_git_clone_github_size_and_shallow_note():
    routes = {"https://api.github.com/repos/torvalds/linux": {"size": 5_000_000}}
    full, _ = run("git clone https://github.com/torvalds/linux.git", routes)
    assert full.total_bytes == 5_000_000 * 1024
    assert "--depth 1" in full.items[0].note
    shallow, _ = run("git clone --depth 1 git@github.com:torvalds/linux.git", routes)
    assert "shallow" in shallow.items[0].note


def test_git_clone_other_hosts_are_unsized():
    result, web = run("git clone https://gitlab.com/a/b.git", {})
    assert result.unsized and web.calls == []


def test_docker_pull_resolves_platform_manifest(monkeypatch):
    monkeypatch.setattr(fc, "_docker_arch", lambda: "arm64")
    base = "https://registry-1.docker.io/v2/library/ubuntu/manifests/"
    result, web = run("docker pull ubuntu:24.04", {
        "https://auth.docker.io/token?service=registry.docker.io&scope=repository:library/ubuntu:pull": {"token": "t"},
        base + "24.04": {"manifests": [
            {"digest": "sha256:amd", "platform": {"os": "linux", "architecture": "amd64"}},
            {"digest": "sha256:arm", "platform": {"os": "linux", "architecture": "arm64"}}]},
        base + "sha256:arm": {"config": {"size": 1000}, "layers": [{"size": 30 * MB}, {"size": 2 * MB}]},
    })
    assert result.total_bytes == 32 * MB + 1000
    assert web.calls[-1].get_header("Authorization") == "Bearer t"


def test_docker_pull_other_registry_is_unsized():
    result, web = run("docker pull ghcr.io/org/app:1", {})
    assert result.unsized and web.calls == []


def test_hugging_face_filters_files():
    routes = {"https://huggingface.co/api/models/org/model?blobs=true": {"siblings": [
        {"rfilename": "model.safetensors", "size": 4000 * MB},
        {"rfilename": "config.json", "size": 1000},
        {"rfilename": "model.Q4.gguf", "size": 900 * MB}]}}
    everything, _ = run("huggingface-cli download org/model", routes)
    assert everything.total_bytes == 4900 * MB + 1000
    one, _ = run("hf download org/model config.json", routes)
    assert one.total_bytes == 1000
    pattern, _ = run("hf download org/model --include '*.gguf'", routes)
    assert pattern.total_bytes == 900 * MB


def test_hugging_face_dataset_route():
    result, web = run("hf download org/data --repo-type dataset", {
        "https://huggingface.co/api/datasets/org/data?blobs=true": {"siblings": [{"rfilename": "a", "size": 5}]}})
    assert result.total_bytes == 5


def test_ollama_pull():
    result, _ = run("ollama pull llama3:8b", {
        "https://registry.ollama.ai/v2/library/llama3/manifests/8b": {
            "config": {"size": 500}, "layers": [{"size": 4000 * MB}]}})
    assert result.total_bytes == 4000 * MB + 500


def test_url_sizes_from_content_range_and_length():
    result, web = run("curl -LO https://example.com/a.iso && wget https://example.com/b.zip", {
        "https://example.com/a.iso": FakeResponse(headers={"Content-Range": "bytes 0-0/734003200"}, status=206),
        "https://example.com/b.zip": FakeResponse(headers={"Content-Length": "2048"}, status=200),
    })
    assert result.total_bytes == 734003200 + 2048
    assert all(call.get_header("Range") == "bytes=0-0" for call in web.calls)


def test_url_never_probes_private_addresses():
    result, web = run("curl http://169.254.169.254/latest/meta-data/ http://localhost:3000/x", {})
    assert web.calls == []
    assert len(result.unsized) == 2


def test_curl_pipe_to_shell_is_flagged():
    result, _ = run("curl -fsSL https://example.com/install.sh | sh", {
        "https://example.com/install.sh": FakeResponse(headers={"Content-Length": "900"})})
    assert any(item.kind == "installer" for item in result.items)
    assert result.lower_bound


def test_unsized_downloaders_are_reported():
    result, _ = run("npm install && playwright install chromium", {})
    assert [item.kind for item in result.items] == ["npm", "playwright"]
    assert result.total_bytes == 0 and result.lower_bound


# ── fetcher behaviour ─────────────────────────────────────────────────────

def test_overhead_is_counted_and_deadline_stops_requests():
    web = FakeWeb({"https://pypi.org/pypi/alpha/json": pypi("alpha", MB)})
    ticks = iter([0.0, 0.0, 100.0, 100.0, 100.0, 100.0])
    fetcher = fc.Fetcher(web, deadline_s=6.0, clock=lambda: next(ticks, 100.0))
    assert fetcher.get_json("https://pypi.org/pypi/alpha/json") is not None
    assert fetcher.spent > 0
    assert fetcher.get_json("https://pypi.org/pypi/alpha/json") is None
    assert fetcher.timed_out and len(web.calls) == 1


def test_forecast_never_raises_on_garbage():
    for command in ["pip install", "git clone", "docker pull", "hf download", "ollama pull", "curl", "'", "&& ;;"]:
        fc.forecast_command(command, fetcher=fc.Fetcher(FakeWeb({})), use_cache=False)


def test_cache_avoids_second_lookup():
    fc.clear_cache()
    web = FakeWeb({"https://pypi.org/pypi/alpha/json": pypi("alpha", MB)})
    first = fc.forecast_command("pip install alpha", fetcher=fc.Fetcher(web))
    second = fc.forecast_command("pip install alpha", fetcher=fc.Fetcher(web))
    assert first.total_bytes == second.total_bytes == MB
    assert len(web.calls) == 1 and second.overhead_bytes == 0
    fc.clear_cache()


def test_package_sizes_are_remembered_between_forecasts(monkeypatch):
    monkeypatch.setattr(fc, "lookup_store", fc.MemoryStore())
    routes = {"https://pypi.org/pypi/alpha/json": pypi("alpha", MB)}
    first, web1 = run("pip install alpha", routes)
    second, web2 = run("pip install alpha && echo done", routes)
    assert first.total_bytes == second.total_bytes == MB
    assert len(web1.calls) == 1 and web2.calls == [] and second.overhead_bytes == 0


def test_ledger_backed_store_expires(monkeypatch):
    import sqlite3
    from data_saver.ledger import Ledger
    clock = {"now": 1000.0}
    ledger = Ledger(lambda: sqlite3.connect(":memory:"), clock=lambda: clock["now"])
    ledger.put("pypi:alpha:latest", {"size": 5, "requires": [], "version": "1"})
    assert ledger.get("pypi:alpha:latest", 100)["size"] == 5
    clock["now"] += 101
    assert ledger.get("pypi:alpha:latest", 100) is None
    assert ledger.get("missing", 100) is None


# ── probe safety (SSRF) ───────────────────────────────────────────────────

@pytest.mark.parametrize("url,ok", [
    ("https://example.com/file.iso", True),
    ("http://8.8.8.8/x", True),
    ("https://rebind.example.org/x", False),        # public name -> 127.0.0.1
    ("http://metadata.example.org/latest/", False),  # public name -> cloud metadata address
    ("https://mixed.example.org/x", False),          # one private answer is enough
    ("https://intranet.corp.example/x", False),      # intranet name -> 10.x
    ("https://v6local.example.org/x", False),
    ("https://mapped.example.org/x", False),         # IPv4-mapped private address
    ("https://does-not-resolve.example/x", False),   # fails closed
    ("http://localhost/x", False),
    ("http://169.254.169.254/latest/meta-data", False),
])
def test_is_probe_safe_resolves_names_and_fails_closed(url, ok):
    assert fc.is_probe_safe(url) is ok


def test_probe_uses_hermes_url_check_when_available(monkeypatch):
    import sys, types
    seen = []
    module = types.ModuleType("tools.url_safety")
    module.is_safe_url = lambda url: (seen.append(url), False)[1]
    monkeypatch.setitem(sys.modules, "tools", types.ModuleType("tools"))
    monkeypatch.setitem(sys.modules, "tools.url_safety", module)
    assert fc.is_probe_safe("https://example.com/a") is False
    assert seen == ["https://example.com/a"]


def redirect(url, location, code=302):
    return urllib.error.HTTPError(url, code, "Found", {"Location": location}, None)


class RedirectWeb(FakeWeb):
    def __call__(self, request, timeout=None):
        value = self.routes.get(request.full_url)
        if isinstance(value, Exception):
            self.calls.append(request)
            raise value
        return super().__call__(request, timeout)


def test_probe_never_requests_a_name_that_resolves_to_a_private_address():
    web = RedirectWeb({})
    result = fc.forecast_command("curl -O https://rebind.example.org/a.iso http://metadata.example.org/x",
                                 fetcher=fc.Fetcher(web), use_cache=False)
    assert web.calls == []
    assert len(result.unsized) == 2 and "not probed" in result.unsized[0].note


def test_probe_follows_a_public_redirect_after_checking_the_hop():
    web = RedirectWeb({
        "https://example.com/a.iso": redirect("https://example.com/a.iso", "https://cdn.example.net/a.iso"),
        "https://cdn.example.net/a.iso": FakeResponse(headers={"Content-Range": "bytes 0-0/5000"}, status=206),
    })
    assert fc.Fetcher(web).content_length("https://example.com/a.iso") == 5000
    assert [call.full_url for call in web.calls] == ["https://example.com/a.iso", "https://cdn.example.net/a.iso"]


@pytest.mark.parametrize("target", [
    "http://127.0.0.1:8080/admin",
    "http://169.254.169.254/latest/meta-data/",
    "https://rebind.example.org/x",
    "http://[::1]/x",
    "/relative-is-fine-but-then-private",
])
def test_probe_does_not_follow_a_redirect_to_a_private_address(target):
    routes = {"https://example.com/a.iso": redirect("https://example.com/a.iso", target)}
    if target.startswith("/"):
        routes["https://example.com/relative-is-fine-but-then-private"] = redirect(
            "https://example.com/relative-is-fine-but-then-private", "http://10.0.0.1/")
    web = RedirectWeb(routes)
    assert fc.Fetcher(web).content_length("https://example.com/a.iso") is None
    assert all(call.full_url.startswith("https://example.com/") for call in web.calls)


def test_probe_stops_after_too_many_redirects():
    routes = {f"https://example.com/{i}": redirect(f"https://example.com/{i}", f"https://example.com/{i + 1}")
              for i in range(10)}
    web = RedirectWeb(routes)
    assert fc.Fetcher(web).content_length("https://example.com/0") is None
    assert len(web.calls) == fc._MAX_REDIRECTS + 1


def test_default_probe_opener_does_not_follow_redirects():
    handler = fc._NoRedirect()
    assert handler.redirect_request(None, None, 302, "Found", {}, "http://127.0.0.1/") is None
    assert any(isinstance(h, fc._NoRedirect) for h in fc._probe_opener.handlers)
    assert fc.Fetcher()._open_probe == fc._probe_opener.open
