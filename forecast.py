"""Forecast how many bytes a shell command will download, before it runs.

Sizes come from the registries themselves (PyPI, GitHub, Docker Hub, Hugging
Face, the ollama registry) or from a HEAD request for a plain URL. Nothing is
downloaded: every lookup is a small metadata request, and the bytes those
requests cost are reported so they can be charged to the ledger.
"""

from __future__ import annotations

import fnmatch
import gzip
import ipaddress
import json
import platform
import re
import shlex
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

USER_AGENT = "hermes-data-saver/0.1 (+https://github.com/ahmardchain/ahmardchain-hermes-data-saver)"
_MAX_BODY = 8 * 1024 * 1024
_HEADER_OVERHEAD = 700  # rough request + response header bytes per lookup
_MAX_PIP_PACKAGES = 60
_CACHE_TTL = 600.0

_OPERATORS = {";", "&&", "||", "|", "&", "(", ")", "|&", ";;"}
_PREFIXES = {"sudo", "nohup", "time", "command", "exec", "env", "nice", "doas"}

# Downloaders whose size cannot be looked up cheaply. They are still flagged.
_UNSIZED = {
    "npm": ({"install", "i", "ci", "add", "update", "exec"}, "npm packages"),
    "pnpm": ({"install", "i", "add", "update", "dlx"}, "pnpm packages"),
    "yarn": ({"install", "add", "upgrade", "dlx"}, "yarn packages"),
    "bun": ({"install", "add", "x"}, "bun packages"),
    "npx": (None, "an npm package"),
    "apt": ({"install", "upgrade", "full-upgrade", "dist-upgrade"}, "apt packages"),
    "apt-get": ({"install", "upgrade", "dist-upgrade"}, "apt packages"),
    "dnf": ({"install", "upgrade", "update"}, "dnf packages"),
    "yum": ({"install", "upgrade", "update"}, "yum packages"),
    "pacman": (None, "pacman packages"),
    "apk": ({"add", "upgrade"}, "apk packages"),
    "brew": ({"install", "upgrade", "reinstall"}, "Homebrew packages"),
    "snap": ({"install", "refresh"}, "a snap"),
    "flatpak": ({"install", "update"}, "a flatpak"),
    "cargo": ({"install", "build", "run", "fetch", "update", "test"}, "Rust crates"),
    "go": ({"get", "install", "mod", "build", "run"}, "Go modules"),
    "conda": ({"install", "create", "update"}, "conda packages"),
    "mamba": ({"install", "create", "update"}, "conda packages"),
    "gem": ({"install", "update"}, "Ruby gems"),
    "bundle": ({"install", "update"}, "Ruby gems"),
    "composer": ({"install", "update", "require"}, "PHP packages"),
    "playwright": ({"install"}, "browser binaries (hundreds of MB)"),
    "yt-dlp": (None, "a video"),
    "youtube-dl": (None, "a video"),
    "rsync": (None, "an rsync transfer"),
    "scp": (None, "an scp transfer"),
}


@dataclass
class Item:
    kind: str
    target: str
    bytes: Optional[int] = None  # None = size unknown
    note: str = ""
    lower_bound: bool = False  # True when the real download can be larger

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "target": self.target, "bytes": self.bytes, "note": self.note,
                "lower_bound": self.lower_bound}


@dataclass
class Forecast:
    items: list[Item] = field(default_factory=list)
    overhead_bytes: int = 0
    timed_out: bool = False

    @property
    def total_bytes(self) -> int:
        return sum(item.bytes or 0 for item in self.items)

    @property
    def sized(self) -> list[Item]:
        return [item for item in self.items if item.bytes is not None]

    @property
    def unsized(self) -> list[Item]:
        return [item for item in self.items if item.bytes is None]

    @property
    def lower_bound(self) -> bool:
        return self.timed_out or any(item.lower_bound for item in self.items) or bool(self.unsized)

    def as_dict(self) -> dict[str, Any]:
        return {"total_bytes": self.total_bytes, "lower_bound": self.lower_bound,
                "items": [item.as_dict() for item in self.items], "overhead_bytes": self.overhead_bytes}


# ── HTTP ──────────────────────────────────────────────────────────────────

class Fetcher:
    """Small metadata requests with a shared deadline and a running byte count."""

    def __init__(self, opener: Optional[Callable[..., Any]] = None, *, deadline_s: float = 10.0,
                 timeout_s: float = 6.0, clock: Callable[[], float] = time.monotonic):
        self._open = opener or urllib.request.urlopen
        self._clock = clock
        self._deadline = clock() + deadline_s
        self._timeout = timeout_s
        self._lock = threading.Lock()
        self.spent = 0
        self.timed_out = False

    def _remaining(self) -> float:
        left = self._deadline - self._clock()
        if left <= 0.05:
            self.timed_out = True
            return 0.0
        return min(self._timeout, left)

    def _charge(self, amount: int) -> None:
        with self._lock:
            self.spent += amount

    def get_json(self, url: str, headers: Optional[dict[str, str]] = None) -> Optional[Any]:
        timeout = self._remaining()
        if not timeout:
            return None
        request = urllib.request.Request(url, headers={
            "User-Agent": USER_AGENT, "Accept": "application/json", "Accept-Encoding": "gzip", **(headers or {})})
        try:
            with self._open(request, timeout=timeout) as response:
                raw = response.read(_MAX_BODY + 1)
                encoding = (response.headers.get("Content-Encoding") or "").lower()
        except Exception:
            self._charge(_HEADER_OVERHEAD)
            return None
        self._charge(len(raw) + _HEADER_OVERHEAD)
        if len(raw) > _MAX_BODY:
            return None
        try:
            if encoding == "gzip":
                raw = gzip.decompress(raw)
            return json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            return None

    def content_length(self, url: str) -> Optional[int]:
        """Size of the resource behind a URL, from response headers only."""
        timeout = self._remaining()
        if not timeout:
            return None
        self._charge(_HEADER_OVERHEAD)
        # Ask for one byte: servers that ignore HEAD still answer with the full size in Content-Range.
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"})
        try:
            with self._open(request, timeout=timeout) as response:
                headers = response.headers
                linked = headers.get("X-Linked-Size")
                content_range = headers.get("Content-Range") or ""
                length = headers.get("Content-Length")
                status = getattr(response, "status", 200)
        except Exception:
            return None
        if linked and linked.isdigit():
            return int(linked)
        match = re.search(r"/(\d+)\s*$", content_range)
        if match:
            return int(match.group(1))
        if status == 200 and length and length.isdigit():
            return int(length)
        return None


def is_public_http_url(url: str) -> bool:
    """Only plain http(s) URLs on public hosts are probed; local and private addresses never are."""
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return False
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
        return False
    host = parts.hostname.lower().rstrip(".")
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal", ".lan", ".home.arpa")):
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return "." in host
    return address.is_global


# ── command parsing ───────────────────────────────────────────────────────

def split_commands(command: str) -> list[list[str]]:
    """Split a shell line into simple commands (token lists), dropping wrappers like sudo."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        tokens = command.split()
    segments: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in _OPERATORS or (token and set(token) <= set(";&|()")):
            if current:
                segments.append(current)
            current = []
        else:
            current.append(token)
    if current:
        segments.append(current)
    return [seg for seg in (_strip_prefixes(seg) for seg in segments) if seg]


def _strip_prefixes(tokens: list[str]) -> list[str]:
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in _PREFIXES or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", token):
            index += 1
        elif index and tokens[index - 1] in ("sudo", "env", "nice", "doas") and token.startswith("-"):
            index += 1
        else:
            break
    return tokens[index:]


def _prog(tokens: list[str]) -> str:
    name = tokens[0].rsplit("/", 1)[-1].lower()
    return name[:-4] if name.endswith(".exe") else name


def _urls(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t.startswith(("http://", "https://"))]


def looks_like_download(command: str) -> bool:
    """Cheap check, no network: does this command fetch anything we know how to size or flag?"""
    for tokens in split_commands(command):
        if _classify(tokens)[0]:
            return True
    return False


def _classify(tokens: list[str]) -> tuple[str, list[str]]:
    """(kind, arguments) for one simple command; kind is '' when it is not a downloader."""
    prog = _prog(tokens)
    rest = tokens[1:]
    if prog in ("python", "python3", "py") or re.fullmatch(r"python3\.\d+", prog):
        if len(rest) >= 2 and rest[0] == "-m" and rest[1] in ("pip", "uv"):
            prog, rest = rest[1], rest[2:]
    if prog == "uv" and rest[:1] == ["pip"]:
        prog, rest = "pip", rest[1:]
    if prog in ("pip", "pip3") or re.fullmatch(r"pip3\.\d+", prog):
        return ("pip", rest[rest.index("install") + 1:]) if "install" in rest else ("", [])
    if prog in ("uv", "poetry", "pdm") and rest[:1] == ["add"]:
        return "pip", rest[1:]
    if prog == "pipx" and rest[:1] in (["install"], ["run"]):
        return "pip", rest[1:]
    if prog == "git":
        positional = [t for t in rest if not t.startswith("-")]
        if positional[:1] == ["clone"]:
            return "git", rest
        if positional[:1] in (["pull"], ["fetch"]) or positional[:2] == ["lfs", "pull"]:
            return "unsized", ["new git history"]
        return "", []
    if prog in ("docker", "podman", "nerdctl"):
        if rest[:1] == ["pull"] or rest[:2] == ["image", "pull"]:
            return "docker", [t for t in rest if not t.startswith("-") and t not in ("pull", "image")]
        if rest[:1] in (["build"], ["compose"], ["buildx"]):
            return "unsized", ["container images"]
        if rest[:1] == ["run"]:
            return "unsized", ["a container image, if it is not already pulled"]
        return "", []
    if prog in ("huggingface-cli", "hf") and rest[:1] == ["download"]:
        return "hf", rest[1:]
    if prog == "ollama" and rest[:1] in (["pull"], ["run"]):
        return "ollama", [t for t in rest[1:] if not t.startswith("-")]
    if prog in ("curl", "wget", "aria2c", "axel", "http", "https", "xh"):
        if prog == "curl" and any(t in ("-I", "--head") or (re.fullmatch(r"-[A-Za-z]+", t) and "I" in t)
                                  for t in rest):
            return "", []
        return ("url", rest) if _urls(rest) else ("", [])
    if prog in _UNSIZED:
        verbs, label = _UNSIZED[prog]
        if verbs is None or any(t in verbs for t in rest[:2]):
            return "unsized", [label]
    return "", []


# ── resolvers ─────────────────────────────────────────────────────────────

_PIP_VALUE_FLAGS = {"-c", "--constraint", "-t", "--target", "--prefix", "--root", "--python", "-f",
                    "--find-links", "--platform", "--python-version", "--implementation", "--abi",
                    "--cache-dir", "--src", "--upgrade-strategy", "--progress-bar", "--group", "--dev",
                    "--optional", "--script", "--config-settings", "-C", "--trusted-host", "--proxy",
                    "--retries", "--timeout", "--exists-action", "--cert", "--client-cert", "--log"}


def _parse_pip_args(args: list[str]) -> tuple[list[str], bool, list[Item]]:
    """(requirement strings, resolve dependencies?, items that cannot be sized)."""
    requirements: list[str] = []
    extra: list[Item] = []
    with_deps = True
    index = 0
    while index < len(args):
        arg = args[index]
        index += 1
        if arg in ("-r", "--requirement") or arg.startswith("--requirement="):
            name = arg.split("=", 1)[1] if "=" in arg else (args[index] if index < len(args) else "?")
            index += 0 if "=" in arg else 1
            extra.append(Item("pip", name, None, "requirements file: size not looked up"))
        elif arg in ("-i", "--index-url", "--extra-index-url") or arg.startswith(("--index-url=", "--extra-index-url=")):
            index += 0 if "=" in arg else 1
            extra.append(Item("pip", "custom package index", None, "sizes on a custom index may differ from PyPI"))
        elif arg in ("-e", "--editable"):
            index += 1
        elif arg == "--no-deps":
            with_deps = False
        elif arg in _PIP_VALUE_FLAGS:
            index += 1
        elif arg.startswith("-"):
            continue
        elif arg.startswith(("http://", "https://", "git+", "file:")) or "://" in arg:
            extra.append(Item("pip", arg, None, "direct URL: size not looked up"))
        elif arg in (".", "..") or arg.startswith(("./", "../", "/", "~")) or arg.endswith((".whl", ".tar.gz", ".zip")):
            continue
        else:
            requirements.append(arg)
    return requirements, with_deps, extra


def _split_requirement(text: str) -> tuple[str, Optional[str], set[str], Any]:
    """(name, pinned version or None, extras, marker or None)."""
    try:
        from packaging.requirements import Requirement

        req = Requirement(text)
        pinned = None
        specs = list(req.specifier)
        if len(specs) == 1 and specs[0].operator in ("==", "===") and "*" not in specs[0].version:
            pinned = specs[0].version
        return req.name, pinned, set(req.extras), req.marker
    except Exception:
        match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[([^\]]*)\])?\s*(?:==\s*([A-Za-z0-9.+!_-]+))?", text)
        if not match:
            return "", None, set(), None
        extras = {e.strip() for e in (match.group(2) or "").split(",") if e.strip()}
        return match.group(1), match.group(3), extras, None


def _marker_ok(marker: Any, extras: set[str]) -> bool:
    if marker is None:
        return True
    try:
        return any(marker.evaluate({"extra": extra}) for extra in (extras or {""}))
    except Exception:
        return False


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _pick_distribution(files: list[dict[str, Any]]) -> Optional[int]:
    """Size of the file pip would download on this machine."""
    wheels = [f for f in files if f.get("packagetype") == "bdist_wheel" and isinstance(f.get("size"), int)]
    sdists = [f for f in files if f.get("packagetype") == "sdist" and isinstance(f.get("size"), int)]
    try:
        from packaging.tags import sys_tags
        from packaging.utils import parse_wheel_filename

        by_tag: dict[Any, int] = {}
        for wheel in wheels:
            try:
                tags = parse_wheel_filename(wheel["filename"])[3]
            except Exception:
                continue
            for tag in tags:
                by_tag.setdefault(tag, wheel["size"])
        for tag in sys_tags():
            if tag in by_tag:
                return by_tag[tag]
    except Exception:
        machine = platform.machine().lower()
        system = platform.system().lower()
        want = {"linux": "linux", "darwin": "macosx", "windows": "win"}.get(system, "")
        for wheel in wheels:
            name = wheel["filename"].lower()
            if name.endswith("-none-any.whl") or (want and want in name and (machine in name or "universal2" in name)):
                return wheel["size"]
    if sdists:
        return sdists[0]["size"]
    if wheels:
        return max(w["size"] for w in wheels)
    return None


class MemoryStore:
    """Default lookup store: lives for the process. The plugin swaps in a store backed by its database
    so a package is looked up on PyPI once, not on every forecast."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[float, Any]] = {}

    def get(self, key: str, max_age: float) -> Optional[Any]:
        hit = self._data.get(key)
        return hit[1] if hit and time.time() - hit[0] < max_age else None

    def put(self, key: str, value: Any) -> None:
        self._data[key] = (time.time(), value)


lookup_store: Any = MemoryStore()
_PINNED_TTL = 30 * 86400.0  # a released version never changes
_LATEST_TTL = 86400.0


def _pypi_lookup(fetcher: Fetcher, name: str, pinned: Optional[str]) -> Optional[dict[str, Any]]:
    key = f"pypi:{_canonical(name)}:{pinned or 'latest'}"
    try:
        cached = lookup_store.get(key, _PINNED_TTL if pinned else _LATEST_TTL)
    except Exception:
        cached = None
    if isinstance(cached, dict) and "size" in cached:
        return cached
    found = _pypi_fetch(fetcher, name, pinned)
    if found and found["size"] is not None:
        try:
            lookup_store.put(key, found)
        except Exception:
            pass
    return found


def _pypi_fetch(fetcher: Fetcher, name: str, pinned: Optional[str]) -> Optional[dict[str, Any]]:
    quoted = urllib.parse.quote(name, safe="")
    url = (f"https://pypi.org/pypi/{quoted}/{urllib.parse.quote(pinned, safe='')}/json" if pinned
           else f"https://pypi.org/pypi/{quoted}/json")
    data = fetcher.get_json(url)
    if not isinstance(data, dict) or not isinstance(data.get("urls"), list):
        return None
    info = data.get("info") if isinstance(data.get("info"), dict) else {}
    return {"size": _pick_distribution(data["urls"]), "requires": info.get("requires_dist") or [],
            "version": info.get("version") or pinned or ""}


def _resolve_pip(fetcher: Fetcher, args: list[str]) -> list[Item]:
    requirements, with_deps, items = _parse_pip_args(args)
    if not requirements:
        return items
    seen: dict[str, Optional[int]] = {}
    roots: dict[str, str] = {}
    queue: list[tuple[str, Optional[str], set[str]]] = []
    for text in requirements:
        name, pinned, extras, _marker = _split_requirement(text)
        if not name:
            items.append(Item("pip", text, None, "could not parse this requirement"))
            continue
        roots[_canonical(name)] = text
        queue.append((name, pinned, extras))
    truncated = False
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        while queue:
            wave: list[tuple[str, Optional[str], set[str]]] = []
            for name, pinned, extras in queue:
                key = _canonical(name)
                if key in seen:
                    continue
                if len(seen) >= _MAX_PIP_PACKAGES:
                    truncated = True
                    break
                seen[key] = None
                wave.append((name, pinned, extras))
            queue = []
            futures = {pool.submit(_pypi_lookup, fetcher, name, pinned): (name, extras) for name, pinned, extras in wave}
            for future in as_completed(futures):
                name, extras = futures[future]
                try:
                    found = future.result()
                except Exception:
                    found = None
                if not found or found["size"] is None:
                    failed.append(name)
                    continue
                seen[_canonical(name)] = found["size"]
                if not with_deps:
                    continue
                for dep_text in found["requires"]:
                    dep_name, dep_pin, dep_extras, marker = _split_requirement(dep_text)
                    if dep_name and _marker_ok(marker, extras) and _canonical(dep_name) not in seen:
                        queue.append((dep_name, dep_pin, dep_extras))
    total = sum(size for size in seen.values() if size)
    sized = sum(1 for size in seen.values() if size)
    label = ", ".join(roots.values())
    if not sized:
        items.append(Item("pip", label, None, "PyPI did not answer in time" if fetcher.timed_out else "not found on PyPI"))
        return items
    note = f"{sized} package{'s' if sized != 1 else ''} from PyPI"
    note += " incl. dependencies" if with_deps and sized > len(roots) else ""
    note += "; already-installed or cached packages are not re-downloaded"
    lower = truncated or bool(failed) or fetcher.timed_out
    if lower:
        note += "; some dependencies could not be sized"
    items.append(Item("pip", label, total, note, lower_bound=lower))
    return items


_GITHUB_RE = re.compile(r"(?:https?://|git@|ssh://git@)(?:www\.)?github\.com[:/]([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?$")


def _resolve_git(fetcher: Fetcher, args: list[str]) -> list[Item]:
    shallow = any(a == "--depth" or a.startswith(("--depth=", "--shallow-since", "--filter")) for a in args)
    target = next((a for a in args if "://" in a or a.startswith("git@")), "")
    if not target:
        return []
    match = _GITHUB_RE.match(target)
    if not match:
        return [Item("git clone", target, None, "only GitHub repositories can be sized")]
    owner, repo = match.group(1), match.group(2)
    data = fetcher.get_json(f"https://api.github.com/repos/{owner}/{repo}",
                            {"Accept": "application/vnd.github+json"})
    size_kb = data.get("size") if isinstance(data, dict) else None
    if not isinstance(size_kb, int):
        return [Item("git clone", f"{owner}/{repo}", None, "GitHub did not report a size")]
    note = "full history, as GitHub reports it"
    if shallow:
        note = "size of the full history; this shallow clone will download less"
    else:
        note += "; add --depth 1 to download much less"
    return [Item("git clone", f"{owner}/{repo}", size_kb * 1024, note)]


def _docker_arch() -> str:
    machine = platform.machine().lower()
    return {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(machine, "amd64")


_MANIFEST_ACCEPT = ", ".join((
    "application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json", "application/vnd.docker.distribution.manifest.v2+json"))


def _layers_total(manifest: Any) -> Optional[int]:
    if not isinstance(manifest, dict) or not isinstance(manifest.get("layers"), list):
        return None
    total = sum(layer.get("size", 0) for layer in manifest["layers"] if isinstance(layer, dict))
    config = manifest.get("config")
    if isinstance(config, dict):
        total += config.get("size", 0) or 0
    return int(total) if total else None


def _resolve_docker(fetcher: Fetcher, args: list[str]) -> list[Item]:
    items: list[Item] = []
    for ref in args:
        name, _, digest = ref.partition("@")
        last = name.rsplit("/", 1)[-1]
        repo, tag = (name.rsplit(":", 1) if ":" in last else (name, "latest"))
        first = repo.split("/", 1)[0]
        if "/" in repo and ("." in first or ":" in first or first == "localhost"):
            if first not in ("docker.io", "index.docker.io", "registry-1.docker.io"):
                items.append(Item("docker pull", ref, None, "only Docker Hub images can be sized"))
                continue
            repo = repo.split("/", 1)[1]
        if "/" not in repo:
            repo = f"library/{repo}"
        token_data = fetcher.get_json(
            f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull")
        token = token_data.get("token") if isinstance(token_data, dict) else None
        if not token:
            items.append(Item("docker pull", ref, None, "Docker Hub did not answer"))
            continue
        headers = {"Authorization": f"Bearer {token}", "Accept": _MANIFEST_ACCEPT}
        base = f"https://registry-1.docker.io/v2/{repo}/manifests/"
        manifest = fetcher.get_json(base + (digest or tag), headers)
        if isinstance(manifest, dict) and isinstance(manifest.get("manifests"), list):
            arch = _docker_arch()
            chosen = next((m for m in manifest["manifests"] if isinstance(m, dict)
                           and (m.get("platform") or {}).get("os") == "linux"
                           and (m.get("platform") or {}).get("architecture") == arch), None)
            manifest = fetcher.get_json(base + chosen["digest"], headers) if chosen else None
        total = _layers_total(manifest)
        if total is None:
            items.append(Item("docker pull", ref, None, "image manifest not found"))
        else:
            items.append(Item("docker pull", ref, total,
                              f"compressed layers for linux/{_docker_arch()}; layers you already have are skipped"))
    return items


def _resolve_hf(fetcher: Fetcher, args: list[str]) -> list[Item]:
    repo_type = "model"
    include: list[str] = []
    exclude: list[str] = []
    positional: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        index += 1
        if arg == "--repo-type" and index < len(args):
            repo_type = args[index]
            index += 1
        elif arg.startswith("--repo-type="):
            repo_type = arg.split("=", 1)[1]
        elif arg in ("--include", "--exclude"):
            bucket = include if arg == "--include" else exclude
            while index < len(args) and not args[index].startswith("-"):
                bucket.append(args[index])
                index += 1
        elif arg in ("--revision", "--cache-dir", "--local-dir", "--token", "--max-workers"):
            index += 1
        elif not arg.startswith("-"):
            positional.append(arg)
    if not positional:
        return []
    repo, files = positional[0], positional[1:]
    plural = {"model": "models", "dataset": "datasets", "space": "spaces"}.get(repo_type, "models")
    data = fetcher.get_json(f"https://huggingface.co/api/{plural}/{urllib.parse.quote(repo, safe='/')}?blobs=true")
    siblings = data.get("siblings") if isinstance(data, dict) else None
    if not isinstance(siblings, list):
        return [Item("hugging face", repo, None, "repository is private, gated or not found")]
    total = count = 0
    for entry in siblings:
        name, size = entry.get("rfilename", ""), entry.get("size")
        if not isinstance(size, int):
            continue
        if files and name not in files:
            continue
        if include and not any(fnmatch.fnmatch(name, pattern) for pattern in include):
            continue
        if exclude and any(fnmatch.fnmatch(name, pattern) for pattern in exclude):
            continue
        total += size
        count += 1
    if not count:
        return [Item("hugging face", repo, None, "no matching files")]
    return [Item("hugging face", repo, total, f"{count} file{'s' if count != 1 else ''}; cached files are skipped")]


def _resolve_ollama(fetcher: Fetcher, args: list[str]) -> list[Item]:
    if not args:
        return []
    ref = args[0]
    name, _, tag = ref.partition(":")
    path = name if "/" in name else f"library/{name}"
    manifest = fetcher.get_json(
        f"https://registry.ollama.ai/v2/{path}/manifests/{tag or 'latest'}",
        {"Accept": "application/vnd.docker.distribution.manifest.v2+json"})
    total = _layers_total(manifest)
    if total is None:
        return [Item("ollama", ref, None, "model not found in the ollama registry")]
    return [Item("ollama", ref, total, "model layers; layers you already have are skipped")]


def _resolve_urls(fetcher: Fetcher, args: list[str], pipes_to_shell: bool) -> list[Item]:
    items: list[Item] = []
    for url in _urls(args):
        short = url if len(url) <= 80 else url[:77] + "..."
        if not is_public_http_url(url):
            items.append(Item("url", short, None, "local or private address: not probed"))
            continue
        size = fetcher.content_length(url)
        if size is None:
            items.append(Item("url", short, None, "the server did not report a size"))
        else:
            items.append(Item("url", short, size))
    if pipes_to_shell:
        items.append(Item("installer", "script piped into a shell", None,
                          "the script can download more than its own size"))
    return items


# ── entry point ───────────────────────────────────────────────────────────

_cache: dict[str, tuple[float, Forecast]] = {}
_cache_lock = threading.Lock()


def forecast_command(command: str, *, fetcher: Optional[Fetcher] = None, use_cache: bool = True) -> Forecast:
    """Forecast the download size of a shell command. Never raises."""
    command = (command or "").strip()
    if not command:
        return Forecast()
    now = time.monotonic()
    if use_cache:
        with _cache_lock:
            hit = _cache.get(command)
            if hit and now - hit[0] < _CACHE_TTL:
                cached = hit[1]
                return Forecast(items=list(cached.items), overhead_bytes=0, timed_out=cached.timed_out)
    fetcher = fetcher or Fetcher()
    result = Forecast()
    segments = split_commands(command)
    for position, tokens in enumerate(segments):
        try:
            kind, args = _classify(tokens)
            if kind == "pip":
                result.items += _resolve_pip(fetcher, args)
            elif kind == "git":
                result.items += _resolve_git(fetcher, args)
            elif kind == "docker":
                result.items += _resolve_docker(fetcher, args)
            elif kind == "hf":
                result.items += _resolve_hf(fetcher, args)
            elif kind == "ollama":
                result.items += _resolve_ollama(fetcher, args)
            elif kind == "url":
                following = segments[position + 1] if position + 1 < len(segments) else []
                to_shell = bool(following) and _prog(following) in ("sh", "bash", "zsh", "dash", "python", "python3")
                result.items += _resolve_urls(fetcher, args, to_shell and "|" in command)
            elif kind == "unsized":
                result.items.append(Item(_prog(tokens), args[0], None, "size cannot be looked up in advance"))
        except Exception:
            result.items.append(Item(_prog(tokens), " ".join(tokens)[:80], None, "could not be sized"))
    result.overhead_bytes = fetcher.spent
    result.timed_out = fetcher.timed_out
    if use_cache and result.items and not result.timed_out:
        with _cache_lock:
            if len(_cache) > 200:
                _cache.clear()
            _cache[command] = (now, result)
    return result


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()
