# Data Saver

**A mobile-data budget for Hermes Agent.** Hermes tells you how many megabytes a command will download *before* it runs, asks first when it would blow your budget, and meters what every tool call and model request really used.

> Everyone meters tokens and dollars. If you are on a metered connection, the thing that runs out is megabytes.

What Hermes's approval prompt shows when the agent tries `pip install torch` on a 500 MB daily budget:

```
Data Saver: this command will download about 2.90 GB. Only 412 MB of the 500 MB
budget for today is left. pip torch: 2.90 GB (29 packages from PyPI incl.
dependencies; already-installed or cached packages are not re-downloaded).
```

Built for the Nous Research Hermes Agent Plugin Hackathon.

## Try it in two minutes

```bash
hermes plugins install ahmardchain/ahmardchain-hermes-data-saver
hermes plugins enable data-saver
hermes
```

No API keys, no account, no extra Python packages. Then, inside Hermes:

| Type this | What you should see |
|---|---|
| `/datasaver` | Today's usage, the budget bar, the biggest users |
| `/datasaver check pip install torch` | A size forecast in about a second, nothing downloaded |
| `/datasaver budget 50` | Budget set to 50 MB a day |
| "clone https://github.com/torvalds/linux" | Hermes's approval prompt, with the size and a `--depth 1` hint |
| "how much data have you used today?" | Hermes answers from the `data_usage` tool |

## Why this exists

On a home fibre line nobody cares that `pip install torch` is 2.9 GB. On a phone hotspot or a prepaid data bundle, one careless command from an agent is the whole week's data. An agent has no feel for this: it will happily pull a 5 GB model to answer a question that needed none of it.

A prompt or a skill cannot fix that. A prompt can ask the model to be careful; it cannot know what a wheel weighs, it cannot stop a tool call, and it cannot read the network counters. Data Saver does all three, in code, on every call.

## How it works

| Step | Hermes surface | What Data Saver does |
|---|---|---|
| Forecast | `pre_tool_call` hook | Reads the `terminal` command, asks the registry how big the download is, compares it with what is left |
| Gate | Hermes's own approval prompt | Over budget or over the ask threshold → `approve` directive (you decide). In `enforce` mode → `block`, with a message the model can act on |
| Meter | `tool_execution` and `llm_execution` middleware | Reads the OS network counters before and after each call and writes the real bytes to a local ledger |
| Go lean | `tool_request` middleware | When the budget runs low, `web_extract` and `web_search` return smaller results, so less is re-sent to the model on every later turn |
| Report | `data_usage` and `data_forecast` tools, `/datasaver` command | The agent can check before it spends; you can see where the data went |

### What it can size before running

| Command | Where the size comes from |
|---|---|
| `pip install`, `uv pip install`, `uv add`, `poetry add`, `pipx` | PyPI: the wheel your machine would get, plus its dependencies |
| `git clone` (GitHub) | GitHub repository size |
| `docker pull` (Docker Hub) | Compressed layers for your CPU architecture |
| `huggingface-cli download`, `hf download` | File sizes from the Hub, honouring file names, `--include` and `--exclude` |
| `ollama pull`, `ollama run` | Model layers from the ollama registry |
| `curl`, `wget`, `aria2c` | Response headers for the URL (one byte requested, nothing downloaded) |

`npm`, `apt`, `brew`, `cargo`, `conda`, `playwright install`, `yt-dlp` and similar are recognised as downloads of unknown size. They are flagged once the budget is spent rather than guessed at.

A forecast is a few small metadata requests, capped at ten seconds. What they cost is written to the ledger as "size lookups", and package sizes are remembered so the same package is not looked up twice.

## Settings

Set them in Hermes Desktop under Settings → Plugins, or with `/datasaver`:

| Setting | Default | Meaning |
|---|---|---|
| `daily_budget_mb` | 500 | Megabytes per day. 0 = off |
| `monthly_budget_mb` | 0 | Megabytes per calendar month. 0 = off |
| `ask_above_mb` | 50 | Any single forecast above this goes to the approval prompt |
| `mode` | `ask` | `ask` → approval prompt. `enforce` → block over-budget downloads. `observe` → meter only |
| `lean_below_pct` | 25 | Trim web results when less than this share of the budget is left. 0 = off |

```
/datasaver                     status
/datasaver budget 300          daily budget, in MB
/datasaver monthly 5000        monthly budget, in MB
/datasaver ask 20              ask before anything over 20 MB
/datasaver mode enforce        ask | enforce | observe
/datasaver pause 60            on Wi-Fi for an hour: stop gating, keep metering
/datasaver resume
/datasaver check <command>     size a command without running it
```

Under cron, the messaging gateway and other unattended runs nobody can answer an approval prompt, so Hermes's approval gate fails closed: an over-budget download does not run while you are asleep.

## Disclosure

- **Network.** Forecasts send read-only metadata requests to `pypi.org`, `api.github.com`, `auth.docker.io`, `registry-1.docker.io`, `huggingface.co` and `registry.ollama.ai`, and one header-only request to a URL that appears in a `curl`/`wget` command. Only the package, repository, image or model name is sent. No credentials, tokens or headers from your command are forwarded. These requests happen before the command reaches Hermes's approval prompt.
- **URL probes.** Before a URL is probed its hostname is resolved, and the probe is sent only if Hermes's `is_safe_url` accepts it and every address it resolves to is public. Redirects are not followed automatically: each hop is checked the same way first, up to four hops. A name that does not resolve, or that points at a private, loopback or link-local address (including the cloud metadata address), is never probed.
- **Storage.** A SQLite ledger in `<HERMES_HOME>/plugin-data/data-saver/data.db`: time, tool name, bytes, and the first 300 characters of the command or URL. It never leaves your machine.
- **No telemetry, no self-update, no shell commands, no background process, no stored credentials.**
- It never auto-approves anything. It only adds approval prompts and blocks; your existing guards are untouched.

## Limits

- The network counters are system-wide. Bytes other apps move while a tool call runs are counted against that call. Close the video stream if you want clean numbers.
- Commands on a remote terminal backend (SSH, Modal, a cloud sandbox) download on that machine, not yours. Forecasts still work; metering only sees this machine.
- A download that Hermes moves to the background is followed until it goes quiet (or for 20 minutes at most). While it runs, all traffic on the machine is counted as that download, including the model requests made meanwhile.
- Forecasts assume nothing is cached. Packages, layers or files you already have are not downloaded again, so the real figure can be lower.
- `git clone --depth 1` is reported with the full repository size and a note that it will be less.

## Tests

```bash
pip install pytest
python -m pytest tests
```

114 tests, no network: parsing, every resolver against recorded registry responses, the budget policy, overlap-safe metering, and a check that `register()` matches `plugin.yaml`.

## License

MIT
