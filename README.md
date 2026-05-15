# NEUROSCAN — Look Inside an AI, Then Try to Break It

> A single container that loads a small AI language model, lets you
> peek inside its "brain" in 3D, then walks you through attacking it,
> defending it, and grading the result — all in your browser, all on
> your local GPU. No prior AI-security background required.

---

## What this demo is, in plain English

AI language models (the kind that power ChatGPT, Copilot, etc.) are
black boxes by default — you give them a prompt, they give you text
back, and what happens in between is a mystery.

NEUROSCAN cracks that box open. It loads a **small, runs-on-your-GPU**
language model (GPT-2 by default — old and tiny, but it works exactly
like the big ones) and turns its internals into something you can
click on:

- **See it think.** Watch thousands of "neurons" light up in 3D as the
  model reads a prompt. Hover over one and see what it actually does.
- **Try to jailbreak it.** Run automated attacks that find weird
  strings of text which trick the model into saying things it
  normally refuses. Watch the attack improve in real time.
- **Find and remove the model's "no" button.** Every safety-trained
  model has an internal "refuse" signal. NEUROSCAN finds it, lets you
  turn it off, and measures the damage.
- **Adjust its personality with sliders.** Make it more honest, more
  formal, funnier — by injecting "concept vectors" the model already
  knows about.
- **Put guardrails in front of it.** Toggle filters for jailbreaks,
  off-topic chat, leaked personal info, unsafe content — then test
  whether they actually catch attacks.
- **Get a report card.** Every test you run feeds into a per-model
  scorecard that ends in a single A–F letter grade. Click "generate
  report" and an external LLM writes you a plain-English summary.

If any of those bullets felt jargon-y — that's fine. The UI walks you
through it, and every tab has a built-in "show me an example" button.

---

## Who this is for

- **Engineers** evaluating an open-source model before shipping it.
- **Security folks** who've heard of "prompt injection" and want to
  see real attacks running, not slides.
- **Researchers** who want a fast workbench for interpretability
  experiments without writing scripts.
- **The curious** — if you've wondered "what's actually happening
  inside ChatGPT?", the small model NEUROSCAN runs works the same way.
  You can poke at it for hours.

You do **not** need a PhD, an ML background, or familiarity with any
of the dozens of papers listed below. The defaults are picked to
"just work," and there's a sample payload behind every button.

---

## What you'll see when you open it

The UI is a single page with **nine tabs** along the top. Here's what
each one is for, in one sentence:

| Tab | What you do here |
|---|---|
| **Dashboard** | See the model's current letter grade and the timeline of every test you've run. |
| **Explore** | Pick a model, type a prompt, and watch its internals — including a rotatable 3D "brain" view. |
| **Generate** | Generate text with a personality slider (honesty / humour / formality / safety / custom). |
| **Red Team** | Attack the model. Manual attacks, automated attacks, and a one-click "throw everything at it" pipeline. |
| **Blue Team** | Turn defenses on and off, then test whether they catch the attacks. |
| **Violet Team** | Combined attacker + defender scenarios — including image-based attacks on vision models. |
| **Understand** | Deep-dive interpretability tools (most jargon-heavy tab — skip on first visit). |
| **Evaluate** | Run standard benchmarks for truthfulness, toxicity, and bias. |
| **History** | Every experiment, exportable as JSON. |

If this is your first time, start at **Explore → 3D Brain**, then go
to **Red Team → Auto Red Team** and click the big button. It'll run
six different attack styles in sequence and produce a report.

---

## A short jargon glossary (for when you click around)

You don't have to read this — but if a label confuses you, here's the
plain-English version.

- **Transformer** — the family of AI models that includes GPT,
  Claude, Gemini. NEUROSCAN works on small open ones.
- **Activations / neurons** — the numbers flowing through the model
  while it processes your prompt. "Neuron lighting up" = that number
  is big right now.
- **Refusal direction** — a specific pattern inside the model that
  fires when it's about to say "I can't help with that." Finding it
  is half the trick; removing it is the other half.
- **Abliteration** — surgically removing that refusal pattern. The
  model stops refusing — but it may also get dumber. NEUROSCAN
  measures both.
- **GCG (Greedy Coordinate Gradient)** — an automated attack that
  finds a gibberish-looking suffix you can add to any prompt to make
  the model comply. The classic "jailbreak as math problem."
- **Red / Blue / Violet team** — attacking / defending /
  attacking-and-defending-at-the-same-time.
- **Linear probe** — a tiny detector trained on the model's
  internals. "Is this prompt about to make the model lie? Probe says
  87% yes." Very fast to train.
- **Circuit / attribution graph** — a map showing which internal
  parts of the model caused which output. Like a flame graph, but for
  thought.
- **MoE (Mixture of Experts)** — modern big models route different
  tokens to different sub-networks. NEUROSCAN visualises that
  routing.
- **KV cache** — the model's short-term memory during generation.
  Surprisingly attackable.
- **Embedding inversion** — embeddings are the "fingerprints" search
  engines store. Inversion = reconstructing the original text from
  the fingerprint. Bad news for RAG pipelines that thought their
  vector stores were private.
- **Sparse autoencoder (SAE)** — a tool that takes a confusing blob
  of activations and decomposes it into human-readable features
  ("this neuron means 'Paris'"). The current hot thing in
  interpretability.
- **TransformerLens / SAELens / repeng / nanoGCG** — the open-source
  libraries doing the heavy lifting. NEUROSCAN is the friendly UI
  over them.

---

## Reference build platform

This demo was built and tested on a **Dell Pro Max GB10** (NVIDIA Grace
Blackwell, **ARM / aarch64** architecture). It will run on standard
x86_64 NVIDIA Linux hosts as well, but the bundled Dockerfile pins
**PyTorch 2.9.1 + CUDA 13.0** wheels because that's the only stable
combination on aarch64 with the GB10's `sm_121` compute capability. On
older GPUs you may need to override `CUDA_ARCH` at build time (see
Configuration below).

---

## Requirements

| Requirement | Minimum | Notes |
|---|---|---|
| OS | Linux | macOS / Windows lack pass-through GPU support — won't work. |
| Docker | 24.x or newer | With Compose **v2** (`docker compose`, not `docker-compose`). |
| GPU | NVIDIA, ≥ 8 GB VRAM | GPT-2 fits in 2 GB; larger models in the UI need ~6 GB. |
| GPU driver | Recent enough for your CUDA version | `nvidia-smi` must work on the host. |
| NVIDIA Container Toolkit | Installed and configured for Docker | Required to expose the GPU to the container. |
| Disk | ~10 GB | Image (~3 GB) + model cache + database volume. |
| RAM | 16 GB recommended | 8 GB will work but may swap during initial build. |
| LLM endpoint | OpenAI-compatible (e.g. host's Ollama) | Used as the **attacker / explainer** model — see below. |
| HF token | Only for gated models | Set `HF_TOKEN` if you switch to Gemma-2 (Google's licence). |

**Quick note on the LLM endpoint:** NEUROSCAN runs a small model
locally on your GPU (that's the one it dissects). It also needs a
separate, *larger* model — typically running in Ollama on the same
machine — to play the role of "attacker" and "report writer." You
configure that endpoint in `.env`.

---

## Installation (step-by-step)

These instructions assume a fresh Linux box. If you already have Docker
+ the NVIDIA Container Toolkit working, skip to step 4.

### 1. Verify your GPU is visible to the host

```bash
nvidia-smi
```

You should see a table with your GPU model, driver version, and CUDA
version. If this command fails, **fix your NVIDIA driver before going
further** — the rest will not work.

### 2. Install Docker Engine + Compose v2

Ubuntu / Debian:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"   # let your user run docker without sudo
newgrp docker                      # apply the new group in this shell
docker compose version             # should print "Docker Compose version v2.x.x"
```

If `docker compose version` reports "command not found", install the
plugin:

```bash
sudo apt install docker-compose-plugin
```

### 3. Install the NVIDIA Container Toolkit

Ubuntu / Debian:

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify it works inside Docker:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

You should see the same `nvidia-smi` table you saw on the host. If
this fails, fix it before continuing.

### 4. Clone the repo

```bash
git clone https://github.com/gleaven/neuroscan.git
cd neuroscan
```

### 5. Create the environment file

```bash
cp .env.example .env
$EDITOR .env
```

The two **required** entries are:

| Variable | Example |
|---|---|
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434/v1` |
| `LLM_MODEL` | `gpt-oss:20b` (or any model already present in your Ollama: `ollama list`) |

The compose file will refuse to start if either is empty. Everything
else has sensible defaults.

### 6. Build and start

```bash
docker compose up -d --build
```

The first build takes **5–15 minutes** (downloads the CUDA base image
~3 GB, installs PyTorch and the rest of the AI stack). Subsequent
starts take ~10 seconds.

### 7. Verify it's healthy

```bash
docker compose ps
# both `demo-neuroscan` and `demo-redis` should show "healthy" within ~2 min

curl -s http://localhost:8080/health
```

Expected output:

```json
{"status": "ok", "service": "neuroscan", "model": "gpt2-small"}
```

The first model load (GPT-2 plus its interpretability "lens") is
downloaded from HuggingFace into a Docker volume; that takes a few
seconds. Switching to a bigger model from the UI later will trigger a
larger download (~5 GB) the first time.

### 8. Open the UI

<http://localhost:8080/>

The UI lands on the **Dashboard** tab. Don't worry that it's empty at
first — it fills up as you run things. Suggested first stops:

1. **Explore → 3D Brain** — type a prompt, watch the model think.
2. **Red Team → Auto Red Team** — push the button, get a report.
3. **Dashboard** — see the model's first letter grade.

### 9. (Optional) Tail the logs

```bash
docker compose logs -f neuroscan
```

You should see the engines initialise in order:

```
NEUROSCAN: loading default model gpt2-small...
Adversarial engine ready
Steering engine ready
Benchmark engine ready
Abliteration engine ready
Optimizer engine ready
Guardrails engine ready
Probe engine ready
FuzzyAI engine ready
Circuit engine ready
KV-cache engine ready
MoE engine ready
Embedding engine ready
VLM engine ready
Auto Red Team pipeline ready
NEUROSCAN ready
```

---

## Configuration

All variables can be set in `.env` or exported in your shell.

| Variable | Default | What it controls |
|---|---|---|
| `OLLAMA_BASE_URL` | _(required)_ | OpenAI-compatible base URL. Use `http://host.docker.internal:11434/v1` for the host's Ollama, or a LAN IP. |
| `LLM_MODEL` | _(required)_ | Model name passed to the attacker / explainer / report writer. Must already be pulled in your Ollama (`ollama pull <name>`). |
| `HF_TOKEN` | _(empty)_ | HuggingFace token. Required only for gated models (e.g. Gemma-2). |
| `APP_PORT` | `8080` | Browser-facing port for the UI and API. |
| `REDIS_HOST_PORT` | `6379` | Where the bundled Redis is exposed on the host. |
| `REDIS_URL` | `redis://demo-redis:6379/14` | Connection string used by NEUROSCAN; override when you BYO Redis. |
| `LANGFUSE_HOST` | _(empty)_ | Optional Langfuse base URL. Empty = observability disabled (no-op). |
| `LANGFUSE_PUBLIC_KEY` | _(empty)_ | Langfuse public key (BYO Langfuse only). |
| `LANGFUSE_SECRET_KEY` | _(empty)_ | Langfuse secret key (BYO Langfuse only). |
| `DEMO_HOSTNAME` | `localhost` | Hostname Caddy serves under (proxy profile only). |
| `HTTP_PORT` | `8081` | Caddy HTTP port (proxy profile only). |
| `HTTPS_PORT` | `8443` | Caddy HTTPS port (proxy profile only). |

### Build-time arguments

If you're not on a GB10 / `sm_121` GPU, override the CUDA arch when
building:

```bash
docker compose build --build-arg CUDA_ARCH=8.6   # e.g. RTX 3090
docker compose up -d
```

Common values: `8.0` (A100), `8.6` (RTX 30xx), `8.9` (RTX 40xx),
`9.0` (H100), `12.0`/`12.1` (Grace Blackwell / GB10).

---

## External services (BYO)

If you'd rather use your own Redis (e.g. a managed instance), uncomment
`REDIS_URL` in `.env` and start with the BYO override:

```bash
docker compose -f docker-compose.yml -f docker-compose.byo.yml up -d
```

`docker-compose.byo.yml` removes the bundled `redis` service so only
the `neuroscan` container runs locally.

| Variable | Example |
|---|---|
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434/v1` |
| `REDIS_URL` | `redis://redis.example.com:6379/14` |
| `LANGFUSE_HOST` | `http://my-langfuse:3000` |
| `LANGFUSE_PUBLIC_KEY` | `pk-lf-…` |
| `LANGFUSE_SECRET_KEY` | `sk-lf-…` |

Redis stores the per-model scorecard, the experiment timeline, and
cached results so you don't recompute. The demo will still load and
run without Redis — it just logs warnings and skips persistence.

The OpenAI-compatible LLM endpoint is always external (NEUROSCAN does
not bundle Ollama). Any endpoint that speaks `/v1/chat/completions`
works: Ollama, vLLM, LM Studio, an OpenAI proxy, etc.

---

## Optional HTTPS reverse proxy

Caddy is bundled as an opt-in profile. It auto-provisions Let's
Encrypt certs when `DEMO_HOSTNAME` is a real DNS name pointing at this
host:

```bash
DEMO_HOSTNAME=neuroscan.example.com docker compose --profile proxy up -d
```

For local testing keep `DEMO_HOSTNAME=localhost` and Caddy will issue
a self-signed cert.

---

## Authentication

NEUROSCAN runs **without authentication** by default. For shared
deployments, put one of these in front of it:

- **Caddy basic auth** — add a `basic_auth` block to the Caddyfile.
- **oauth2-proxy in front of Caddy** — for SSO-style auth.
- **Cloudflare Tunnel + Access policies** — easiest if you're already
  on Cloudflare.

The demo intentionally exposes every red-team capability without
gating. **Do not put it on the public internet unauthenticated.** It
will happily teach a stranger how to jailbreak a model.

---

## Under the hood (for the curious)

If you don't care how it's built, skip this section — the UI is the
product. But for anyone who wants to extend it:

NEUROSCAN is a single FastAPI server (`server.py`, ~4100 lines) that
wires together ~14 specialised "engines." Each engine owns one
capability and streams progress over WebSockets so the UI updates
live.

| Engine | What it does (in one line) |
|---|---|
| `activation_engine.py` | Loads the model, captures activations, drives the 3D viewer and feature decomposition. |
| `abliteration_engine.py` | Finds and removes the model's "refusal" pattern; exports the modified weights. |
| `adversarial_engine.py` | Runs GCG jailbreak attacks one step at a time so the UI can show the loss curve. |
| `steering_engine.py` | The "personality sliders" — concept vectors for honesty, humour, etc. |
| `benchmark_engine.py` | Standard truthfulness / toxicity / bias tests, plus a built-in security probe suite. |
| `optimizer_engine.py` | Optuna search over abliteration parameters — finds the best trade-off automatically. |
| `guardrails_engine.py` | The Blue Team filters (jailbreak / topic / PII / content / I-O moderation). |
| `probe_engine.py` | Trains tiny "mind-reading" detectors on the model's hidden states. |
| `fuzzyai_engine.py` | LLM-driven attacks: PAIR, Crescendo, Best-of-N, ActorAttack, Genetic. |
| `circuit_engine.py` | Builds an attribution graph showing what inside the model caused the output. |
| `moe_engine.py` | Mixture-of-Experts routing analysis (simulated for dense models). |
| `embedding_engine.py` | Embedding inversion attacks + a 3D embedding-space view. |
| `vlm_engine.py` | Sends sneaky images (typography / LSB / EXIF hidden prompts) to a vision model. |
| `kvcache_engine.py` | Pokes at the model's short-term memory: sinks, prefix poisoning, drift, exhaustion. |
| `redteam_suite.py` | Native re-implementations of Garak / DeepTeam / PyRIT / Promptfoo / promptmap2 probes. |
| `auto_redteam.py` | The "throw everything at it" pipeline that chains six attack stages. |

The frontend is a single page (`static/index.html`) with Three.js
viewers for the 3D and graph views. ~120 REST endpoints plus a single
WebSocket channel (`/ws/activations`) — discover them via DevTools or
the OpenAPI spec at `/docs`.

---

## Troubleshooting

- **Compose refuses to start with `set OLLAMA_BASE_URL …` / `set
  LLM_MODEL …`** — both are required. Edit `.env` and set them to
  your Ollama (or other OpenAI-compatible) endpoint and a model name
  already pulled there.
- **`OLLAMA_BASE_URL` unreachable from inside the container** — from
  the container, run `curl ${OLLAMA_BASE_URL%/v1}/api/tags`. On Linux,
  `host.docker.internal` requires either an `extra_hosts` entry on the
  service or a LAN IP. Switching to `http://<LAN-IP>:11434/v1` is the
  most robust fix.
- **`nvidia-smi` works on host but not in container** — the NVIDIA
  Container Toolkit isn't wired into Docker. Run `sudo nvidia-ctk
  runtime configure --runtime=docker && sudo systemctl restart docker`
  and try the test container in step 3 again.
- **Gemma-2 download fails 401/403** — accept the licence on
  huggingface.co for `google/gemma-2-2b`, then set `HF_TOKEN` in
  `.env` and `docker compose restart neuroscan`.
- **Out of memory loading a larger model** — stay on GPT-2 in the UI,
  or run on a host with more VRAM. Switching models is reversible.
- **`unsupported gpu architecture` during PyTorch import** — your
  GPU's compute capability isn't in the wheel's arch list. Rebuild
  with `--build-arg CUDA_ARCH=<your arch>` (see Configuration).
- **Container restarts in a loop** — almost always a GPU/driver
  mismatch. Confirm `docker run --rm --gpus all
  nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi` works from your
  shell.
- **Port collision on 8080 or 6379** — change `APP_PORT` and/or
  `REDIS_HOST_PORT` in `.env`.
- **Dashboard loses history after `docker compose down -v`** — `-v`
  removes the named Redis volume. Drop the `-v` to keep it.
- **`demo-neuroscan` health check failing on first boot** — give it
  longer; the first model download can push past the 120 s
  start-period. Check `docker compose logs neuroscan` for stack
  traces.

---

## FAQ

**Q: I don't know anything about AI security. Will I understand this?**
Yes. Open the UI, click around the Red Team tab, push the "Auto Red
Team" button, and read the report it generates. The terminology will
start to click within ~15 minutes. The glossary above covers the
words you'll see most often.

**Q: Can I use a CPU?** No. The interpretability and attack
techniques all need a GPU; the demo will refuse to start without one.

**Q: Do the AI models actually run on my GPU, or do they call out to
the internet?** The model being studied (GPT-2, Pythia, Gemma-2) runs
**locally on your GPU** — that's the whole point. The Ollama endpoint
you configure is also local (just on the host instead of in the
container) and only plays the *attacker* and *report-writer* roles.

**Q: Is anything sent to the internet?** Only the initial model
downloads from HuggingFace (cached forever in a Docker volume). All
inference, attacks, and analyses run locally.

**Q: Can I add my own model?** Yes — extend `MODEL_REGISTRY` in
`activation_engine.py` with a TransformerLens-compatible model name,
layer count, hidden size, and a matching SAE release.

**Q: I just want one number that tells me if my model is secure.**
Run **Auto Red Team** from the Red Team tab. It chains six attack
stages and writes a single composite score to the Dashboard, with an
LLM-written narrative report. It is *not* a substitute for a real
security review — but it's a great starting point.

**Q: Why GPT-2? Isn't that ancient?** It is — and that's the point.
GPT-2 is small enough to fit on any GPU, but it's the same family of
model as the giants. Every technique here works on bigger models too;
GPT-2 just lets you iterate fast.

---

## Credits

Built by Andrew Meinecke.
