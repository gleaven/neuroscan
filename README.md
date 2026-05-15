# NEUROSCAN — AI Model Security & Interpretability Workbench

> A single container that loads a transformer, dissects its activations
> in 3D, attacks it with state-of-the-art jailbreak techniques, defends
> it with programmable guardrails, and grades the result — all in your
> browser, all on your local GPU.

---

## What this demo is

NEUROSCAN is an end-to-end workbench for **mechanistic interpretability
and AI red/blue/violet teaming**. It loads a small transformer
(GPT-2 Small by default; Pythia-70M and Gemma-2 2B available) into
**TransformerLens**, hooks every layer for live activation capture, and
exposes a tabbed UI that lets you:

- Watch thousands of neurons light up in 3D as the model processes a
  prompt, then decompose those activations into human-interpretable
  features with **Sparse Autoencoders (SAELens)**.
- Run **GCG (Greedy Coordinate Gradient)** adversarial-suffix
  optimisation step-by-step, watch the loss descend in real time, and
  see whether the discovered suffix actually breaks the model.
- Discover the **refusal direction** in residual space, project it out
  ("abliteration"), and measure both the bypass rate and the KL
  divergence cost — then let an Optuna optimiser sweep configurations
  and return a Pareto front.
- Steer generation with **representation-engineering vectors**
  (honesty, humour, formality, …) via simple sliders.
- Run a stack of red-team frameworks natively (GCG, FuzzyAI PAIR /
  Crescendo / Best-of-N / ActorAttack / Genetic, plus Garak-,
  DeepTeam-, PyRIT-, promptmap2-style probes) and a blue-team layer
  (NeMo-Guardrails-style rails for jailbreak / topic / PII / content
  safety).
- Trace **circuits** — attribution graphs from input tokens to output
  logits — train **linear probes** for hidden properties, analyse
  **MoE routing**, demonstrate **embedding inversion** privacy
  attacks, and probe **KV-cache** vulnerabilities (sink detection,
  prefix poisoning, quantization drift, cache exhaustion).
- Test **vision-language models** with steganographic prompt-injection
  images (typography, LSB, EXIF metadata).
- Run alignment benchmarks (TruthfulQA, toxicity via Detoxify, CrowS
  bias) and aggregate everything into a per-model **dashboard with
  letter grades**, persisted in Redis.

There is one **persistent backend** (FastAPI + WebSockets) and one
**single-page UI** (`static/index.html`) organised into eight tabs:
**Dashboard, Explore, Generate, Red Team, Blue Team, Violet Team,
Understand, Evaluate, History**. An external **OpenAI-compatible LLM
endpoint** (typically Ollama on the host) is used for the explainer
narratives, FuzzyAI attacker model, and PDF-style report generation.

---

## The engines

NEUROSCAN is organised as a thin FastAPI shell (`server.py`) on top of
~14 specialised engines. Each engine owns one capability domain and
streams progress over WebSockets so the UI updates live.

| Engine | Lines | What it does |
|---|---|---|
| `activation_engine.py` | 1451 | TransformerLens model manager + activation extraction. Owns the model registry (GPT-2 Small, Pythia-70M, Gemma-2 2B), runs forward passes with full caching, drives the 3D activation viewer, attention-head ablation, neuron diff scans, residual-stream geometry, activation patching, and SAELens sparse-autoencoder feature decomposition. |
| `abliteration_engine.py` | 1354 | Refusal-direction discovery and removal. Implements standard, norm-preserving, and biprojected abliteration; multi-layer extraction; per-layer SNR and quality metrics; KL-divergence capability scoring; 33-marker refusal detection; HuggingFace dataset loading; permanent weight orthogonalisation; export of directions and abliterated weights. |
| `adversarial_engine.py` | 299 | nanoGCG-based **Greedy Coordinate Gradient** suffix optimisation, run step-by-step (not via `nanogcg.run()`) so each step's loss + best suffix can be streamed to the browser. Supports pause / resume / stop. |
| `steering_engine.py` | 282 | **Representation engineering** via `repeng`. Pre-defined contrastive prompt pairs for honesty, humour, formality, safety, etc.; extracts concept vectors; injects them at generation time with adjustable strength. |
| `benchmark_engine.py` | 828 | Alignment benchmarks (TruthfulQA, Detoxify toxicity, CrowS bias) and the built-in security probe suite — 7 categories (jailbreak, injection, exfiltration, system_prompt, encoding_attacks, multi_turn, toxicity), ~54 probes total. |
| `optimizer_engine.py` | 342 | **Optuna TPE multi-objective optimiser** for abliteration parameters. Dual objective: minimise post-abliteration refusal rate **and** minimise KL divergence. Returns a Pareto front (inspired by Heretic). |
| `guardrails_engine.py` | 471 | Programmable safety rails — jailbreak, topic, PII, content-safety, input/output moderation. Uses NeMo Guardrails when installed, otherwise falls back to a regex layer that still demonstrates the defensive flow. |
| `probe_engine.py` | 325 | **Linear probes on activations** (Anthropic "Simple probes can catch sleeper agents"). Built-in datasets for refusal intent, truthfulness, toxicity; user-defined concept training from positive/negative prompt pairs. |
| `fuzzyai_engine.py` | 789 | LLM-assisted jailbreak techniques inspired by CyberArk FuzzyAI: **PAIR** (iterative refinement by attacker LLM), **Crescendo** (multi-turn escalation), **Best-of-N**, **ActorAttack** (semantic network), **Genetic** (GA crossover/mutation). All attacker calls go through the configured OpenAI-compatible endpoint. |
| `circuit_engine.py` | 340 | Attribution-graph generation — nodes for attention heads, MLP neurons, residual positions; edges for causal influence with attribution scores. Falls back to attention-based attribution when `circuit-tracer` is unavailable. |
| `moe_engine.py` | 184 | Mixture-of-Experts routing analysis — per-layer / per-token expert assignments, router probabilities, expert specialisation, ablation. Simulates routing on dense models. |
| `embedding_engine.py` | 214 | Embedding security: **inversion attacks** (partial text reconstruction from embedding vectors, demonstrating RAG-pipeline leakage) and dimensionality-reduced (PaCMAP) embedding-space visualisation. |
| `vlm_engine.py` | 305 | Vision-language model testing. Generates **steganographic injection images** (typography / LSB / EXIF) and submits them to an external VLM endpoint to measure visual prompt-injection susceptibility. |
| `kvcache_engine.py` | 325 | KV-cache forensics — attention-sink detection, cache-influence scoring, **prefix poisoning** simulation, **quantization drift**, **cache exhaustion** projection, cross-turn drift. The cache is treated as an unsigned attack surface. |

Two orchestration layers ride on top:

| Module | Purpose |
|---|---|
| `redteam_suite.py` | Native re-implementations of probes from **Garak** (hallucination, data leakage, misinformation), **DeepTeam** (RAG, LLM-as-judge), **PyRIT** (multi-turn orchestration with converters), **Promptfoo** (OWASP LLM Top 10), and **promptmap2** (system-prompt extraction). No external tool installs required. |
| `auto_redteam.py` | **Automated end-to-end pipeline** — chains Security Scan → Fuzz Mutations → Red Team Suite → GCG → FuzzyAI PAIR → Abliteration Test, weights each stage, streams per-stage progress, and produces a single composite vulnerability report. |

`server.py` itself (~4100 lines) wires these engines together, exposes
~120 REST endpoints + a WebSocket channel, and maintains a per-model
**ModelDashboard** (Redis-backed) that converts every test result into
domain scores (input safety, output safety, attack resistance,
capability, alignment depth, defense coverage, interpretability
coverage) and a final A–F letter grade.

---

## Capabilities (at a glance)

- One transformer loaded into TransformerLens, hooked at every layer,
  with live SAELens sparse-feature decomposition.
- 3D activation visualisation (Three.js) plus heatmap, attention,
  logit-lens, brain-view, thought-map, knowledge-graph, animated, and
  KV-cache views.
- GCG adversarial-suffix optimisation with live loss curve,
  pause/resume/stop.
- Refusal-direction abliteration with three algorithms, Optuna-driven
  Pareto-front parameter search, batch testing, perm-weight
  orthogonalisation, direction export.
- Representation-engineering steering across pre-defined and custom
  concepts.
- Native red-team stack: GCG, FuzzyAI (PAIR / Crescendo / Best-of-N /
  ActorAttack / Genetic), Garak / DeepTeam / PyRIT / Promptfoo /
  promptmap2 probes.
- NeMo-Guardrails-style blue-team rails (jailbreak / topic / PII /
  content safety / I/O moderation).
- Linear activation probes for refusal intent, truthfulness, toxicity,
  and user-defined concepts.
- Circuit tracing, MoE routing analysis, embedding inversion, KV-cache
  attack surface.
- VLM injection via typography / LSB / EXIF steganography.
- Alignment benchmarks: TruthfulQA, Detoxify toxicity, CrowS bias.
- One-click **Auto Red Team** pipeline producing a composite report.
- Per-model dashboard with letter grade, persisted timeline, and
  LLM-generated narrative reports.
- Bundled Redis for dashboard / experiment / cached-direction
  persistence; optional Caddy reverse proxy with HTTPS.

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
| GPU | NVIDIA, ≥ 8 GB VRAM | GPT-2 Small fits in 2 GB; Gemma-2 2B + SAEs needs ~6 GB. |
| GPU driver | Recent enough for your CUDA version | `nvidia-smi` must work on the host. |
| NVIDIA Container Toolkit | Installed and configured for Docker | Required to expose the GPU to the container. |
| Disk | ~10 GB | Image (~3 GB) + HuggingFace cache (Gemma-2 ≈ 5 GB) + Redis volume. |
| RAM | 16 GB recommended | 8 GB will work but may swap during initial build. |
| LLM endpoint | OpenAI-compatible (e.g. host's Ollama) | Required for explainer, FuzzyAI attacker, and report generation. |
| HF token | Only for gated models | Set `HF_TOKEN` if you switch to Gemma-2 2B (licence acceptance). |

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
~3 GB, installs PyTorch + TransformerLens + SAELens + the long
dependency tree). Subsequent starts take ~10 seconds.

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

The first model load (GPT-2 Small + its SAE release) is downloaded
from HuggingFace into the `neuroscan-model-cache` volume; that takes a
few seconds. Switching to Gemma-2 2B from the UI later will trigger a
larger download (~5 GB) the first time.

### 8. Open the UI

<http://localhost:8080/>

The UI lands on the **Dashboard** tab. Switch a model in the top bar
(GPT-2 Small / Pythia-70M / Gemma-2 2B), then explore Red Team / Blue
Team / Violet Team / Understand / Evaluate. The 3D activation viewer
lives under **Explore → 3D Brain**.

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
| `OLLAMA_BASE_URL` | _(required)_ | OpenAI-compatible base URL. Mapped internally to `LITELLM_BASE_URL` and `OPENAI_BASE_URL`. Use `http://host.docker.internal:11434/v1` for the host's Ollama, or a LAN IP. |
| `LLM_MODEL` | _(required)_ | Model name passed to the explainer / FuzzyAI attacker / report generator. Must already be pulled in your Ollama (`ollama pull <name>`). |
| `HF_TOKEN` | _(empty)_ | HuggingFace token. Required only for gated probe models (Gemma-2 2B). |
| `APP_PORT` | `8080` | Browser-facing port for the UI and REST/WebSocket API. |
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

## Live controls (in the browser)

The UI is tabbed; the major surfaces are:

- **Dashboard** — per-model letter grade and timeline. Each completed
  test (security scan, GCG, FuzzyAI, abliteration, benchmark, probe,
  guardrails check, interpretability tool) writes a timeline entry,
  contributes to a domain score, and can be turned into an
  LLM-narrated report on demand.
- **Explore** — model selector (GPT-2 Small / Pythia-70M / Gemma-2
  2B), prompt input, and nine activation visualisations: executive
  summary, technical, heatmap, attention, logit-lens, **3D brain**,
  thought map, knowledge graph, animated, KV-cache. SAE feature
  decomposition is a click away from any neuron.
- **Generate** — sampling with steering. Pick a concept vector
  (honesty / humour / formality / safety / custom), set strength, and
  compare baseline vs. steered output side-by-side.
- **Red Team** — start/stop GCG with live loss curve; run FuzzyAI
  techniques (PAIR / Crescendo / Best-of-N / ActorAttack / Genetic);
  run the Garak/DeepTeam/PyRIT/Promptfoo/promptmap2 probe suite; fire
  the **Auto Red Team** pipeline.
- **Blue Team** — toggle each guardrail on/off, send test prompts,
  see which rail fired, compare guarded vs. unguarded responses.
- **Violet Team** — combined offense+defense scenarios:
  persuasion-resistance, social-engineering, trust-claim scoring,
  sycophancy detection, vision-language injection.
- **Understand** — interpretability deep dives: activation patching,
  neuron diff scans, SAE decomposition, logit lens, **circuit
  tracing**, linear-probe training, abliteration workbench (compute
  direction, batch test, strength sweep, Optuna optimisation, cache
  management), MoE routing, embedding inversion.
- **Evaluate** — TruthfulQA / Detoxify toxicity / CrowS bias;
  OWASP LLM Top 10 compliance.
- **History** — Redis-backed experiment log, exportable as JSON.

All controls are also exposed as REST + WebSocket endpoints (~120
endpoints under `/api/*` and a single `/ws/activations` channel for
streaming). Open the network panel in DevTools to discover them.

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

Redis stores the per-model dashboard, the experiment timeline, and
cached abliteration directions. The demo will still load and run
without Redis — it just logs warnings and skips persistence.

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
gating; do not put it on the public internet unauthenticated.

---

## Architecture (file map)

| File | Purpose |
|---|---|
| `server.py` | FastAPI app, ~120 REST endpoints + `/ws/activations`. Lifespan boots all engines, ModelDashboard, ExperimentTracker. ~4100 lines. |
| `activation_engine.py` | Model registry + TransformerLens manager. Owns hooks, SAELens decomposition, geometry, patching, head ablation, diff scans. |
| `abliteration_engine.py` | Refusal-direction discovery + removal (standard / norm-preserving / biprojected), batch testing, perplexity, weight orthogonalisation, export. |
| `optimizer_engine.py` | Optuna multi-objective optimiser over abliteration parameters. |
| `adversarial_engine.py` | Step-by-step nanoGCG loop with live progress streaming. |
| `steering_engine.py` | repeng-based concept-vector steering. |
| `benchmark_engine.py` | TruthfulQA, Detoxify toxicity, CrowS bias, built-in 7-category security probes. |
| `guardrails_engine.py` | NeMo-Guardrails-style rails with regex fallback. |
| `probe_engine.py` | Linear probes on hidden states. |
| `fuzzyai_engine.py` | LLM-assisted attacks: PAIR, Crescendo, Best-of-N, ActorAttack, Genetic. |
| `circuit_engine.py` | Attribution-graph generation (with attention-based fallback). |
| `kvcache_engine.py` | KV-cache forensics + attack simulations. |
| `moe_engine.py` | Mixture-of-Experts routing analysis (simulated for dense models). |
| `embedding_engine.py` | Embedding inversion + PaCMAP visualisation. |
| `vlm_engine.py` | Steganographic VLM injection (typography / LSB / EXIF). |
| `redteam_suite.py` | Native Garak / DeepTeam / PyRIT / Promptfoo / promptmap2 probes. |
| `auto_redteam.py` | Six-stage automated red-team pipeline orchestrator. |
| `static/index.html` | Single-page UI (~290 KB). Eight tabs, nine activation views. |
| `static/js/` | `app.js`, `neural-viz.js`, `viz-views.js` — Three.js viewers + tab logic. |
| `static/css/neuroscan.css` | UI styling. |
| `demo_samples/` | Pre-canned JSON payloads served by `/api/demo/{feature}` so each tab has an instant "show me an example" button. |
| `Dockerfile` | CUDA 13.0 base, PyTorch 2.9.1, all Python deps. |
| `docker-compose.yml` | NEUROSCAN + Redis + (opt-in) Caddy. |
| `docker-compose.byo.yml` | Strips out the bundled Redis when you BYO. |
| `Caddyfile` | Reverse-proxy config. |
| `requirements.txt` | FastAPI, transformer-lens, sae-lens, nanogcg, repeng, detoxify, datasets, optuna, pacmap, redis. |

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
- **Out of memory loading Gemma-2** — stay on GPT-2 Small in the UI,
  or run on a host with more VRAM. Switching models is reversible.
- **`unsupported gpu architecture` during PyTorch import** — your GPU's
  compute capability isn't in the wheel's arch list. Rebuild with
  `--build-arg CUDA_ARCH=<your arch>` (see Configuration).
- **Container restarts in a loop** — almost always a GPU/driver
  mismatch. Confirm `docker run --rm --gpus all
  nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi` works from your
  shell.
- **Port collision on 8080 or 6379** — change `APP_PORT` and/or
  `REDIS_HOST_PORT` in `.env`.
- **Dashboard loses history after `docker compose down -v`** — `-v`
  removes the named Redis volume. Drop the `-v` to keep it.
- **`demo-neuroscan` health check failing on first boot** — give it
  longer; the first model + SAE download can push past the 120 s
  start-period. Check `docker compose logs neuroscan` for stack
  traces.

---

## FAQ

**Q: Can I use a CPU?** No. TransformerLens activation extraction +
SAELens decomposition + GCG gradients all assume CUDA; the demo will
refuse to start without a visible NVIDIA GPU.

**Q: Do the models actually run on my GPU, or do they call Ollama?**
The probe model (GPT-2 Small / Pythia-70M / Gemma-2 2B) runs locally
on your GPU via TransformerLens — that's what every interpretability
and abliteration feature operates on. The **Ollama endpoint** is only
used as the *attacker* / *explainer* / *report-writer* LLM (FuzzyAI
attacker model, dashboard report narratives, executive summaries).

**Q: Is anything sent to the internet?** Only the initial HuggingFace
model + SAE downloads (cached in a Docker volume). All inference,
attacks, and analyses run locally. The Ollama endpoint can also be on
the same host.

**Q: Can I add my own model?** Yes — extend `MODEL_REGISTRY` in
`activation_engine.py` with the TransformerLens model name, layer
count, hidden size, and matching SAE release.

**Q: How do I get a one-shot vulnerability report?** Run **Auto Red
Team** from the Red Team tab. It chains six stages and posts a
weighted summary back to the Dashboard with an LLM-narrated report.

---

## Credits

Built by Andrew Meinecke.
