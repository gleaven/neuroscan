# NEUROSCAN — AI Model Security & Interpretability Workbench

> A single container that loads a small language model, exposes its
> internals in a tabbed UI, runs a full stack of attacks and defenses
> against it, and produces a per-model security report. Browser
> frontend, local NVIDIA GPU backend.

---

## What this demo does

NEUROSCAN loads a small transformer language model (GPT-2 Small by
default; Pythia-70M and Gemma-2 2B also available) onto your GPU,
hooks every layer for live activation capture, and exposes a tabbed
UI for:

- **Activation inspection.** A 3D viewer over every neuron in every
  layer, plus heatmap, attention, logit-lens, KV-cache, and graph
  views. Sparse-autoencoder decomposition exposes human-readable
  features behind each activation.
- **Adversarial attacks.** GCG (Greedy Coordinate Gradient) suffix
  optimisation streamed step-by-step. LLM-driven attacks via the
  FuzzyAI techniques (PAIR, Crescendo, Best-of-N, ActorAttack,
  Genetic). Native probes from Garak, DeepTeam, PyRIT, Promptfoo,
  and promptmap2.
- **Refusal-direction abliteration.** Discover, project out, and
  optionally write-back the model's refusal direction. Three
  algorithms; Optuna multi-objective search returns a Pareto front
  trading bypass rate against KL divergence.
- **Representation engineering.** Concept-vector steering across
  honesty, humour, formality, safety, and user-defined concepts.
- **Defenses.** Programmable guardrails for jailbreak / topic / PII /
  content / I-O moderation. NeMo Guardrails when available, regex
  fallback otherwise.
- **Interpretability.** Linear probes on activations, attribution-graph
  circuit tracing, MoE routing analysis, embedding inversion, and
  KV-cache attack-surface analysis (sink detection, prefix poisoning,
  quantization drift, exhaustion).
- **Vision-language injection.** Steganographic prompts hidden in
  images via typography, LSB, and EXIF metadata, submitted to an
  external VLM.
- **Benchmarks.** TruthfulQA, Detoxify toxicity, CrowS bias, plus a
  built-in security probe suite covering seven categories.
- **Scoring.** Every test feeds a per-model dashboard scored across
  eight domains (input safety, output safety, attack resistance,
  capability, alignment depth, defense coverage, interpretability
  coverage, plus a composite) ending in an A–F letter grade. Optional
  LLM-narrated reports on demand.

Backend: FastAPI + WebSockets, ~120 REST endpoints. Frontend: a single
HTML page with nine tabs and Three.js viewers. Persistence: Redis.

---

## Audience

NEUROSCAN is intended for engineers evaluating an open-source model
before deployment, security teams running red-team assessments, and
researchers running interpretability experiments. The UI is
self-guiding: every tab includes a sample payload, and the defaults
produce results without configuration. Familiarity with transformer
internals helps but is not required.

---

## UI tabs

| Tab | Purpose |
|---|---|
| Dashboard | Per-model letter grade and experiment timeline. |
| Explore | Model selector, prompt input, and nine activation views including the 3D brain. |
| Generate | Sampling with concept-vector steering; side-by-side baseline vs. steered output. |
| Red Team | GCG, FuzzyAI techniques, Garak/DeepTeam/PyRIT/Promptfoo/promptmap2 probes, Auto Red Team pipeline. |
| Blue Team | Per-rail toggles for guardrails; guarded vs. unguarded comparisons. |
| Violet Team | Combined offense+defense scenarios including VLM injection. |
| Understand | Activation patching, neuron diff, SAE decomposition, logit lens, circuit tracing, probe training, abliteration workbench, MoE routing, embedding inversion. |
| Evaluate | TruthfulQA, Detoxify toxicity, CrowS bias, OWASP LLM Top 10 compliance. |
| History | Experiment log, exportable as JSON. |

---

## Glossary

- **Activations / neurons** — the numeric values produced at each
  layer as the model processes a prompt.
- **Transformer** — the model architecture used by GPT, Claude,
  Gemini, etc. NEUROSCAN operates on small open variants.
- **Refusal direction** — a direction in residual-stream space whose
  activation correlates with the model refusing a request.
- **Abliteration** — projecting out (or permanently writing out) the
  refusal direction to suppress refusal behaviour. Measured against
  KL divergence to detect capability loss.
- **GCG (Greedy Coordinate Gradient)** — gradient-based search for an
  adversarial suffix that, appended to a prompt, induces a target
  completion.
- **Red / Blue / Violet team** — offensive testing / defensive
  controls / combined offense-and-defense scenarios.
- **Linear probe** — a small classifier trained on hidden states to
  detect a property (refusal intent, truthfulness, toxicity, etc.).
- **Circuit / attribution graph** — a graph of attention heads, MLP
  neurons, and residual positions with edges weighted by causal
  influence on a chosen output.
- **MoE (Mixture of Experts)** — per-token expert routing in modern
  large models. NEUROSCAN visualises routing (simulated for dense
  models).
- **KV cache** — the keys and values cached during generation. Treated
  here as an attack surface.
- **Embedding inversion** — partial text reconstruction from a stored
  embedding vector; relevant to RAG-pipeline data leakage.
- **Sparse autoencoder (SAE)** — decomposes activations into a larger
  set of sparse, individually interpretable features.
- **TransformerLens / SAELens / repeng / nanoGCG** — upstream
  libraries used for activation capture, SAE decomposition, steering,
  and GCG respectively.

---

## Reference build platform

This demo was built and tested on a **Dell Pro Max GB10** (NVIDIA Grace
Blackwell, **ARM / aarch64** architecture). It will run on standard
x86_64 NVIDIA Linux hosts as well, but the bundled Dockerfile pins
**PyTorch 2.9.1 + CUDA 13.0** wheels because that's the only stable
combination on aarch64 with the GB10's `sm_121` compute capability. On
older GPUs override `CUDA_ARCH` at build time (see Configuration).

---

## Requirements

| Requirement | Minimum | Notes |
|---|---|---|
| OS | Linux | macOS / Windows lack pass-through GPU support. |
| Docker | 24.x or newer | With Compose v2 (`docker compose`). |
| GPU | NVIDIA, ≥ 8 GB VRAM | GPT-2 Small fits in 2 GB; Gemma-2 2B + SAEs needs ~6 GB. |
| GPU driver | Recent enough for your CUDA version | `nvidia-smi` must work on the host. |
| NVIDIA Container Toolkit | Installed and configured for Docker | Required to expose the GPU to the container. |
| Disk | ~10 GB | Image (~3 GB) + HuggingFace cache + Redis volume. |
| RAM | 16 GB recommended | 8 GB will work but may swap during initial build. |
| LLM endpoint | OpenAI-compatible (e.g. host's Ollama) | Used as attacker / explainer / report generator. |
| HF token | Only for gated models | Set `HF_TOKEN` for Gemma-2 (licence acceptance required). |

The probe model (GPT-2 / Pythia / Gemma-2) runs locally on your GPU.
The external LLM endpoint is a separate, typically larger model used
only for attacker roleplay, explainer narratives, and report
generation.

---

## Installation

These instructions assume a fresh Linux host. If Docker and the NVIDIA
Container Toolkit are already configured, skip to step 4.

### 1. Verify GPU visibility

```bash
nvidia-smi
```

If this fails, fix the NVIDIA driver before continuing.

### 2. Install Docker Engine + Compose v2

Ubuntu / Debian:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
newgrp docker
docker compose version
```

If `docker compose version` reports "command not found":

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

Verify GPU access inside Docker:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

### 4. Clone the repo

```bash
git clone https://github.com/gleaven/neuroscan.git
cd neuroscan
```

### 5. Configure `.env`

```bash
cp .env.example .env
$EDITOR .env
```

Required entries:

| Variable | Example |
|---|---|
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434/v1` |
| `LLM_MODEL` | `gpt-oss:20b` (or any model already pulled in your Ollama) |

Compose refuses to start if either is empty.

### 6. Build and start

```bash
docker compose up -d --build
```

First build: 5–15 minutes. Subsequent starts: ~10 seconds.

### 7. Verify health

```bash
docker compose ps
curl -s http://localhost:8080/health
```

Expected:

```json
{"status": "ok", "service": "neuroscan", "model": "gpt2-small"}
```

### 8. Open the UI

<http://localhost:8080/>

The UI lands on Dashboard. Suggested starting path: Explore → 3D
Brain, then Red Team → Auto Red Team.

### 9. (Optional) Tail logs

```bash
docker compose logs -f neuroscan
```

Engines initialise in order:

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

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | _(required)_ | OpenAI-compatible base URL. Use `http://host.docker.internal:11434/v1` for the host's Ollama, or a LAN IP. |
| `LLM_MODEL` | _(required)_ | Model name passed to the attacker / explainer / report generator. Must already be pulled in your Ollama. |
| `HF_TOKEN` | _(empty)_ | HuggingFace token, required only for gated models (Gemma-2). |
| `APP_PORT` | `8080` | UI and REST/WebSocket port. |
| `REDIS_HOST_PORT` | `6379` | Bundled Redis host port. |
| `REDIS_URL` | `redis://demo-redis:6379/14` | Connection string; override for BYO Redis. |
| `LANGFUSE_HOST` | _(empty)_ | Optional Langfuse base URL. Empty = observability disabled. |
| `LANGFUSE_PUBLIC_KEY` | _(empty)_ | Langfuse public key. |
| `LANGFUSE_SECRET_KEY` | _(empty)_ | Langfuse secret key. |
| `DEMO_HOSTNAME` | `localhost` | Caddy server name (proxy profile only). |
| `HTTP_PORT` | `8081` | Caddy HTTP port (proxy profile only). |
| `HTTPS_PORT` | `8443` | Caddy HTTPS port (proxy profile only). |

### Build-time arguments

Override the CUDA arch on non-GB10 GPUs:

```bash
docker compose build --build-arg CUDA_ARCH=8.6
docker compose up -d
```

Common values: `8.0` (A100), `8.6` (RTX 30xx), `8.9` (RTX 40xx),
`9.0` (H100), `12.0`/`12.1` (Grace Blackwell / GB10).

---

## External services (BYO)

To use an external Redis, uncomment `REDIS_URL` in `.env` and start
with the BYO override:

```bash
docker compose -f docker-compose.yml -f docker-compose.byo.yml up -d
```

`docker-compose.byo.yml` removes the bundled Redis service.

| Variable | Example |
|---|---|
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434/v1` |
| `REDIS_URL` | `redis://redis.example.com:6379/14` |
| `LANGFUSE_HOST` | `http://my-langfuse:3000` |
| `LANGFUSE_PUBLIC_KEY` | `pk-lf-…` |
| `LANGFUSE_SECRET_KEY` | `sk-lf-…` |

Redis stores the per-model dashboard, the experiment timeline, and
cached abliteration directions. NEUROSCAN runs without Redis but logs
warnings and skips persistence.

The OpenAI-compatible LLM endpoint is always external. Any endpoint
implementing `/v1/chat/completions` works: Ollama, vLLM, LM Studio, an
OpenAI proxy, etc.

---

## Optional HTTPS reverse proxy

Caddy is bundled as an opt-in profile. It auto-provisions Let's
Encrypt certs when `DEMO_HOSTNAME` is a real DNS name pointing at this
host:

```bash
DEMO_HOSTNAME=neuroscan.example.com docker compose --profile proxy up -d
```

For local testing keep `DEMO_HOSTNAME=localhost`; Caddy issues a
self-signed cert.

---

## Authentication

NEUROSCAN runs without authentication by default. For shared
deployments, place one of these in front:

- Caddy basic auth (`basic_auth` block in the Caddyfile)
- oauth2-proxy in front of Caddy
- Cloudflare Tunnel + Access policies

Every red-team capability is exposed without gating. Do not expose
this service to the public internet unauthenticated.

---

## Architecture

`server.py` is a FastAPI shell (~4100 lines) over ~14 specialised
engines. Each engine owns one capability and streams progress over
WebSockets.

| Module | Purpose |
|---|---|
| `activation_engine.py` | TransformerLens model manager, hooks, SAELens decomposition, 3D viewer, attention-head ablation, neuron diff scans, activation patching. |
| `abliteration_engine.py` | Refusal-direction discovery and removal (standard / norm-preserving / biprojected), batch testing, KL scoring, weight orthogonalisation, export. |
| `adversarial_engine.py` | Step-by-step nanoGCG loop with streamed progress. |
| `steering_engine.py` | repeng-based concept-vector steering. |
| `benchmark_engine.py` | TruthfulQA, Detoxify toxicity, CrowS bias, built-in seven-category probe suite. |
| `optimizer_engine.py` | Optuna multi-objective search over abliteration parameters. |
| `guardrails_engine.py` | NeMo-Guardrails-style rails with regex fallback. |
| `probe_engine.py` | Linear probes on hidden states; built-in and user-defined concepts. |
| `fuzzyai_engine.py` | LLM-assisted attacks: PAIR, Crescendo, Best-of-N, ActorAttack, Genetic. |
| `circuit_engine.py` | Attribution-graph generation; attention-based fallback. |
| `kvcache_engine.py` | KV-cache analysis and attack simulations. |
| `moe_engine.py` | Mixture-of-Experts routing analysis. |
| `embedding_engine.py` | Embedding inversion and PaCMAP visualisation. |
| `vlm_engine.py` | Steganographic VLM prompt-injection images (typography / LSB / EXIF). |
| `redteam_suite.py` | Native Garak / DeepTeam / PyRIT / Promptfoo / promptmap2 probes. |
| `auto_redteam.py` | Six-stage automated red-team pipeline orchestrator. |

Frontend: `static/index.html` (single page) with `static/js/app.js`,
`neural-viz.js`, `viz-views.js` driving the Three.js views. ~120 REST
endpoints plus `/ws/activations` for streaming.

---

## Troubleshooting

- **Compose refuses to start with `set OLLAMA_BASE_URL …` /
  `set LLM_MODEL …`** — both are required. Edit `.env`.
- **`OLLAMA_BASE_URL` unreachable from inside the container** — from
  the container, run `curl ${OLLAMA_BASE_URL%/v1}/api/tags`. On Linux,
  `host.docker.internal` requires either an `extra_hosts` entry or a
  LAN IP. Switching to `http://<LAN-IP>:11434/v1` is the most robust
  fix.
- **`nvidia-smi` works on host but not in container** — the NVIDIA
  Container Toolkit isn't wired into Docker. Run
  `sudo nvidia-ctk runtime configure --runtime=docker && sudo
  systemctl restart docker` and re-run the step-3 test container.
- **Gemma-2 download fails 401/403** — accept the licence on
  huggingface.co for `google/gemma-2-2b`, set `HF_TOKEN` in `.env`,
  `docker compose restart neuroscan`.
- **Out of memory loading Gemma-2** — stay on GPT-2 Small or run on a
  host with more VRAM. Model switching is reversible.
- **`unsupported gpu architecture` during PyTorch import** — rebuild
  with `--build-arg CUDA_ARCH=<your arch>`.
- **Container restarts in a loop** — almost always a GPU/driver
  mismatch. Confirm `docker run --rm --gpus all
  nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi` works.
- **Port collision on 8080 or 6379** — change `APP_PORT` and/or
  `REDIS_HOST_PORT` in `.env`.
- **Dashboard loses history after `docker compose down -v`** — `-v`
  removes the named Redis volume.
- **Health check failing on first boot** — the first model + SAE
  download can exceed the 120 s start-period. Check
  `docker compose logs neuroscan`.

---

## FAQ

**Can I use a CPU?** No. TransformerLens activation extraction,
SAELens decomposition, and GCG gradients all require CUDA.

**Do the models run locally?** The probe model (GPT-2 Small /
Pythia-70M / Gemma-2 2B) runs locally on the GPU via TransformerLens;
every interpretability and abliteration feature operates on it. The
configured Ollama endpoint is used only for attacker roleplay,
explainer narratives, and report generation.

**Is anything sent to the internet?** Only the initial HuggingFace
model and SAE downloads, cached in a Docker volume. All inference,
attacks, and analyses run locally.

**Can I add my own model?** Yes. Extend `MODEL_REGISTRY` in
`activation_engine.py` with the TransformerLens model name, layer
count, hidden size, and matching SAE release.

**How do I get a one-shot vulnerability report?** Run Auto Red Team
from the Red Team tab. It chains six attack stages and writes a
composite report to the Dashboard.

---

## Credits

Built by Andrew Meinecke.
