# mlx-audiogen

Text-to-audio generation and stem separation on Apple Silicon using [MLX](https://github.com/ml-explore/mlx). Supports **MusicGen**, **Stable Audio Open**, and **Demucs v4** stem separation.

Runs entirely on-device via Metal GPU — no cloud API needed.

**Full stack:** CLI, HTTP server, React web UI, VST3/AU plugin, Max for Live device, cloud deployment.

## Supported Models

| Model | Variants | Output | Sample Rate | Architecture |
|-------|----------|--------|-------------|--------------|
| MusicGen | small, medium, large | Mono | 32 kHz | Autoregressive (T5 + Transformer + EnCodec) |
| MusicGen Stereo | small, medium, large | Stereo | 32 kHz | Autoregressive (8 codebooks) |
| MusicGen Melody | base, large | Mono | 32 kHz | Autoregressive + Chroma conditioning |
| MusicGen Stereo Melody | base, large | Stereo | 32 kHz | Autoregressive (8 codebooks + Chroma) |
| MusicGen Style | base | Mono | 32 kHz | Autoregressive + MERT style conditioning |
| Stable Audio Open | small | Stereo | 44.1 kHz | Diffusion (T5 + DiT + Oobleck VAE) |
| Stable Audio Open | 1.0 | Stereo | 44.1 kHz | Diffusion (larger DiT, dual time conditioning) |
| HTDemucs (Demucs v4) | htdemucs, htdemucs_6s | 4 or 6 stems | 44.1 kHz | Hybrid U-Net + Cross-Transformer |

Pre-converted MLX weights for all variants are available on [HuggingFace](https://huggingface.co/jasonvassallo).

## Quick Start

```bash
# Install
git clone https://github.com/jasonvassallo/mlx-audiogen
cd mlx-audiogen
uv sync

# Convert weights (one-time per model variant)
uv run mlx-audiogen-convert --model facebook/musicgen-small --output ./converted/musicgen-small

# Generate audio
uv run mlx-audiogen \
  --model musicgen \
  --prompt "happy upbeat rock song with electric guitar" \
  --seconds 10 \
  --weights-dir ./converted/musicgen-small \
  --output my_song.wav
```

### Stable Audio Example

```bash
# Convert weights
uv run mlx-audiogen-convert --model stabilityai/stable-audio-open-small --output ./converted/stable-audio

# Generate (stereo, 44.1kHz)
uv run mlx-audiogen \
  --model stable_audio \
  --prompt "ambient electronic pad with warm reverb" \
  --seconds 15 \
  --steps 100 \
  --cfg-scale 7.0 \
  --weights-dir ./converted/stable-audio \
  --output ambient.wav
```

### Melody Conditioning Example

MusicGen melody variants can condition generation on an existing audio file, extracting its pitch contour (chromagram) to guide the output:

```bash
# Convert a melody variant
uv run mlx-audiogen-convert --model facebook/musicgen-melody --output ./converted/musicgen-melody

# Generate with melody conditioning
uv run mlx-audiogen \
  --model musicgen \
  --prompt "orchestral arrangement with strings" \
  --melody my_humming.wav \
  --seconds 10 \
  --weights-dir ./converted/musicgen-melody \
  --output orchestral.wav
```

The `--melody` flag accepts any WAV file. The pipeline extracts a 12-bin chromagram (one-hot pitch class per frame) and uses it as additional cross-attention conditioning alongside the text prompt. Melody variants also work without `--melody` for text-only generation.

### Style Conditioning Example

MusicGen style variants use a frozen MERT audio feature extractor to capture the timbre and texture of a reference audio clip, then guide generation via dual classifier-free guidance:

```bash
# Convert the style variant (uses audiocraft format, downloads MERT weights)
uv run mlx-audiogen-convert --model facebook/musicgen-style --output ./converted/musicgen-style

# Generate with style conditioning
uv run mlx-audiogen \
  --model musicgen \
  --prompt "upbeat electronic dance music" \
  --style-audio reference_track.wav \
  --style-coef 5.0 \
  --seconds 10 \
  --weights-dir ./converted/musicgen-style \
  --output styled.wav
```

The `--style-audio` flag accepts any WAV file as a timbre reference. The pipeline uses dual-CFG with three forward passes per step to blend text semantics with audio style. `--style-coef` controls how strongly the text prompt influences the output relative to the style (default: 5.0). Style variants also work without `--style-audio` for text-only generation.

### Stem Separation (Demucs v4)

Separate any audio into drums, bass, vocals, and other stems using the native MLX port of Meta's HTDemucs:

```bash
# Convert Demucs weights (one-time, requires torch)
uv sync --extra convert
uv run mlx-audiogen-convert --model htdemucs --output ./converted/demucs-htdemucs
```

The pipeline auto-downloads pre-converted weights from [HuggingFace](https://huggingface.co/jasonvassallo/demucs-htdemucs-mlx) if no local weights are found. Inference runs 100% on MLX — PyTorch is only needed for one-time weight conversion.

## HTTP Server

An optional FastAPI server enables integration with the web UI, DAW plugins, Max for Live, or any HTTP client:

```bash
# Install server dependencies
uv sync --extra server

# Launch app with web UI and auto-discover all converted models
uv run mlx-audiogen-app

# Or start server with specific models
uv run mlx-audiogen-server --weights-dir ./converted/musicgen-small --port 8420

# Multiple models (LRU cache keeps the 2 most recently used loaded)
uv run mlx-audiogen-server \
  --weights-dir ./converted/musicgen-small \
  --weights-dir ./converted/stable-audio \
  --port 8420

# Open browser on launch
uv run mlx-audiogen-server --weights-dir ./converted/musicgen-small --open

# Remote access (for web UI on other devices or cloud deployment)
uv run mlx-audiogen-server --weights-dir ./converted/musicgen-small --host 0.0.0.0
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/generate` | Submit generation request (returns job ID). Supports `output_mode`: `audio`, `midi`, or `both` |
| `GET` | `/api/status/{id}` | Poll job status (`queued`/`running`/`done`/`error`) with real-time `progress` (0.0-1.0) |
| `GET` | `/api/audio/{id}` | Download generated WAV |
| `GET` | `/api/midi/{id}` | Download generated MIDI (when `output_mode` is `midi` or `both`) |
| `GET` | `/api/models` | List available models and loading status |
| `GET` | `/api/jobs` | List all active/recent jobs |
| `GET` | `/api/health` | Health check for browser heartbeat |
| `POST` | `/api/suggest` | AI prompt suggestions (analyze prompt + return refined versions) |
| `POST` | `/api/enhance` | Enhance prompt via local LLM or template fallback |
| `POST` | `/api/midi-to-prompt` | Convert MIDI file to descriptive text prompt |
| `POST` | `/api/separate/{id}` | Separate audio into stems (drums/bass/vocals/other) |
| `GET` | `/api/presets` | List shared presets |
| `POST` | `/api/presets/{name}` | Save a preset |
| `GET` | `/api/presets/{name}` | Load a preset |
| `GET` | `/api/tags` | Tag database for prompt autocomplete |
| `GET` | `/api/llm/models` | List discovered local LLM models |
| `POST` | `/api/llm/select` | Select and load an LLM model |
| `GET` | `/api/memory` | Get prompt memory (history + style profile) |
| `GET` | `/api/settings` | Get server settings |
| `POST` | `/api/settings` | Update server settings |

Interactive API docs at `http://localhost:8420/docs` when running.

## Web UI

A React + TypeScript SPA with a dark, DAW-inspired interface:

```bash
# Development (hot reload, proxies API to :8420)
cd web && npm install && npm run dev   # http://localhost:3000

# Production (built and served by FastAPI)
cd web && npm run build
```

Features:
- **Generation**: Model selector, prompt textarea with tag autocomplete, model-aware parameter sliders, BPM-based duration mode
- **AI Enhancement**: Local LLM prompt enhancement with preview (accept/edit/use original)
- **Suggestions**: Prompt analysis tags (genre, mood, instruments) + refined suggestion cards
- **Playback**: Web Audio API waveform visualization, BPM-synced looping, time-stretch or vinyl pitch modes
- **Transport Bar**: DAW-style bottom bar with master BPM, pitch mode, audio device selector, connection status
- **History**: IndexedDB-persisted generation history with favorites, auto-delete retention, MIDI download
- **Stem Separation**: Color-coded inline audio players for drums/bass/vocals/other
- **Presets**: Save/load generation parameter presets
- **Remote Server**: Connect to a remote mlx-audiogen server (e.g., cloud Mac Mini)

## Native Plugin (VST3 / AU)

A JUCE-based plugin for Ableton Live, Logic Pro, and other DAWs:

```bash
cd plugin && git submodule update --init  # first time: clone JUCE
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release
# Installs to ~/Library/Audio/Plug-Ins/{VST3,Components}/
```

Features: auto-server-launch, local/remote server fallback (with Cloudflare Access auth), model auto-discovery, BPM sync, MIDI trigger, A/B/C/D variations, keep/discard workflow, beat-grid trimming, effects chain, Push 2 APVTS compatibility.

## Max for Live Integration

A Node for Max client (`m4l/mlx-audiogen.js`) connects Ableton Live directly to the HTTP server for generating audio from within your session:

1. Start the server (see above)
2. Load the Max for Live device onto a MIDI track
3. Type a prompt and click Generate — the WAV is auto-saved and the path is output for drag-to-track

See [`m4l/README.md`](m4l/README.md) for message reference and outlet documentation.

## Requirements

- Python 3.11+
- Apple Silicon Mac (M1/M2/M3/M4)
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

For converting models that only provide PyTorch `.bin` or `.th` weights:
```bash
uv sync --extra convert   # installs torch for weight conversion
```

## CLI Parameters

### Generation (`mlx-audiogen`)

| Parameter | MusicGen | Stable Audio | Default | Description |
|-----------|:--------:|:------------:|---------|-------------|
| `--model` | required | required | — | `musicgen` or `stable_audio` |
| `--prompt` | required | required | — | Text description of desired audio |
| `--seconds` | yes | yes | 5.0 | Duration (max ~30s for MusicGen, ~47s for Stable Audio) |
| `--output` | yes | yes | auto | Output WAV file path |
| `--seed` | yes | yes | random | Random seed for reproducibility |
| `--weights-dir` | yes | yes | — | Path to converted weights directory |
| `--temperature` | yes | — | 1.0 | Sampling temperature (higher = more creative) |
| `--top-k` | yes | — | 250 | Top-k sampling candidates |
| `--guidance-coef` | yes | — | 3.0 | Classifier-free guidance scale |
| `--melody` | yes | — | — | Audio file for melody conditioning (melody variants only) |
| `--style-audio` | yes | — | — | Audio file for style conditioning (style variants only) |
| `--style-coef` | yes | — | 5.0 | Dual-CFG text influence coefficient (style variants only) |
| `--steps` | — | yes | 8 | Number of diffusion steps |
| `--cfg-scale` | — | yes | 6.0 | CFG guidance scale |
| `--sampler` | — | yes | euler | ODE sampler (`euler` or `rk4`) |
| `--negative-prompt` | — | yes | "" | Negative prompt for CFG |

### Conversion (`mlx-audiogen-convert`)

| Parameter | Description |
|-----------|-------------|
| `--model` | HuggingFace repo ID (e.g., `facebook/musicgen-small`) or Demucs variant (`htdemucs`) |
| `--output` | Output directory for converted weights |
| `--dtype` | Optional: `float16`, `bfloat16`, or `float32` |
| `--trust-remote-code` | Allow non-whitelisted repo IDs |

#### Supported Repos

**MusicGen (mono):** `facebook/musicgen-small`, `facebook/musicgen-medium`, `facebook/musicgen-large`

**MusicGen Stereo:** `facebook/musicgen-stereo-small`, `facebook/musicgen-stereo-medium`, `facebook/musicgen-stereo-large`

**MusicGen Melody:** `facebook/musicgen-melody`, `facebook/musicgen-melody-large`

**MusicGen Stereo Melody:** `facebook/musicgen-stereo-melody`, `facebook/musicgen-stereo-melody-large`

**MusicGen Style:** `facebook/musicgen-style`

**Stable Audio:** `stabilityai/stable-audio-open-small`, `stabilityai/stable-audio-open-1.0`

**Demucs:** `htdemucs`, `htdemucs_6s`

> **Note:** Some HF repos (musicgen-medium, musicgen-large) only provide `pytorch_model.bin` files instead of safetensors. The converter handles both formats automatically, but PyTorch must be installed (`uv sync --extra convert`).

> **Note:** `stabilityai/stable-audio-open-1.0` is a gated model. You must accept the license agreement on the [HuggingFace model page](https://huggingface.co/stabilityai/stable-audio-open-1.0) before converting.

## Architecture

```
mlx_audiogen/
├── shared/           # T5 encoder, EnCodec, hub utils, audio I/O, MIDI, stem separation
├── models/
│   ├── musicgen/     # Autoregressive: T5 -> transformer -> EnCodec decode
│   │   ├── chroma.py # Chromagram extraction for melody conditioning
│   │   ├── mert.py   # MERT feature extractor for style conditioning
│   │   └── style_conditioner.py  # Style transformer + RVQ + BatchNorm
│   ├── stable_audio/ # Diffusion: T5 -> DiT (rectified flow) -> VAE decode
│   └── demucs/       # Source separation: HTDemucs v4 (hybrid U-Net + cross-transformer)
├── server/           # FastAPI HTTP server with LRU pipeline cache
├── cli/              # Unified CLI for generation and conversion
web/                  # React + Vite + TypeScript SPA (dark/pro audio UI)
plugin/               # JUCE native VST3/AU plugin
m4l/                  # Max for Live Node.js HTTP client for Ableton
```

**MusicGen**: Text -> T5 encode -> autoregressive transformer with KV cache + classifier-free guidance + codebook delay pattern -> top-k sampling -> EnCodec decode -> 32kHz WAV

**MusicGen Melody**: Text -> T5 encode + chromagram from audio -> cross-attention conditioning -> same pipeline as above

**MusicGen Style**: MERT extracts features from reference audio -> style transformer + RVQ -> dual-CFG with 3 forward passes per step (full, style-only, unconditional) -> same decode pipeline

**Stable Audio**: Text -> T5 encode + time conditioning -> rectified flow ODE sampling through DiT -> Oobleck VAE decode -> 44.1kHz stereo WAV

**HTDemucs (Demucs v4)**: Stereo 44.1kHz audio -> STFT -> complex-as-channels -> instance normalize -> parallel spectral U-Net + temporal U-Net with DConv residual branches -> CrossTransformerEncoder (5 layers, alternating self-attention and cross-attention) -> parallel decoder U-Nets with skip connections -> CaC mask + iSTFT + temporal denormalize -> 4 stems (drums, bass, other, vocals)

## Development

```bash
# Install with dev dependencies
uv sync

# Lint and format
uv run ruff check .
uv run ruff format .

# Run tests
uv run pytest                                     # unit tests (137 tests, ~13s)
uv run pytest -m integration -v                   # integration tests (real weights + GPU)
uv run pytest tests/test_specific.py::test_name   # single test

# Type checking
uv run mypy mlx_audiogen/

# Security audit
uv run bandit -r mlx_audiogen/ -c pyproject.toml
uv run pip-audit

# Import smoke test (no weights needed)
uv run python -c "from mlx_audiogen.models.musicgen import MusicGenPipeline; print('OK')"
```

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Links

- [GitHub](https://github.com/jasonvassallo/mlx-audiogen)
- [HuggingFace](https://huggingface.co/jasonvassallo)
- [Demucs MLX Weights](https://huggingface.co/jasonvassallo/demucs-htdemucs-mlx)
- [MusicGen paper](https://arxiv.org/abs/2306.05284)
- [Stable Audio Open paper](https://arxiv.org/abs/2407.14358)
- [Demucs v4 paper](https://arxiv.org/abs/2211.08553)
