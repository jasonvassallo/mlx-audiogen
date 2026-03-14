# Phase 9g: LoRA Fine-Tuning for MusicGen

**Date:** 2026-03-13
**Scope:** MusicGen LoRA training pipeline, CLI, server/API integration, web UI (this window). Stable Audio LoRA training + music library scanner deferred to subsequent windows.

## Overview

Train low-rank adaptation (LoRA) style adapters on a user's music so the MusicGen model learns their personal aesthetic. The system is model-agnostic across MusicGen variants (small, medium, large, melody, stereo, style). The LoRA injection code is designed to also support Stable Audio's DiT in a future phase.

LoRA adds small trainable matrices to frozen base model layers. At inference, the adapter modifies the model's behavior without changing base weights. Adapters are ~3-50MB vs. the full model's 1-5GB.

## Training Data Pipeline

### Dataset Format

A directory with audio files + optional `metadata.jsonl`:

```
my-training-data/
├── metadata.jsonl        # {"file": "track1.wav", "text": "upbeat house, 128 BPM, A minor"}
├── track1.wav
├── track2.wav            # no metadata entry -> filename parsed as description
└── deep_bass_groove.flac # -> "deep bass groove"
```

- **Supported formats:** `.wav`, `.mp3`, `.flac`, `.aiff`
- **metadata.jsonl:** One JSON object per line with `file` and `text` fields. Malformed lines are skipped with a warning. Missing `file` or `text` fields skip with warning. References to non-existent files skip with warning.
- **Filename fallback:** Replace `_` and `-` with spaces, strip extension, use as description

### Data Loading Flow

1. Scan directory for supported audio files
2. Load `metadata.jsonl` if present; for files without entries, generate description from filename
3. Load each audio file via `soundfile`, convert to mono, resample to model's native rate (32kHz for MusicGen)
4. Chunk into segments: default 10s, configurable 5-40s via settings. Tracks shorter than chunk size used whole. Tracks longer than 40s always chunked (MusicGen position limit is 2048 tokens = 41s). Chunks are contiguous (no overlap). Last chunk shorter than half the chunk size is discarded; otherwise used as-is. Same text description is applied to all chunks from a track.
5. For each chunk: encode through EnCodec -> token sequence of shape `(B, num_codebooks, T)` (EnCodec's native output shape), then transpose to `(B, T, num_codebooks)` for the decoder
6. Apply codebook delay pattern to tokens (see Training Loop section)
7. Tokenize text description through T5 tokenizer -> input IDs + attention mask

### Output

A list of `(text_input_ids, text_attention_mask, delayed_audio_tokens, valid_mask)` tuples ready for the training loop.

## LoRA Architecture

### Custom LoRALinear Implementation

MLX core (`mlx.nn`) does not include a `LoRALinear` class. We implement our own in `lora/inject.py` (~30 lines):

```python
class LoRALinear(nn.Module):
    """Low-rank adaptation wrapper around a frozen nn.Linear."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        in_features = base.weight.shape[1]  # MLX: (out, in)
        out_features = base.weight.shape[0]
        self.base = base
        self.base.freeze()  # Freeze base weights
        self.scale = alpha / rank
        # A initialized with small random values, B initialized to zero
        # so LoRA output starts at zero (no initial perturbation)
        self.lora_a = mx.random.normal((in_features, rank)) * 0.01
        self.lora_b = mx.zeros((rank, out_features))

    def __call__(self, x: mx.array) -> mx.array:
        base_out = self.base(x)
        lora_out = (x @ self.lora_a @ self.lora_b) * self.scale
        return base_out + lora_out
```

Key properties:
- Base weights frozen at construction time
- B initialized to zero -> LoRA contribution starts at zero (model behaves identically to base until trained)
- `lora_a` and `lora_b` are the only trainable parameters
- `scale = alpha / rank` controls the magnitude of LoRA's contribution

### Target Layers

MusicGen's decoder has 24 `TransformerBlock` layers, each containing:
- `self_attn`: q_proj, k_proj, v_proj, out_proj -- `nn.Linear(1024, 1024, bias=False)`
- `encoder_attn`: q_proj, k_proj, v_proj, out_proj -- same (cross-attention to T5 text)
- `fc1`, `fc2` -- FFN layers (not targeted by default)

### Training Profiles (Basic Mode)

The web UI and CLI expose three beginner-friendly presets:

| Profile | Description | Layers Targeted | Rank | Alpha | Approx. Size | Speed |
|---------|-------------|-----------------|------|-------|---------------|-------|
| Quick & Light | Subtle style nudges, fastest training | self_attn: q, v | 8 | 16 | ~3 MB | Fastest |
| Balanced *(default)* | Best quality-to-speed ratio | self_attn: q, v, out | 16 | 32 | ~9 MB | Moderate |
| Deep | Captures nuanced stylistic details | self_attn + encoder_attn: all | 32 | 64 | ~50 MB | Slowest |

Size estimates (float32): `2 * rank * dim * num_targets * num_layers * 4 bytes`. For small model (dim=1024, 24 layers): Quick = 2*8*1024*2*24*4 = 3.1MB, Balanced = 2*16*1024*3*24*4 = 9.4MB, Deep = 2*32*1024*8*24*4 = 50.3MB.

### Advanced Controls (Expandable Section)

Revealed by toggling "Advanced" in the web UI:

- **Target layers:** Checkboxes for q_proj, k_proj, v_proj, out_proj x self_attn / encoder_attn
- **Rank:** Slider 4-64 (default 16)
- **Alpha:** Slider 8-128 (default 32)
- **Learning rate:** 1e-5 to 1e-3 (default 1e-4)
- **Epochs:** 1-100 (default 10)
- **Chunk duration:** Slider 5s-40s (default 10s), with "Max (40s)" label at top end
- **Batch size:** 1-8 (default 1, constrained by available unified memory)

Profile selection pre-fills the advanced fields; switching to Advanced lets users override individually.

### CLI `--targets` Format

CLI accepts short names: `--targets q_proj,v_proj,out_proj`. These are applied to self_attn by default. To target encoder_attn, prefix with `encoder_attn.`: `--targets q_proj,v_proj,encoder_attn.q_proj,encoder_attn.v_proj`. Config.json stores fully-qualified names: `["self_attn.q_proj", "self_attn.v_proj"]`.

## Training Loop

### Causal Mask for Teacher-Forcing

MusicGen's `MusicGenModel.__call__` currently does not pass a causal attention mask to its transformer layers (generation uses KV cache instead). For teacher-forcing training, we need the full sequence processed at once with a causal mask to prevent attending to future tokens.

**Modification to `MusicGenModel.__call__`:** Add an optional `mask` parameter that is threaded through to each `TransformerBlock`. When `mask=None` (default), behavior is unchanged (backward compatible with existing generation code). When a causal mask is provided, it is passed to `self_attn` calls.

```python
def __call__(self, audio_tokens, conditioning, cache=None, cross_kv_caches=None, mask=None):
    # ... existing embedding code ...
    for layer, c, xc in zip(self.layers, cache, cross_kv_caches):
        x = layer(x, conditioning, mask=mask, cache=c, cross_kv_cache=xc)
    # ... existing logits code ...
```

The trainer creates a standard causal mask:
```python
seq_len = audio_tokens.shape[1] - 1  # input length
mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
```

### Codebook Delay Pattern in Training

MusicGen uses a codebook delay pattern during generation: codebook `k` is delayed by `k` steps. Training data must replicate this pattern for the model to learn the correct distribution.

**Delay application to ground-truth tokens:**
1. Start with EnCodec output: shape `(1, T, num_codebooks)`, all valid tokens
2. Create delayed version: codebook 0 at position 0, codebook 1 at position 1, etc.
3. Fill delayed positions with BOS token (2048)
4. Create a validity mask: `valid[t, k] = True if t >= k` -- loss is only computed on valid (non-BOS) positions

```python
def apply_delay_pattern(tokens, num_codebooks, bos_token_id):
    """Apply MusicGen's codebook delay pattern to ground-truth tokens.

    Args:
        tokens: Shape (B, T, K) -- raw EnCodec tokens
        num_codebooks: K
        bos_token_id: Token ID for BOS/padding (2048)

    Returns:
        delayed_tokens: Shape (B, T + K - 1, K) -- delayed with BOS fill
        valid_mask: Shape (B, T + K - 1, K) -- True where loss should be computed
    """
    B, T, K = tokens.shape
    new_T = T + K - 1
    delayed = mx.full((B, new_T, K), bos_token_id)
    valid = mx.zeros((B, new_T, K), dtype=mx.bool_)
    for k in range(K):
        delayed[:, k:k+T, k] = tokens[:, :, k]
        valid[:, k:k+T, k] = True
    return delayed, valid
```

### Teacher-Forcing with Delay Pattern

- **Input:** `delayed_tokens[:, :-1, :]` (all positions except last)
- **Target:** `delayed_tokens[:, 1:, :]` (shifted by one)
- **Loss:** Cross-entropy between predicted logits and target tokens, masked by `valid_mask[:, 1:, :]` (only compute loss on valid, non-BOS positions)
- **Conditioning:** T5 encoding of the text description, projected to decoder dimension via `enc_to_dec_proj`

### Full Training Flow

1. Load base model via `MusicGenPipeline.from_pretrained()` (existing auto-download)
2. Freeze all base parameters: `pipeline.model.freeze()`, `pipeline.t5.freeze()`, `pipeline.encodec.freeze()`
3. Apply LoRA to target layers: `apply_lora(pipeline.model, targets, rank, alpha)`
4. Create AdamW optimizer on trainable (LoRA) parameters only
5. Pre-encode all training data: for each audio chunk, run through EnCodec to get token sequences, apply delay pattern, tokenize text through T5. Cache these to avoid re-encoding each epoch.
6. For each epoch:
   - Shuffle training samples
   - For each sample (or mini-batch):
     a. T5 encode text -> conditioning
     b. Forward pass with causal mask: `logits = model(input_tokens, conditioning, mask=causal_mask)`
     c. Compute masked cross-entropy loss across all codebooks
     d. Backward pass (only LoRA parameters have gradients)
     e. Optimizer step
     f. Graph materialization to bound memory
   - Log: epoch, step, loss (printed + progress callback for server/UI)
7. Save LoRA weights + config to output directory

### Early Stopping

Simple patience-based early stopping to prevent overfitting on small datasets:
- Track average loss per epoch
- If loss hasn't improved for `patience` epochs (default: 3), stop training
- Save the best-loss checkpoint, not just the final one
- Can be disabled via `--no-early-stop` CLI flag or Advanced settings toggle

### Stop Signal

Training checks a threading.Event between steps. When `/api/train/stop/{id}` is called or the user clicks Stop in the UI, the event is set. The trainer saves the current weights (completed steps from current epoch) and exits cleanly.

### Memory Considerations

- 10s chunk at 50 Hz = 500 tokens x 4 codebooks x 2048 vocab -> manageable on 24GB
- 40s chunk = 2000 tokens -- fits on 24GB for small model, may need batch_size=1
- Gradient checkpointing not needed for LoRA (only A/B matrices have gradients)
- Graph materialization after each step prevents lazy graph accumulation
- Pre-encoded tokens cached in memory (small: ~few MB for typical dataset)

## Storage & Discovery

### LoRA Adapter Format

```
~/.mlx-audiogen/loras/my-style/
├── lora.safetensors    # LoRA A/B weight matrices only (~3-50MB)
├── config.json         # Training configuration + metadata
```

**Saving LoRA weights:** The trainer must extract only LoRA parameters for saving, NOT the full model. `list_lora_params(model)` walks the parameter tree and returns only keys containing `lora_a` or `lora_b`. These are saved via `mx.save_safetensors()`. Do NOT use `model.save_weights()` which would include all parameters (including frozen base weights, resulting in a multi-GB file).

**config.json contents:**
```json
{
  "name": "my-style",
  "base_model": "musicgen-small",
  "hidden_size": 1024,
  "rank": 16,
  "alpha": 32,
  "targets": ["self_attn.q_proj", "self_attn.v_proj", "self_attn.out_proj"],
  "profile": "balanced",
  "chunk_seconds": 10,
  "epochs": 10,
  "final_loss": 2.34,
  "best_loss": 2.18,
  "training_samples": 42,
  "created_at": "2026-03-13T15:30:00Z"
}
```

`hidden_size` is stored for validation: LoRA weights are dimension-specific, so a LoRA trained on musicgen-small (1024) cannot be loaded on musicgen-large (2048).

### Discovery

- **Default directory:** `~/.mlx-audiogen/loras/`
- **Auto-discovered** by server on startup (same pattern as `~/.mlx-audiogen/models/`)
- **CLI override:** `--lora-path ./custom/path/` for non-standard locations
- **Validation:** config.json must exist with valid `base_model`, `targets`, and `hidden_size`. At load time, hidden_size is checked against the base model's decoder.hidden_size. Mismatch raises ValueError with a clear message.

## Inference Integration

### Loading a LoRA

1. Load base model normally via `from_pretrained()`
2. Read LoRA config.json to get targets, rank, alpha, hidden_size
3. **Validate:** Check config.hidden_size matches model.hidden_size. Raise ValueError on mismatch.
4. Apply LoRA: `apply_lora(model.model, config.targets, config.rank, config.alpha)`
5. Load LoRA weights: `model.load_weights(lora_weights, strict=False)`
6. Generate as normal -- LoRA modifies forward passes transparently

### Removing a LoRA

`remove_lora(model)` walks the model tree, finds `LoRALinear` instances, and replaces them with their stored `base` Linear. This restores the original model exactly (base weights were frozen, never modified).

### CLI

```bash
# By name (auto-discovered from ~/.mlx-audiogen/loras/):
uv run mlx-audiogen --model musicgen --prompt "..." --lora my-style \
  --weights-dir ./converted/musicgen-small

# By explicit path:
uv run mlx-audiogen --model musicgen --prompt "..." \
  --lora-path ./my-loras/custom/ --weights-dir ./converted/musicgen-small
```

### Training CLI

```bash
# Basic (uses "balanced" profile):
uv run mlx-audiogen-train \
  --data ./my-training-data/ \
  --base-model musicgen-small \
  --name my-style \
  --epochs 10

# With profile:
uv run mlx-audiogen-train \
  --data ./my-training-data/ \
  --base-model musicgen-small \
  --name my-style \
  --profile deep

# Advanced overrides:
uv run mlx-audiogen-train \
  --data ./my-training-data/ \
  --base-model musicgen-small \
  --name my-style \
  --rank 32 --alpha 64 \
  --targets q_proj,v_proj,out_proj \
  --chunk-seconds 20 \
  --lr 5e-4 --epochs 20 --batch-size 2
```

## Server & API

### New Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/loras` | List available LoRA adapters (name, base_model, profile, size, created_at) |
| `GET` | `/api/loras/{name}` | Get LoRA details (full config.json) |
| `DELETE` | `/api/loras/{name}` | Delete a LoRA adapter |
| `POST` | `/api/train` | Start LoRA training job (returns job ID, runs in dedicated thread) |
| `GET` | `/api/train/status/{id}` | Poll training status (epoch, step, loss, progress 0.0-1.0) |
| `POST` | `/api/train/stop/{id}` | Stop training early (saves current checkpoint) |

### Modified Endpoints

- `POST /api/generate`: New optional `lora` field (string -- name from registry or path). Server applies LoRA before generation, removes after (or keeps if same LoRA will be reused).

### Training Thread Isolation

Training runs in a **dedicated thread** separate from the generation `ThreadPoolExecutor`. While training is active:
- Generation requests are still accepted but will contend for GPU
- The server returns a warning header `X-Training-Active: true` on generation responses so the UI can show an indicator
- Only one training job can run at a time; submitting a second returns HTTP 409 Conflict

### LoRA Lifecycle in Pipeline Cache

LoRA application modifies the model in-place (replaces `nn.Linear` with `LoRALinear`). To manage this cleanly:

1. **Apply:** When a generate request includes `lora=X`, check if the pipeline already has LoRA X applied. If not, remove any existing LoRA first (`remove_lora`), then apply LoRA X.
2. **Remove:** `remove_lora()` replaces `LoRALinear` back with the original `base` Linear. Base weights are preserved inside the `LoRALinear.base` attribute, so removal is lossless.
3. **Cache key:** Pipeline cache key remains `model_name` (string). The server tracks which LoRA (if any) is currently applied to each cached pipeline via a separate dict: `active_loras: dict[str, Optional[str]]`.

This avoids duplicating full model instances for different LoRA variants.

## Web UI

### LoRA Selector (Generate Tab)

Below the model selector dropdown, a new "LoRA" dropdown:
- Options: "None" (default), plus all discovered LoRA names
- Shows base model compatibility: if selected LoRA's `base_model` doesn't match current model, show amber warning text
- Refresh button to re-scan `~/.mlx-audiogen/loras/`

### Train Tab (New Sidebar Tab)

A new "Train" tab alongside Generate / Suggest / Settings:

**Basic mode (default view):**
- **Data directory:** Text input for server-side path ("Path to folder with WAV files + optional metadata.jsonl")
- **Name:** Text input for the LoRA adapter name (validated: `^[a-zA-Z0-9_-]{1,64}$`)
- **Base model:** Dropdown (populated from available models, MusicGen variants only)
- **Training profile:** Three cards -- Quick & Light / Balanced / Deep -- with plain-English descriptions
- **Chunk duration:** Slider 5s-40s with "Max (40s)" option
- **Start Training** button -> shows progress bar + live loss value
- **Stop** button (saves current checkpoint)

**Advanced mode (expandable):**
- All advanced controls listed in the Architecture section above
- Pre-filled from the selected profile; editable independently
- Early stopping toggle + patience slider

**Training status area:**
- Progress bar: "Epoch 3/10 - Step 12/42 - Loss: 2.34"
- Simple line chart of loss over steps (canvas-based, no external charting lib)
- "Training complete" badge with final loss and best loss when done
- "Early stopped" badge if patience was exceeded

### Settings Tab Addition

LoRA section in Settings tab:
- Default LoRA directory path (read-only, shows `~/.mlx-audiogen/loras/`)
- List of installed LoRAs with delete button

## New Files

```
mlx_audiogen/
├── lora/
│   ├── __init__.py        # Public API: apply_lora, remove_lora, load_lora_config, LoRALinear
│   ├── inject.py          # LoRALinear class, apply_lora(), remove_lora(), list_lora_params()
│   ├── dataset.py         # LoRADataset: scan dir, load audio, chunk, encode, apply delay
│   ├── trainer.py         # Training loop: freeze, inject, train, save, early stopping
│   └── config.py          # LoRAConfig dataclass, PROFILES dict, DEFAULT_LORAS_DIR
├── cli/
│   └── train.py           # mlx-audiogen-train entry point
web/src/
├── components/
│   ├── TrainPanel.tsx      # Training UI: data dir, profile, basic/advanced, progress, loss chart
│   └── LoRASelector.tsx    # Dropdown for selecting active LoRA in generation
```

## Modified Files

- `pyproject.toml` -- new `mlx-audiogen-train` entry point
- `mlx_audiogen/models/musicgen/model.py` -- add optional `mask` parameter to `__call__`
- `mlx_audiogen/cli/generate.py` -- `--lora` and `--lora-path` flags
- `mlx_audiogen/server/app.py` -- `/api/loras`, `/api/train` endpoints, `lora` field in generate, training thread, active_loras tracking
- `mlx_audiogen/models/musicgen/pipeline.py` -- `load_lora()` / `remove_lora()` methods
- `web/src/store/useStore.ts` -- LoRA state: available list, selected, training job status
- `web/src/api/client.ts` -- LoRA + training API calls
- `web/src/components/ParameterPanel.tsx` -- LoRA selector integration
- `web/src/components/App.tsx` -- Train tab in sidebar TabBar
- `web/src/types/api.ts` -- LoRA + training TypeScript types

## Constraints

- **Position limit:** MusicGen max_position_embeddings=2048 -> max ~40s chunks. Auto-enforced.
- **Memory:** 24GB unified memory on M4 Pro. Small model + 10s chunks + batch_size=1 fits comfortably. Larger models or longer chunks may need batch_size=1 and the upcoming M5 Pro (64GB).
- **Base model compatibility:** LoRA trained on musicgen-small won't work on musicgen-large (different hidden dimensions). config.json records hidden_size + base_model for validation at load time.
- **No quantized base models:** LoRA injection targets `nn.Linear` layers. If the base model were quantized, those layers would be different types. Since we don't use `nn.quantize()` (per CLAUDE.md), this is not an issue.
- **EnCodec dependency:** Training requires EnCodec to tokenize audio. EnCodec is already loaded as part of the pipeline, so no new dependencies.
- **Single training job:** Only one training job at a time (GPU contention). Second job submission returns 409.
- **Future: Stable Audio LoRA:** The `LoRALinear` class and `apply_lora`/`remove_lora` functions are model-agnostic (they work on any `nn.Linear`). A future phase will add a diffusion-specific trainer for Stable Audio's DiT (noise prediction + MSE loss), reusing the injection infrastructure from this phase.
