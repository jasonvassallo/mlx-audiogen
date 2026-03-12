"""Tests for Phase 7b: LLM prompt enhancement, prompt memory, tag autocomplete."""

from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Tag Database
# ---------------------------------------------------------------------------


def test_tag_database_has_all_categories():
    from mlx_audiogen.shared.prompt_suggestions import TAG_DATABASE

    assert set(TAG_DATABASE.keys()) == {
        "genre",
        "mood",
        "instrument",
        "era",
        "production",
    }


def test_tag_database_each_category_nonempty():
    from mlx_audiogen.shared.prompt_suggestions import TAG_DATABASE

    for category, tags in TAG_DATABASE.items():
        assert len(tags) >= 10, f"Category '{category}' has too few tags: {len(tags)}"


def test_tag_database_entries_are_strings():
    from mlx_audiogen.shared.prompt_suggestions import TAG_DATABASE

    for category, tags in TAG_DATABASE.items():
        for tag in tags:
            assert isinstance(tag, str), f"Non-string tag in '{category}': {tag}"
            assert len(tag) > 0, f"Empty tag in '{category}'"


# ---------------------------------------------------------------------------
# Prompt Memory
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_dir(tmp_path):
    """Create a temp directory to use instead of ~/.mlx-audiogen/."""
    return tmp_path


def test_prompt_memory_init_creates_empty(memory_dir):
    from mlx_audiogen.shared.prompt_suggestions import PromptMemory

    mem = PromptMemory(memory_dir / "prompt_memory.json")
    assert mem.history == []
    assert mem.style_profile["generation_count"] == 0


def test_prompt_memory_add_entry(memory_dir):
    from mlx_audiogen.shared.prompt_suggestions import PromptMemory

    mem = PromptMemory(memory_dir / "prompt_memory.json")
    mem.add_entry("dark ambient pad", "musicgen", {"seconds": 10})
    assert len(mem.history) == 1
    assert mem.history[0]["prompt"] == "dark ambient pad"
    assert mem.style_profile["generation_count"] == 1


def test_prompt_memory_persist_and_load(memory_dir):
    from mlx_audiogen.shared.prompt_suggestions import PromptMemory

    path = memory_dir / "prompt_memory.json"
    mem = PromptMemory(path)
    mem.add_entry("synthwave arpeggio", "musicgen", {"seconds": 5})
    mem.save()

    mem2 = PromptMemory(path)
    assert len(mem2.history) == 1
    assert mem2.history[0]["prompt"] == "synthwave arpeggio"


def test_prompt_memory_eviction_at_max(memory_dir):
    from mlx_audiogen.shared.prompt_suggestions import PromptMemory

    mem = PromptMemory(memory_dir / "prompt_memory.json", max_entries=5)
    for i in range(7):
        mem.add_entry(f"prompt {i}", "musicgen", {})
    assert len(mem.history) == 5
    # Oldest evicted — newest kept
    assert mem.history[0]["prompt"] == "prompt 2"
    assert mem.history[-1]["prompt"] == "prompt 6"


def test_prompt_memory_style_profile_derivation(memory_dir):
    from mlx_audiogen.shared.prompt_suggestions import PromptMemory

    mem = PromptMemory(memory_dir / "prompt_memory.json")
    mem.add_entry("dark ambient pad, warm analog", "musicgen", {"seconds": 10})
    mem.add_entry("dark electronic, synth bass", "musicgen", {"seconds": 8})
    mem.add_entry("ambient dreamy, synth pad", "musicgen", {"seconds": 12})

    profile = mem.style_profile
    assert profile["generation_count"] == 3
    assert "ambient" in profile["top_genres"]
    assert profile["preferred_duration"] == 10  # median of [10, 8, 12]


def test_prompt_memory_enhanced_prompt_stored(memory_dir):
    from mlx_audiogen.shared.prompt_suggestions import PromptMemory

    mem = PromptMemory(memory_dir / "prompt_memory.json")
    mem.add_entry(
        "lo-fi", "musicgen", {}, enhanced_prompt="lo-fi chill beats, warm vinyl"
    )
    assert mem.history[0]["enhanced_prompt"] == "lo-fi chill beats, warm vinyl"


def test_prompt_memory_recent_prompts(memory_dir):
    from mlx_audiogen.shared.prompt_suggestions import PromptMemory

    mem = PromptMemory(memory_dir / "prompt_memory.json")
    for i in range(10):
        mem.add_entry(f"prompt {i}", "musicgen", {})
    recent = mem.recent_prompts(5)
    assert len(recent) == 5
    assert recent[0] == "prompt 9"  # newest first
    assert recent[4] == "prompt 5"


# ---------------------------------------------------------------------------
# MLX Model Discovery
# ---------------------------------------------------------------------------


def test_discover_mlx_models_empty_dir(tmp_path):
    from mlx_audiogen.shared.prompt_suggestions import discover_mlx_models

    models = discover_mlx_models([tmp_path])
    assert models == []


def test_discover_mlx_models_valid_model(tmp_path):
    from mlx_audiogen.shared.prompt_suggestions import discover_mlx_models

    # Create a fake MLX model directory
    model_dir = tmp_path / "mlx-community" / "Qwen3.5-9B-6bit"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"model_type": "qwen2"}')
    (model_dir / "model.safetensors").write_bytes(b"fake")
    (model_dir / "tokenizer_config.json").write_text("{}")

    models = discover_mlx_models([tmp_path])
    assert len(models) == 1
    assert models[0]["id"] == "mlx-community/Qwen3.5-9B-6bit"
    assert models[0]["name"] == "Qwen3.5-9B-6bit"
    assert "path" not in models[0]  # no path exposed


def test_discover_mlx_models_filters_non_llm(tmp_path):
    from mlx_audiogen.shared.prompt_suggestions import discover_mlx_models

    # Create a model dir WITHOUT tokenizer (like EnCodec)
    model_dir = tmp_path / "mlx-community" / "encodec-32khz"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"model_type": "encodec"}')
    (model_dir / "model.safetensors").write_bytes(b"fake")
    # No tokenizer_config.json or tokenizer.json

    models = discover_mlx_models([tmp_path])
    assert models == []


def test_discover_mlx_models_hf_snapshot_structure(tmp_path):
    from mlx_audiogen.shared.prompt_suggestions import discover_mlx_models

    # HF cache: models--org--name/snapshots/<hash>/
    model_root = tmp_path / "models--mlx-community--Qwen3.5-9B-6bit"
    snapshot = model_root / "snapshots" / "abc123def"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"model_type": "qwen2"}')
    (snapshot / "model.safetensors").write_bytes(b"fake")
    (snapshot / "tokenizer_config.json").write_text("{}")

    models = discover_mlx_models([tmp_path])
    assert len(models) == 1
    assert models[0]["id"] == "mlx-community/Qwen3.5-9B-6bit"


# ---------------------------------------------------------------------------
# LLM Enhancement
# ---------------------------------------------------------------------------


def test_enhance_with_llm_fallback_no_model():
    from mlx_audiogen.shared.prompt_suggestions import enhance_with_llm

    result = enhance_with_llm("dark ambient", model_path=None)
    assert result["original"] == "dark ambient"
    assert result["used_llm"] is False
    # Falls back to template engine
    assert len(result["enhanced"]) > len("dark ambient")


def test_enhance_with_llm_mock_success():
    from mlx_audiogen.shared.prompt_suggestions import enhance_with_llm

    with patch("mlx_audiogen.shared.prompt_suggestions._run_llm_inference") as mock_llm:
        mock_llm.return_value = (
            "dark ambient pad, warm analog, slow tempo, reverb-drenched"
        )
        result = enhance_with_llm("dark ambient", model_path="/fake/path")
        assert result["used_llm"] is True
        assert "reverb-drenched" in result["enhanced"]
        assert result["original"] == "dark ambient"


def test_enhance_with_llm_timeout_fallback():
    from mlx_audiogen.shared.prompt_suggestions import enhance_with_llm

    with patch("mlx_audiogen.shared.prompt_suggestions._run_llm_inference") as mock_llm:
        mock_llm.side_effect = TimeoutError("LLM timed out")
        result = enhance_with_llm("dark ambient", model_path="/fake/path")
        assert result["used_llm"] is False
        assert result["warning"] is not None
        assert (
            "timeout" in result["warning"].lower()
            or "timed out" in result["warning"].lower()
        )
