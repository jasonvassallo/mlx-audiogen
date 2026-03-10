"""Configuration for HTDemucs model."""

from dataclasses import dataclass, field


@dataclass
class DemucsConfig:
    """HTDemucs model configuration.

    Default values match the pretrained ``htdemucs`` checkpoint.
    """

    sources: list[str] = field(
        default_factory=lambda: ["drums", "bass", "other", "vocals"]
    )
    audio_channels: int = 2
    channels: int = 48
    growth: int = 2
    depth: int = 4
    nfft: int = 4096
    kernel_size: int = 8
    stride: int = 4
    time_stride: int = 2
    cac: bool = True
    freq_emb: float = 0.2
    emb_scale: int = 10
    emb_smooth: bool = True
    norm_starts: int = 4
    norm_groups: int = 4
    dconv_mode: int = 1
    dconv_depth: int = 2
    dconv_comp: int = 8
    dconv_init: float = 1e-3
    rewrite: bool = True
    context: int = 1
    context_enc: int = 0
    bottom_channels: int = 0
    # Transformer
    t_layers: int = 5
    t_heads: int = 8
    t_hidden_scale: float = 4.0
    t_emb: str = "sin"
    t_norm_first: bool = True
    t_norm_in: bool = True
    t_norm_out: bool = True
    t_layer_scale: bool = True
    t_gelu: bool = True
    t_weight_pos_embed: float = 1.0
    t_max_period: float = 10000.0
    t_cross_first: bool = False
    # Metadata
    samplerate: int = 44100
    segment: float = 10.0
    use_train_segment: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "DemucsConfig":
        """Create config from a dictionary, ignoring unknown keys."""
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})
