from enum import Enum


class SAEType(str, Enum):
    """Enum for different SAE types."""
    RELU = "relu"
    LAGRANGIAN = "lagrangian"
    GATED = "gated"
    TOPK = "topk"
    BATCH_TOPK = "batch_topk"
    JUMP_RELU = "jump_relu"
    MATRYOSHKA = "matryoshka"


class EncoderType(str, Enum):
    """Enum for different encoder types."""
    SCALE = "scale"
    SEPARATE = "separate"
    DECODER_TRANSPOSE = "decoder_transpose"
    NONE = "none"
