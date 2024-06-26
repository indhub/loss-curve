from typing import Any, Dict, Optional, Union

from axlearn.common import causal_lm, config
from axlearn.common.attention import (
    CausalAttentionLogitBiasLayer,
    FusedQKVLinear,
    RepeatedTransformerLayer,
    StackedTransformerLayer,
    RoFormerQKVLinear,
)
from axlearn.common.embedding import TransformerTextEmbeddings
from axlearn.common.layers import RMSNorm
from axlearn.experiments.text.gpt.common import STEP_DTYPE, learner_config, mesh_shape_from_axes
from axlearn.experiments.text.gpt.common import model_config as common_model_config
from axlearn.experiments.text.gpt.common import scaled_hidden_dim
from axlearn.common.utils import DataPartitionType
import jax
import os

MODEL_SIZES = ("test", "7B", "testgpu", "testtrn")
MAX_SEQUENCE_LENGTH=512
TRN_MODEL_AXIS_SIZE=8
GRADIENT_ACCUMULATION_MICROBATCHES=8

if "NEURON_CC_FLAGS" in os.environ:
    os.environ["NEURON_CC_FLAGS"] += " --internal-hlo2tensorizer-options='--verify-hlo --num-concat-graphs=" + str(GRADIENT_ACCUMULATION_MICROBATCHES) + '\''

def get_trainer_kwargs(model_size: str, *, vocab_size: int) -> Dict[str, Any]:
    """Construct default trainer kwargs given a model size."""
    # dict() is more readable here.
    # pylint: disable=use-dict-literal
    if model_size == "test":
        trainer_kwargs = dict(
            model_kwargs=dict(
                num_layers=4,
                hidden_dim=8,
                ffn_dim=scaled_hidden_dim(scale=8 / 3, round_up_to_multiples_of=16),
                num_heads=4,
                vocab_size=32,
            ),
            learner_kwargs=dict(
                peak_lr=6e-4,
                weight_decay=0.01,
            ),
            input_partition_type=DataPartitionType.FULL,
            max_sequence_length=64,
            train_batch_size=16,
            max_step=3000,
            mesh_shape=mesh_shape_from_axes(),  # cpu
        )
    elif model_size == "testgpu":
        trainer_kwargs = dict(
            model_kwargs=dict(
                num_layers=4,
                hidden_dim=128 * 8,
                num_heads=8,
                dropout_rate=0,
            ),
            learner_kwargs=dict(peak_lr=1e-4, lr_warmup_steps=500, weight_decay=1e-5),
            input_partition_type=DataPartitionType.FULL,
            train_batch_size=32,
            max_step=20000,
            mesh_shape=mesh_shape_from_axes(fsdp=-1),
        )
    elif model_size == "testtrn":
        trainer_kwargs = dict(
            model_kwargs=dict(
                num_layers=4,
                hidden_dim=128 * 8,
                num_heads=8,
                dropout_rate=0,
            ),
            learner_kwargs=dict(peak_lr=1e-4, lr_warmup_steps=500, weight_decay=1e-5),
            input_partition_type = DataPartitionType.DATA if hasattr(DataPartitionType, 'DATA') else DataPartitionType.FULL,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            train_batch_size=int((jax.device_count()/TRN_MODEL_AXIS_SIZE)*GRADIENT_ACCUMULATION_MICROBATCHES),
            max_step=20000,
            mesh_shape=mesh_shape_from_axes(data=-1, model=TRN_MODEL_AXIS_SIZE),
            eval_batch_size=int((jax.device_count()/TRN_MODEL_AXIS_SIZE)*GRADIENT_ACCUMULATION_MICROBATCHES),
        )
    elif model_size == "7B":
        trainer_kwargs = dict(
            model_kwargs=dict(
                num_layers=32,
                hidden_dim=128 * 32,
                num_heads=32,
            ),
            learner_kwargs=dict(peak_lr=3e-4, weight_decay=0.1),
            input_partition_type=DataPartitionType.FULL,
            train_batch_size=4 * 1024 * 1024 // MAX_SEQUENCE_LENGTH,  # 4M tokens.
            max_step=500_000,  # 2T tokens // 4M tokens/step.
            mesh_shape=mesh_shape_from_axes(fsdp=-1),
            mesh_rules=(
                # tpu-v4. step time: 3.03s.
                ("tpu-v4-(1024|2048)", mesh_shape_from_axes(data=-1, fsdp=16)),
                # tpu-v5e. step time: TBD.
                ("tpu-v5litepod-256", mesh_shape_from_axes(data=-1, fsdp=16)),
                # H100/A100 80G. Maximum per-node batch size = 64, hence need >= 32 nodes.
                (
                    "gpu-(p5.48xlarge|p4de.24xlarge)-(256|512|1024)",
                    mesh_shape_from_axes(data=-1, fsdp=8),
                ),
            ),
        )
    else:
        raise NotImplementedError(f"Unknown model size {model_size}.")
    model_kwargs = trainer_kwargs.pop("model_kwargs")
    model_kwargs.setdefault("vocab_size", vocab_size)
    trainer_kwargs["model_cfg"] = model_config(**model_kwargs)
    trainer_kwargs["learner_cfg"] = learner_config(
        max_step=trainer_kwargs["max_step"],
        # gradient_accumulation_microbatches=GRADIENT_ACCUMULATION_MICROBATCHES,
        **({"gradient_accumulation_microbatches": GRADIENT_ACCUMULATION_MICROBATCHES} if hasattr(DataPartitionType, 'DATA') else {}),
        **trainer_kwargs.pop("learner_kwargs"),
    )
    # pylint: enable=use-dict-literal
    return trainer_kwargs

def model_config(
    *,
    num_layers: int,
    hidden_dim: int,
    num_heads: int,
    vocab_size: int,
    dropout_rate: float = 0.0,
    ffn_dim: Optional[Union[int, config.FunctionConfigBase]] = None,
) -> causal_lm.Model.Config:
    """Returns an LM model config based on the given hyperparams.

    Args:
        num_layers: The number of Transformer Layers.
        hidden_dim: The Transformer layer input/output dim.
        num_heads: The number of attention heads.
        vocab_size: The vocabulary size.
        dropout_rate: The dropout rate applied throughout the model.
            Defaults to 0.0 (i.e. no dropout).
        ffn_dim: The feed-forward dimension or config function.
            If None, defaults to a setting from https://arxiv.org/abs/2002.05202.

    Returns:
        A causal LM config.
    """
    # SwiGLU from https://arxiv.org/abs/2002.05202.
    activation_fn = ("nn.silu", "linear")
    if ffn_dim is None:
        ffn_dim = scaled_hidden_dim(scale=8 / 3, round_up_to_multiples_of=256)
    cfg = common_model_config(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        vocab_size=vocab_size,
        stack_cfg=StackedTransformerLayer.default_config(),
        activation_fn=activation_fn,
        ffn_dim=ffn_dim,
        normalization=RMSNorm.default_config().set(eps=1e-5, forward_dtype=None),
        dropout_rate=dropout_rate,
        emb_cfg=TransformerTextEmbeddings.default_config().set(pos_emb=None),
        attention_mask=CausalAttentionLogitBiasLayer.default_config(),
        # RoPE embeddings: https://arxiv.org/abs/2104.09864.
        attention_qkv_linear=RoFormerQKVLinear.default_config().set(
            input_linear=FusedQKVLinear.default_config().set(cache_dtype=STEP_DTYPE),
            rotary_value=False,
        ),
    )
    return cfg
