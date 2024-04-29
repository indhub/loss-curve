from typing import Dict
import os

import tensorflow as tf

from axlearn.common.config import InstantiableConfig, config_for_function, config_for_class
from axlearn.common.input_lm import lm_text_preprocessor, text_to_lm_eval_input, seqio
from axlearn.common.trainer import SpmdTrainer
from axlearn.experiments.text.common import DataMixtureComponent, vocab
from tinystories import fuji
from axlearn.experiments.text.gpt.common import (
    evaler_config_dict,
    get_trainer_config_fn,
    make_config_name,
    mixture_train_input_source,
    tfds_input,
)
from axlearn.common.utils import DataPartitionType

from axlearn.common import (
    input_tf_data,
)

from axlearn.experiments.trainer_config_utils import TrainerConfigFn

t5_sentence_piece_vocab_file = os.path.join(os.getcwd(), "sentencepiece/t5-base")

def ds_fn(in_data_path) -> input_tf_data.BuildDatasetFn:

    def _parse_function(proto):
        keys_to_features = {'text': tf.io.FixedLenFeature([], tf.string)}
        parsed_features = tf.io.parse_single_example(proto, keys_to_features)
        return parsed_features

    def ds_fn() -> tf.data.Dataset:
        dataset = tf.data.TFRecordDataset(in_data_path)
        return dataset.map(_parse_function)

    return ds_fn

def _eval_input_sources() -> Dict[str, InstantiableConfig]:
    tf_record_path = os.path.join(os.getcwd(), "data/tinystories-valid.tfrecord")
    input_source = config_for_function(ds_fn).set(in_data_path=tf_record_path)
    vocab_cfg=config_for_class(seqio.SentencePieceVocabulary).set(sentencepiece_model_file=t5_sentence_piece_vocab_file)
    processor=config_for_function(text_to_lm_eval_input).set(
        vocab_cfg=vocab_cfg,
        max_len=fuji.MAX_SEQUENCE_LENGTH,
        stride=256
    )
    processed_input_source = config_for_function(input_tf_data.with_processor).set(
            source=input_source,
            processor=processor)
    return {"validation": processed_input_source}

def named_trainer_configs() -> Dict[str, TrainerConfigFn]:
    """Returns a mapping from trainer config names to TrainerConfigFn's."""
    arch = "fuji"
    vocab_size = 32_768
    vocab_cfg=config_for_class(seqio.SentencePieceVocabulary).set(sentencepiece_model_file=t5_sentence_piece_vocab_file)
    tf_record_path = os.path.join(os.getcwd(), "data/tinystories-train.tfrecord")
    input_source = config_for_function(ds_fn).set(in_data_path=tf_record_path)
    preprocessor=config_for_function(lm_text_preprocessor).set(max_padding_fraction=0.5,
                                                               vocab_cfg=vocab_cfg,
                                                               max_sequence_length=fuji.MAX_SEQUENCE_LENGTH,
                                                               replace_newlines_with="<n>",
                                                               window_size=256)
    processed_input_source = config_for_function(input_tf_data.with_processor).set(
            source=input_source,
            processor=preprocessor)

    config_map = {}
    for model_size in fuji.MODEL_SIZES:
        config_name = make_config_name(arch=arch, model_size=model_size)
        kwargs = fuji.get_trainer_kwargs(model_size, vocab_size=vocab_size)
        # pylint: disable-next=unexpected-keyword-arg,missing-kwoa
        kwargs.pop("max_sequence_length", 0)
        config_map[config_name] = get_trainer_config_fn(
            train_input_source=processed_input_source,
            #input_partition_type = DataPartitionType.DATA,
            evalers=evaler_config_dict(_eval_input_sources()),
            **kwargs,
        )
    # Make a variant of fuji-7B that can run on a single machine with 8 80G GPUs.
    # pytype: disable=annotation-type-mismatch
    cfg: SpmdTrainer.Config = config_map["fuji-7B"]().clone()
    # pytype: enable=annotation-type-mismatch
    cfg.input.batcher.global_batch_size = 32
    for evaler in cfg.evalers.values():
        evaler.input.batcher.global_batch_size = 32
    config_map["fuji-7B-single"] = lambda: cfg
    return config_map
