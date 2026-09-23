"""transformers 5 writes artifacts that transformers 4 reads wrongly, or not at all.

Most people are still on 4, and both SGLang and llama.cpp's converter pin it, so a model
published from this repository has to carry a config both generations load. This is the
regression test for that: the published repositories were broken by exactly this.
"""

import json

from assay.compat import (
    normalise_model_config,
    normalise_saved_model,
    normalise_tokenizer_config,
)

SPECIALS = ["<|im_start|>", "<|im_end|>", "<|vision_pad|>"]


def write(tmp_path, config):
    path = tmp_path / "tokenizer_config.json"
    path.write_text(json.dumps(config))
    return str(tmp_path)


def read(directory):
    with open(f"{directory}/tokenizer_config.json") as f:
        return json.load(f)


def test_a_list_moves_to_the_name_both_versions_read(tmp_path):
    directory = write(
        tmp_path, {"tokenizer_class": "Qwen2Tokenizer", "extra_special_tokens": SPECIALS}
    )
    assert normalise_tokenizer_config(directory) is True
    config = read(directory)
    assert "extra_special_tokens" not in config
    assert config["additional_special_tokens"] == SPECIALS
    assert config["tokenizer_class"] == "Qwen2Tokenizer"  # nothing else is touched


def test_existing_tokens_are_kept_and_not_duplicated(tmp_path):
    directory = write(
        tmp_path,
        {
            "additional_special_tokens": ["<|im_start|>", "<|custom|>"],
            "extra_special_tokens": SPECIALS,
        },
    )
    assert normalise_tokenizer_config(directory) is True
    assert read(directory)["additional_special_tokens"] == [
        "<|im_start|>",
        "<|custom|>",
        "<|im_end|>",
        "<|vision_pad|>",
    ]


def test_a_mapping_is_left_alone(tmp_path):
    """transformers 4 writes a mapping there and means something else by it."""
    config = {"extra_special_tokens": {"bos_token": "<s>"}}
    directory = write(tmp_path, config)
    assert normalise_tokenizer_config(directory) is False
    assert read(directory) == config


def test_a_directory_without_a_tokenizer_is_fine(tmp_path):
    assert normalise_tokenizer_config(str(tmp_path)) is False


def write_model_config(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return str(tmp_path)


def read_model_config(directory):
    with open(f"{directory}/config.json") as f:
        return json.load(f)


def test_rope_theta_is_restored_for_transformers_4(tmp_path):
    """The bug this catches is silent: transformers 4 falls back to the class default, 10000
    against a true 1000000, and serves a different model without saying so. Measured through
    SGLang on assay-0.6b before the fix: readout hidden state cosine 0.9940, probabilities out
    by 7.1e-2."""
    directory = write_model_config(
        tmp_path,
        {
            "model_type": "qwen3",
            "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
        },
    )
    assert normalise_model_config(directory) is True
    config = read_model_config(directory)
    assert config["rope_theta"] == 1000000
    # the new spelling stays, so transformers 5 is unaffected
    assert config["rope_parameters"] == {"rope_theta": 1000000, "rope_type": "default"}


def test_a_scaled_rope_also_carries_the_old_spelling(tmp_path):
    directory = write_model_config(
        tmp_path,
        {"rope_parameters": {"rope_theta": 500000, "rope_type": "yarn", "factor": 4.0}},
    )
    assert normalise_model_config(directory) is True
    config = read_model_config(directory)
    assert config["rope_theta"] == 500000
    assert config["rope_scaling"] == {"rope_type": "yarn", "factor": 4.0}


def test_a_nested_backbone_config_is_reached(tmp_path):
    directory = write_model_config(
        tmp_path,
        {"model_type": "hybrid", "text_config": {"rope_parameters": {"rope_theta": 10000000}}},
    )
    assert normalise_model_config(directory) is True
    assert read_model_config(directory)["text_config"]["rope_theta"] == 10000000


def test_an_existing_rope_theta_is_left_alone(tmp_path):
    directory = write_model_config(
        tmp_path, {"rope_theta": 123.0, "rope_parameters": {"rope_theta": 456.0}}
    )
    assert normalise_model_config(directory) is False
    assert read_model_config(directory)["rope_theta"] == 123.0


def test_a_transformers_4_config_is_untouched(tmp_path):
    directory = write_model_config(tmp_path, {"model_type": "qwen3", "rope_theta": 1000000})
    assert normalise_model_config(directory) is False


def test_a_missing_model_config_is_not_an_error(tmp_path):
    assert normalise_model_config(str(tmp_path)) is False


def test_normalise_saved_model_reports_both_files(tmp_path):
    """Publishing goes through this, so a new publish path cannot miss one of the two."""
    write(tmp_path, {"extra_special_tokens": SPECIALS})
    write_model_config(tmp_path, {"rope_parameters": {"rope_theta": 1000000}})
    assert normalise_saved_model(str(tmp_path)) == ["tokenizer_config.json", "config.json"]
    assert normalise_saved_model(str(tmp_path)) == []


def test_a_per_attention_type_rope_block_is_translated(tmp_path):
    """ModernBERT gives each attention type its own theta; transformers 4 reads them as two
    fields. Our published compiled-base happens to match the class defaults, so this is the
    test that stops a retuned theta from being silently dropped."""
    directory = write_model_config(
        tmp_path,
        {
            "model_type": "modernbert",
            "rope_parameters": {
                "full_attention": {"rope_theta": 160000.0, "rope_type": "default"},
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
            },
        },
    )
    assert normalise_model_config(directory) is True
    config = read_model_config(directory)
    assert config["global_rope_theta"] == 160000.0
    assert config["local_rope_theta"] == 10000.0
    assert "rope_theta" not in config


def test_the_generic_tokenizer_class_is_renamed(tmp_path):
    """transformers 5 saves a tokenizer with no concrete class as TokenizersBackend, which 4
    rejects outright. Only the ModernBERT tier hit this; the Qwen3 tokenizers keep a real
    class name and were unaffected."""
    directory = write(tmp_path, {"tokenizer_class": "TokenizersBackend", "backend": "tokenizers"})
    assert normalise_tokenizer_config(directory) is True
    assert read(directory)["tokenizer_class"] == "PreTrainedTokenizerFast"


def test_a_concrete_tokenizer_class_is_left_alone(tmp_path):
    directory = write(tmp_path, {"tokenizer_class": "Qwen2Tokenizer"})
    assert normalise_tokenizer_config(directory) is False
    assert read(directory)["tokenizer_class"] == "Qwen2Tokenizer"


def test_both_tokenizer_fixes_apply_together(tmp_path):
    directory = write(
        tmp_path, {"tokenizer_class": "TokenizersBackend", "extra_special_tokens": SPECIALS}
    )
    assert normalise_tokenizer_config(directory) is True
    config = read(directory)
    assert config["tokenizer_class"] == "PreTrainedTokenizerFast"
    assert config["additional_special_tokens"] == SPECIALS
    assert "extra_special_tokens" not in config
