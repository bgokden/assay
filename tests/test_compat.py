"""transformers 5 writes a tokenizer config that transformers 4 cannot read.

Most people are still on 4, and both SGLang and llama.cpp's converter pin it, so a model
published from this repository has to carry a config both generations load. This is the
regression test for that: the published repositories were broken by exactly this.
"""

import json

from assay.compat import normalise_tokenizer_config

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
