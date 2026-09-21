"""The decision model: a causal LM backbone (optionally with a LoRA adapter) whose next-token
logits over option label tokens are the answer distribution, plus an evidence head."""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any

import safetensors.torch
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from assay.encoding import Batch, Packed, collate, encode, identity_order
from assay.labels import LabelAlphabet
from assay.schema import Answer, Question, make_answer

HEAD_FILE = "assay_head.safetensors"
CONFIG_FILE = "assay_config.json"


@dataclasses.dataclass
class ModelOutput:
    option_logits: torch.Tensor  # (Q, K_max), -inf beyond each question's options
    evidence_logits: torch.Tensor  # (Q,)


class AssayModel(nn.Module):
    def __init__(
        self,
        lm: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        base_model_id: str,
        temperature: float = 1.0,
        normalize_evidence_input: bool = True,
        quantized_base: str | None = None,
    ) -> None:
        super().__init__()
        self.lm = lm
        self.tokenizer = tokenizer
        self.base_model_id = base_model_id
        self.quantized_base = quantized_base  # "4bit"/"8bit" when trained as QLoRA
        # layer-normalise the decision vector before the evidence head so the head's scale
        # does not depend on the backbone's hidden-state norm (older models were saved without)
        self.normalize_evidence_input = normalize_evidence_input
        self.alphabet = LabelAlphabet(tokenizer)
        config = self._backbone().config
        hidden = config.hidden_size
        # hybrid backbones (linear-attention layers) carry state along the sequence, so the
        # block mask cannot isolate packed questions: they are answered one per sequence
        self.hybrid = "linear_attention" in set(getattr(config, "layer_types", None) or [])
        self.evidence_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.evidence_head.weight)
        nn.init.zeros_(self.evidence_head.bias)
        self.temperature = temperature

    # --- construction -------------------------------------------------------------------

    @classmethod
    def from_base(
        cls,
        base_model_id: str,
        lora_r: int | None = None,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        quantization: str | None = None,
        max_memory: dict[str | int, str] | None = None,
    ) -> AssayModel:
        """quantization: None, "8bit" or "4bit" (bitsandbytes, nn.Linear weights only).
        max_memory: when given, layers are spread over the listed devices (accelerate), for
        example {0: "28GiB", "cpu": "50GiB"} to run a model larger than the GPU."""
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        kwargs: dict[str, Any] = {"dtype": dtype, "attn_implementation": "sdpa"}
        if quantization is not None:
            from transformers import BitsAndBytesConfig

            if quantization == "8bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            elif quantization == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
                )
            else:
                raise ValueError(f"unknown quantization {quantization!r}")
            kwargs["device_map"] = {"": device}
        if max_memory is not None:
            kwargs["device_map"] = "auto"
            kwargs["max_memory"] = max_memory
        lm = AutoModelForCausalLM.from_pretrained(base_model_id, **kwargs)
        if lora_r is not None:
            from peft import LoraConfig, get_peft_model

            if quantization is not None:
                from peft import prepare_model_for_kbit_training

                lm = prepare_model_for_kbit_training(lm, use_gradient_checkpointing=False)
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            lm = get_peft_model(lm, config)
        model = cls(lm, tokenizer, base_model_id, quantized_base=quantization)
        model.evidence_head.to(dtype=torch.float32)
        if quantization is not None or max_memory is not None:
            model.evidence_head.to(device)
            return model
        return model.to(device)

    @classmethod
    def from_pretrained(
        cls, path: str, dtype: torch.dtype = torch.bfloat16, device: str = "cuda"
    ) -> AssayModel:
        """Load a local directory saved by `save_pretrained`, or a Hub repository published by
        `assay.publish` (merged weights or base + adapter, plus the evidence head)."""
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download

            path = snapshot_download(path)
        with open(os.path.join(path, CONFIG_FILE)) as f:
            config = json.load(f)
        base_model_id = config["base_model_id"]
        weights_source = path if config.get("merged_from") else base_model_id
        tokenizer = AutoTokenizer.from_pretrained(weights_source)
        kwargs: dict[str, Any] = {"dtype": dtype, "attn_implementation": "sdpa"}
        quantized_base = config.get("quantized_base")
        if quantized_base:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = (
                BitsAndBytesConfig(load_in_8bit=True)
                if quantized_base == "8bit"
                else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
            )
            kwargs["device_map"] = {"": device}
        lm = AutoModelForCausalLM.from_pretrained(weights_source, **kwargs)
        if config.get("adapter"):
            from peft import PeftModel

            lm = PeftModel.from_pretrained(lm, os.path.join(path, config["adapter"]))
        model = cls(
            lm,
            tokenizer,
            base_model_id,
            temperature=config.get("temperature", 1.0),
            normalize_evidence_input=config.get("normalize_evidence_input", False),
            quantized_base=quantized_base,
        )
        head_state = safetensors.torch.load_file(os.path.join(path, HEAD_FILE))
        model.evidence_head.load_state_dict(head_state)
        model.evidence_head.to(dtype=torch.float32)
        if quantized_base:
            model.evidence_head.to(device)
            return model
        return model.to(device)

    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        config: dict[str, Any] = {
            "base_model_id": self.base_model_id,
            "temperature": self.temperature,
            "adapter": None,
            "normalize_evidence_input": self.normalize_evidence_input,
            "quantized_base": self.quantized_base,
        }
        if hasattr(self.lm, "peft_config"):
            self.lm.save_pretrained(os.path.join(path, "adapter"))
            config["adapter"] = "adapter"
        safetensors.torch.save_file(
            {k: v.detach().cpu() for k, v in self.evidence_head.state_dict().items()},
            os.path.join(path, HEAD_FILE),
        )
        with open(os.path.join(path, CONFIG_FILE), "w") as f:
            json.dump(config, f, indent=2)

    # --- forward ------------------------------------------------------------------------

    def _causal_lm(self) -> nn.Module:
        if hasattr(self.lm, "get_base_model"):
            return self.lm.get_base_model()
        return self.lm

    def _backbone(self) -> nn.Module:
        return self._causal_lm().model

    def _lm_head(self) -> nn.Module:
        return self._causal_lm().lm_head

    @property
    def device(self) -> torch.device:
        return next(self._backbone().parameters()).device

    def forward(self, batch: Batch) -> ModelOutput:
        if self.hybrid:
            if batch.num_questions != batch.input_ids.shape[0]:
                raise ValueError("hybrid backbones answer one question per sequence; do not pack")
            attention_mask: Any = {
                "full_attention": batch.attention_mask,
                "linear_attention": batch.padding_mask,
            }
        else:
            attention_mask = batch.attention_mask
        hidden = self._backbone()(
            input_ids=batch.input_ids,
            position_ids=batch.position_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state
        decision = hidden[batch.q_batch_index, batch.q_readout]  # (Q, H)
        logits = self._lm_head()(decision).float()  # (Q, V)
        option_ids = batch.q_option_ids.clamp(min=0)
        option_logits = logits.gather(1, option_ids)
        valid = batch.q_option_ids >= 0
        option_logits = option_logits.masked_fill(~valid, float("-inf"))
        head_device = self.evidence_head.weight.device
        head_input = decision.float().to(head_device)
        if self.normalize_evidence_input:
            head_input = torch.nn.functional.layer_norm(head_input, head_input.shape[-1:])
        evidence_logits = self.evidence_head(head_input).squeeze(-1)
        evidence_logits = evidence_logits.to(option_logits.device)
        return ModelOutput(option_logits=option_logits, evidence_logits=evidence_logits)

    # --- inference ----------------------------------------------------------------------

    @torch.no_grad()
    def answer(
        self,
        state: Any,
        questions: dict[str, Question],
        max_state_tokens: int = 2048,
    ) -> dict[str, Answer]:
        names = list(questions.keys())
        qs = [questions[n] for n in names]
        if self.hybrid:
            packs = [
                encode(self.tokenizer, self.alphabet, state, [q], orders=[identity_order(q)], max_state_tokens=max_state_tokens)
                for q in qs
            ]
            answers = [a[0] for a in self.answer_packed(packs, [[q] for q in qs])]
            return dict(zip(names, answers))
        packed = encode(
            self.tokenizer,
            self.alphabet,
            state,
            qs,
            orders=[identity_order(q) for q in qs],
            max_state_tokens=max_state_tokens,
        )
        answers = self.answer_packed([packed], [qs])[0]
        return dict(zip(names, answers))

    @torch.no_grad()
    def answer_packed(
        self, packed: list[Packed], questions: list[list[Question]]
    ) -> list[list[Answer]]:
        """Answer already-encoded requests; returns one list of Answers per request in question
        order (canonical option order regardless of display order)."""
        batch = collate(packed, self.tokenizer.pad_token_id).to(self.device)
        out = self.forward(batch)
        probs = torch.softmax(out.option_logits / self.temperature, dim=-1)
        evidence = torch.sigmoid(out.evidence_logits)
        results: list[list[Answer]] = []
        i = 0
        for qs in questions:
            answers = []
            for q in qs:
                k = len(q.keys)
                answers.append(make_answer(q, probs[i, :k].tolist(), evidence[i].item()))
                i += 1
            results.append(answers)
        return results
