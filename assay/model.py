"""The decision model: a causal LM backbone (optionally with a LoRA adapter) whose next-token
logits over option label tokens are the answer distribution, plus an evidence head."""

from __future__ import annotations

import dataclasses
import json
import math
import os
from typing import Any

import safetensors.torch
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from assay.encoding import Batch, Packed, collate, encode, identity_order
from assay.labels import LabelAlphabet
from assay.schema import Answer, Question, make_answer
from assay.tiers import CONFIG_FILES, WEIGHTS_FILES

HEAD_FILE = WEIGHTS_FILES["decoder"]
CONFIG_FILE = CONFIG_FILES["decoder"]


@dataclasses.dataclass
class ModelOutput:
    option_logits: torch.Tensor  # (Q, K_max), -inf beyond each question's options
    evidence_logits: torch.Tensor  # (Q,)


def adapter_source(path: str, adapter: str) -> str:
    """Where the LoRA adapter lives, given the path recorded in the config.

    "." means the repository root, which is where PEFT and the Hub both expect it. The Hub
    counts a model's downloads from a query file, and for a peft repository that file is
    `adapter_config.json` at the root exactly -- an adapter under `adapter/` is never
    counted, so such a repository reports no downloads at all rather than reporting few.
    """
    return path if adapter in (".", "") else os.path.join(path, adapter)


class AssayModel(nn.Module):
    def __init__(
        self,
        lm: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        base_model_id: str,
        temperature: float = 1.0,
        normalize_evidence_input: bool = True,
        quantized_base: str | None = None,
        content_term: bool = False,
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
        # content-scored option term: a bilinear score between the decision vector and each
        # option line's pooled hidden state, added to the label-token logit. Zero at start, so
        # a fresh model answers exactly as the label readout alone.
        self.content_proj = nn.Linear(hidden, hidden, bias=False) if content_term else None
        if self.content_proj is not None:
            nn.init.zeros_(self.content_proj.weight)
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
        content_term: bool = False,
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

            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                bias="none",
                task_type="CAUSAL_LM",
            )
            lm = get_peft_model(lm, config)
        model = cls(
            lm, tokenizer, base_model_id, quantized_base=quantization, content_term=content_term
        )
        model.heads().to(dtype=torch.float32)
        if quantization is not None or max_memory is not None:
            model.heads().to(device)
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
                else BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
                )
            )
            kwargs["device_map"] = {"": device}
        lm = AutoModelForCausalLM.from_pretrained(weights_source, **kwargs)
        if config.get("adapter"):
            from peft import PeftModel

            lm = PeftModel.from_pretrained(lm, adapter_source(path, config["adapter"]))
        model = cls(
            lm,
            tokenizer,
            base_model_id,
            temperature=config.get("temperature", 1.0),
            normalize_evidence_input=config.get("normalize_evidence_input", False),
            quantized_base=quantized_base,
            content_term=config.get("content_term", False),
        )
        model.load_heads(safetensors.torch.load_file(os.path.join(path, HEAD_FILE)))
        model.heads().to(dtype=torch.float32)
        if quantized_base:
            model.heads().to(device)
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
            "content_term": self.content_proj is not None,
        }
        if hasattr(self.lm, "peft_config"):
            self.lm.save_pretrained(os.path.join(path, "adapter"))
            config["adapter"] = "adapter"
        safetensors.torch.save_file(
            {k: v.detach().cpu() for k, v in self.head_state_dict().items()},
            os.path.join(path, HEAD_FILE),
        )
        with open(os.path.join(path, CONFIG_FILE), "w") as f:
            json.dump(config, f, indent=2)

    # --- heads (everything trained outside the backbone) --------------------------------

    def heads(self) -> nn.ModuleList:
        modules = [self.evidence_head]
        if self.content_proj is not None:
            modules.append(self.content_proj)
        return nn.ModuleList(modules)

    def head_state_dict(self) -> dict[str, torch.Tensor]:
        """Evidence head under its bare keys (as saved since the first release), content
        projection under a prefix."""
        state = dict(self.evidence_head.state_dict())
        if self.content_proj is not None:
            state.update(
                {f"content_proj.{k}": v for k, v in self.content_proj.state_dict().items()}
            )
        return state

    def load_heads(self, state: dict[str, torch.Tensor]) -> None:
        content = {
            k[len("content_proj.") :]: v for k, v in state.items() if k.startswith("content_proj.")
        }
        evidence = {k: v for k, v in state.items() if not k.startswith("content_proj.")}
        self.evidence_head.load_state_dict(evidence)
        if self.content_proj is not None:
            self.content_proj.load_state_dict(content)
        elif content:
            raise ValueError(
                "head file has a content projection but the config has no content_term"
            )

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
        head_device = self.evidence_head.weight.device
        head_input = decision.float().to(head_device)
        if self.content_proj is not None:
            option_logits = option_logits + self._content_scores(hidden, head_input, batch).to(
                option_logits.device
            )
        option_logits = option_logits.masked_fill(~valid, float("-inf"))
        if self.normalize_evidence_input:
            head_input = torch.nn.functional.layer_norm(head_input, head_input.shape[-1:])
        evidence_logits = self.evidence_head(head_input).squeeze(-1)
        evidence_logits = evidence_logits.to(option_logits.device)
        return ModelOutput(option_logits=option_logits, evidence_logits=evidence_logits)

    def _content_scores(
        self, hidden: torch.Tensor, decision: torch.Tensor, batch: Batch
    ) -> torch.Tensor:
        """(Q, K_max) bilinear scores between each question's decision vector and the mean
        hidden state over each option's line; zero where an option has no line."""
        spans = batch.q_option_spans.to(hidden.device)  # (Q, K, 2)
        present = spans[..., 0] >= 0
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        inside = (positions >= spans[..., :1]) & (positions < spans[..., 1:])  # (Q, K, L)
        lengths = (spans[..., 1] - spans[..., 0]).clamp(min=1).unsqueeze(-1)
        weights = inside.to(hidden.dtype) / lengths.to(hidden.dtype)
        pooled = torch.bmm(weights, hidden[batch.q_batch_index])  # (Q, K, H)
        size = hidden.shape[-1]
        pooled = torch.nn.functional.layer_norm(pooled.float().to(decision.device), (size,))
        query = self.content_proj(torch.nn.functional.layer_norm(decision, (size,)))
        scores = torch.einsum("qh,qkh->qk", query, pooled) / math.sqrt(size)
        return scores.masked_fill(~present.to(scores.device), 0.0)

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
            from assay.prefix import PREFIX_MIN_QUESTIONS, prefix_answers

            if len(qs) >= PREFIX_MIN_QUESTIONS:
                return dict(zip(names, prefix_answers(self, state, qs, max_state_tokens)))
            packs = [
                encode(
                    self.tokenizer,
                    self.alphabet,
                    state,
                    [q],
                    orders=[identity_order(q)],
                    max_state_tokens=max_state_tokens,
                )
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
        from assay.prefix import PREFIX_MIN_QUESTIONS, prefix_logits

        if self.hybrid and all(len(p.questions) < PREFIX_MIN_QUESTIONS for p in packed):
            # few questions per request: one sequence each, batched by collate, which beats
            # paying for a separate state pass (see PREFIX_MIN_QUESTIONS)
            batch = collate(packed, self.tokenizer.pad_token_id).to(self.device)
            out = self.forward(batch)
            option_logits, evidence_logits = out.option_logits, out.evidence_logits
        elif self.hybrid:
            # a block mask cannot isolate questions here, so a request with enough questions
            # runs its own state once and answers them from that cache
            parts = [prefix_logits(self, p) for p in packed]
            width = max(part[0].shape[1] for part in parts)
            option_logits = torch.cat(
                [
                    torch.nn.functional.pad(
                        part[0], (0, width - part[0].shape[1]), value=float("-inf")
                    )
                    for part in parts
                ]
            )
            evidence_logits = torch.cat([part[1] for part in parts])
        else:
            batch = collate(packed, self.tokenizer.pad_token_id).to(self.device)
            out = self.forward(batch)
            option_logits, evidence_logits = out.option_logits, out.evidence_logits
        probs = torch.softmax(option_logits / self.temperature, dim=-1)
        evidence = torch.sigmoid(evidence_logits)
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
