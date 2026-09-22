"""Encoder-decoder decision model: the state is encoded once, each question is a decoder pass.

The answer is read from the decoder's next-token logits over single-token option labels, the
same readout the decoder models use, but the state lives in the encoder's output rather than
in the prompt. Several questions over one state therefore share one encoder pass, which is
the cost property the compiled tier was built for, without a learned scoring head.

    state ---- encoder ----> memory (L x d)                 once per state
    question --decoder ----> logits at the last position    once per question, cross-attends
    answer = softmax(logits[option label ids])              no generation

`decoder_question=False` gives the conventional arrangement instead (state and question both
in the encoder, the decoder reads one start token), which is how instruction-tuned seq2seq
models were trained and is the baseline this design has to beat.
"""

from __future__ import annotations

import json
import os
from typing import Any

import safetensors.torch
import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from assay.encoding import render_question
from assay.labels import LabelAlphabet
from assay.schema import Answer, Question, make_answer, render_state

CONFIG_FILE = "assay_seq2seq_config.json"
WEIGHTS_FILE = "assay_seq2seq.safetensors"


class Seq2SeqModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        model_id: str,
        temperature: float = 1.0,
        decoder_question: bool = True,
        alphabet_size: int = 255,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.alphabet = LabelAlphabet(tokenizer, size=alphabet_size)
        self.decoder_question = decoder_question
        self.temperature = temperature
        self.max_state_tokens = 512
        self.max_question_tokens = 384
        self.start_id = tokenizer.bos_token_id
        d = model.config.decoder.get_text_config().hidden_size
        self.d = d
        self.evidence_head = nn.Linear(d, 1)
        nn.init.zeros_(self.evidence_head.weight)
        nn.init.zeros_(self.evidence_head.bias)
        self.hybrid = False
        self.pooling = "last"  # for trainer compatibility

    # --- construction ---------------------------------------------------------------------

    @classmethod
    def from_encoder(
        cls,
        model_id: str,
        device: str = "cuda",
        lora_r: int | None = None,
        decoder_question: bool = True,
        **kw,
    ) -> Seq2SeqModel:
        """Named to match the other tiers so the same trainer can build any of them."""
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.bfloat16 if lora_r else torch.float32
        # transformers 5 keeps per-module dtypes from the checkpoint, which mixes bf16 and
        # fp32 inside one model; force one dtype so the heads and the backbone agree
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=dtype).to(dtype)
        if lora_r:
            from peft import LoraConfig, get_peft_model

            model = get_peft_model(
                model,
                LoraConfig(
                    r=lora_r,
                    lora_alpha=2 * lora_r,
                    lora_dropout=0.05,
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
                ),
            )
        return cls(model, tokenizer, model_id, decoder_question=decoder_question, **kw).to(device)

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cuda") -> Seq2SeqModel:
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download

            path = snapshot_download(path)
        with open(os.path.join(path, CONFIG_FILE)) as f:
            config = json.load(f)
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSeq2SeqLM.from_pretrained(path, dtype=torch.float32).to(torch.float32)
        out = cls(
            model,
            tokenizer,
            config["model_id"],
            temperature=config.get("temperature", 1.0),
            decoder_question=config.get("decoder_question", True),
        )
        out.load_state_dict(
            safetensors.torch.load_file(os.path.join(path, WEIGHTS_FILE)), strict=False
        )
        out.eval()
        return out.to(device)

    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        model = (
            self.model.merge_and_unload() if hasattr(self.model, "merge_and_unload") else self.model
        )
        self.model = model
        model.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)
        head = {
            k: v.detach().cpu() for k, v in self.state_dict().items() if not k.startswith("model.")
        }
        safetensors.torch.save_file(head, os.path.join(path, WEIGHTS_FILE))
        with open(os.path.join(path, CONFIG_FILE), "w") as f:
            json.dump(
                {
                    "model_id": self.model_id,
                    "temperature": self.temperature,
                    "decoder_question": self.decoder_question,
                    "arch": "seq2seq",
                },
                f,
                indent=2,
            )

    @property
    def device(self) -> torch.device:
        return self.evidence_head.weight.device

    def enable_gradient_checkpointing(self) -> None:
        self.model.gradient_checkpointing_enable()

    # --- forward --------------------------------------------------------------------------

    def _tokenize(self, texts: list[str], max_tokens: int, add_special: bool = True):
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_tokens,
            add_special_tokens=add_special,
            return_tensors="pt",
        )
        return batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device)

    def _decoder_inputs(self, questions: list[Question]):
        """Decoder input ids ending at the readout position, left-padded so the last column is
        every question's readout position."""
        if not self.decoder_question:
            ids = torch.full((len(questions), 1), self.start_id, device=self.device)
            return ids, torch.ones_like(ids)
        texts = [render_question(q, self.alphabet, list(range(len(q.keys)))) for q in questions]
        encoded = [
            [self.start_id]
            + self.tokenizer.encode(t, add_special_tokens=False)[-(self.max_question_tokens - 1) :]
            for t in texts
        ]
        width = max(len(e) for e in encoded)
        ids = torch.full((len(encoded), width), self.tokenizer.pad_token_id, device=self.device)
        mask = torch.zeros((len(encoded), width), dtype=torch.long, device=self.device)
        for i, e in enumerate(encoded):  # left pad: readout is always the last column
            ids[i, width - len(e) :] = torch.tensor(e, device=self.device)
            mask[i, width - len(e) :] = 1
        return ids, mask

    def forward(self, states: list[Any], questions: list[Question]):
        if self.decoder_question:
            enc_ids, enc_mask = self._tokenize(
                [render_state(s) for s in states], self.max_state_tokens
            )
        else:
            texts = [
                render_state(s) + "\n" + render_question(q, self.alphabet, list(range(len(q.keys))))
                for s, q in zip(states, questions)
            ]
            enc_ids, enc_mask = self._tokenize(
                texts, self.max_state_tokens + self.max_question_tokens
            )
        dec_ids, dec_mask = self._decoder_inputs(questions)
        out = self.model(
            input_ids=enc_ids,
            attention_mask=enc_mask,
            decoder_input_ids=dec_ids,
            decoder_attention_mask=dec_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        logits = out.logits[:, -1, :].float()
        hidden = out.decoder_hidden_states[-1][:, -1].to(self.evidence_head.weight.dtype)
        k_max = max(len(q.keys) for q in questions)
        option_logits = logits.new_full((len(questions), k_max), float("-inf"))
        for i, q in enumerate(questions):
            ids = (
                [self.alphabet.yes_id, self.alphabet.no_id]
                if q.type == "bool"
                else self.alphabet.token_ids[: len(q.keys)]
            )
            option_logits[i, : len(ids)] = logits[i, torch.tensor(ids, device=logits.device)]
        evidence = self.evidence_head(hidden).squeeze(-1)
        return option_logits, evidence

    # --- inference ------------------------------------------------------------------------

    @torch.no_grad()
    def answer(self, state: Any, questions: dict[str, Question]) -> dict[str, Answer]:
        names = list(questions)
        qs = [questions[n] for n in names]
        logits, evidence = self.forward([state] * len(qs), qs)
        probs = torch.softmax(logits / self.temperature, dim=-1)
        ev = torch.sigmoid(evidence)
        return {
            n: make_answer(q, probs[i, : len(q.keys)].tolist(), ev[i].item())
            for i, (n, q) in enumerate(zip(names, qs))
        }
