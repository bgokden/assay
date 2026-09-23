"""Answer the same questions through each runtime, and check they agree.

    python examples/backends.py --runtime transformers --model runs/assay-0.6b
    python examples/backends.py --runtime llamacpp --url http://127.0.0.1:8081 \\
        --model runs/assay-0.6b
    python examples/backends.py --runtime sglang --url http://127.0.0.1:30000 \\
        --model Berk/assay-0.6b

A model is the weights plus a temperature, an evidence head and a set of conformal
thresholds. Only the weights move to another runtime; the rest runs in the client, which is
why the same model gives the same answers wherever it is served -- and why `--verify` is worth
running against a deployment before trusting it.

llama.cpp needs a GGUF and a server started with the embedding flags:

    python -m assay.publish --run runs/assay-0.6b --repo local/assay --dry-run
    python convert_hf_to_gguf.py runs/assay-0.6b/hub --outfile assay.gguf --outtype f16
    llama-server -m assay.gguf --port 8081 -c 4096 \\
        --embeddings --pooling last --embd-normalize -1

SGLang needs the hidden-states flag, which is not optional here -- the evidence head reads
them, and a server started without it rejects the request:

    python -m sglang.launch_server --model-path Berk/assay-0.6b --port 30000 \\
        --enable-return-hidden-states
"""

import argparse
import time

from assay.schema import Question

STATE = {
    "channel": "email",
    "message": (
        "I was charged twice for order A-104 last week and the second charge is still there. "
        "We cannot close the books until this is fixed."
    ),
}

QUESTIONS = {
    "route": Question(
        type="choice",
        instructions="Which team should handle this ticket?",
        options={
            "billing": "Charges, invoices and refunds",
            "technical": "Faults, errors and outages",
            "account": "Sign-in, passwords and profile changes",
        },
    ),
    "refund_requested": Question(
        type="bool", instructions="Is the customer asking for money back?"
    ),
    "urgency": Question(
        type="score",
        instructions="How urgent is this ticket?",
        levels=["Can wait", "Needs attention today", "Blocking work right now"],
    ),
}


def client_for(runtime: str, model: str, url: str, device: str):
    if runtime == "transformers":
        from assay import load_model

        return load_model(model, device=device)
    if runtime == "llamacpp":
        from assay.backends.llamacpp import LlamaCppClient

        return LlamaCppClient(url, model)
    from assay.backends.sglang import SGLangClient

    return SGLangClient(url, model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", choices=["transformers", "llamacpp", "sglang"], required=True)
    ap.add_argument("--model", required=True, help="a run directory or a Hub id")
    ap.add_argument("--url", help="where the server is, for llamacpp and sglang")
    ap.add_argument("--device", default="cuda", help="transformers only")
    ap.add_argument(
        "--verify", action="store_true", help="also compare the runtime with local transformers"
    )
    args = ap.parse_args()
    if args.runtime != "transformers" and not args.url:
        ap.error(f"--url is required for {args.runtime}")

    client = client_for(args.runtime, args.model, args.url, args.device)
    started = time.perf_counter()
    answers = client.answer(STATE, QUESTIONS)
    elapsed = (time.perf_counter() - started) * 1000.0

    print(f"{args.runtime}, {len(QUESTIONS)} questions, {elapsed:.0f} ms\n")
    for name, answer in answers.items():
        distribution = ", ".join(f"{k} {v:.3f}" for k, v in answer.probabilities.items())
        print(f"  {name:17s} {answer.argmax:<12} {distribution}")
        print(f"  {'':17s} confidence {answer.confidence:.2f}, evidence {answer.evidence:.2f}")

    if args.verify:
        if args.runtime == "transformers":
            print("\nnothing to verify: this is the reference runtime")
            return
        module = (
            "assay.backends.llamacpp" if args.runtime == "llamacpp" else "assay.backends.sglang"
        )
        verify = __import__(module, fromlist=["verify_against_local"]).verify_against_local
        print("\ncomparing with local transformers...")
        print(f"  {verify(args.url, args.model, STATE, QUESTIONS)}")


if __name__ == "__main__":
    main()
