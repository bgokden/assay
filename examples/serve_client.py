"""Call a running Assay server through both request shapes.

Start the server first:

    python -m assay.server --model Berk/assay-0.6b --port 8000

then

    python examples/serve_client.py --url http://127.0.0.1:8000

`/v1/decide` is the native shape; `/v1/systemone` accepts the shape other open decision
models take (`criteria` for options, `noul` for a boolean), so the same deployment answers
clients written for either. `/v1/decide_graph` walks a tree in one pass.
"""

import argparse
import json

import httpx

STATE = "I was charged twice for order A-104 and nobody has replied for three days."

NATIVE = {
    "state": STATE,
    "questions": {
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "options": {"billing": "Charges and refunds", "technical": "Faults and outages"},
        },
        "refund_requested": {
            "type": "bool",
            "instructions": "Is the customer asking for money back?",
        },
    },
}

SYSTEM_ONE = {
    "model": "assay",
    "state": STATE,
    "questions": {
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {"billing": "Charges and refunds", "technical": "Faults and outages"},
        },
        "refund_requested": {
            "type": "noul",
            "instructions": "Is the customer asking for money back?",
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this ticket?",
            "criteria": ["Can wait", "Needs attention today", "Blocking work right now"],
        },
    },
}


def post(client: httpx.Client, url: str, path: str, body: dict) -> dict:
    response = client.post(url + path, json=body, timeout=120.0)
    response.raise_for_status()
    return response.json()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key", help="sent as a bearer token when the server requires one")
    args = ap.parse_args()
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}

    with httpx.Client(headers=headers) as client:
        print("models:", client.get(args.url + "/v1/models").json())

        native = post(client, args.url, "/v1/decide", NATIVE)
        print("\n/v1/decide")
        print(json.dumps(native["answers"], indent=2))

        other = post(client, args.url, "/v1/systemone", SYSTEM_ONE)
        print("\n/v1/systemone")
        print(json.dumps({"choices": other["choices"], "nouls": other["nouls"]}, indent=2))

        batch = post(
            client,
            args.url,
            "/v1/systemone/batch",
            {
                "requests": [
                    SYSTEM_ONE,
                    {**SYSTEM_ONE, "state": "The dashboard is down for everyone."},
                ]
            },
        )
        print("\n/v1/systemone/batch usage:", json.dumps(batch["usage"]))


if __name__ == "__main__":
    main()
