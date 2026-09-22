"""Calibrated typed decisions from one forward pass.

    from assay import Agent, load_model
    from assay.schema import Question

    model = load_model("Berk/assay-0.6b")
    answers = model.answer("state text", {"q": Question(type="bool", instructions="...")})

    agent = Agent.from_file("examples/agents/support_triage.json")
    run = agent.run(model, "state text", handlers={"refund": my_refund_function})

`load_model` reads the tier from the saved configuration, so the decoder, encoder and
encoder-decoder models all load, serve and publish the same way. An `Agent` is a decision
graph plus the actions its outcomes stand for; it decides, and the caller acts.
"""

from assay.agent import Agent
from assay.evaluate import load_model

__all__ = ["Agent", "load_model"]
