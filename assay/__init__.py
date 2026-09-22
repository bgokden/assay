"""Calibrated typed decisions from one forward pass.

    from assay import load_model
    from assay.schema import Question

    model = load_model("Berk/assay-0.6b")
    answers = model.answer("state text", {"q": Question(type="bool", instructions="...")})

`load_model` reads the tier from the saved configuration, so the decoder, encoder and
encoder-decoder models all load, serve and publish the same way.
"""

from assay.evaluate import load_model

__all__ = ["load_model"]
