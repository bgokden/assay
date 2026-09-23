"""The examples are documentation that runs, so they break like code and are tested like it.

Nothing here loads a model: these check that each script imports, that its arguments parse,
and that the data it ships with is valid, which is what breaks when an interface moves.
"""

import importlib.util
import json
import pathlib
import sys

import pytest

EXAMPLES = pathlib.Path("examples")
SCRIPTS = sorted(p for p in EXAMPLES.glob("*.py"))


def load(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(f"example_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_there_are_examples():
    assert {p.name for p in SCRIPTS} >= {
        "quickstart.py",
        "decision_graph.py",
        "run_agent.py",
        "serve_client.py",
        "support_data.py",
    }


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_an_example_imports_and_documents_itself(path):
    module = load(path)
    assert module.__doc__ and len(module.__doc__.splitlines()) > 1, path.name
    assert hasattr(module, "main"), path.name


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_an_example_parses_its_own_help(path, monkeypatch, capsys):
    """--help exercises the argument parser, which is where a renamed flag shows up."""
    module = load(path)
    monkeypatch.setattr(sys, "argv", [path.name, "--help"])
    with pytest.raises(SystemExit) as exit_info:
        module.main()
    assert exit_info.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_the_pipeline_configurations_are_valid():
    from assay.pipeline import TIERS, validate

    configs = sorted(EXAMPLES.glob("pipelines/*.json"))
    assert len(configs) == 3
    tiers = set()
    for path in configs:
        with open(path) as f:
            config = json.load(f)
        tiers.add(config["tier"])
        if not pathlib.Path(config["data"]).exists():
            pytest.skip("run examples/support_data.py to generate the example dataset")
        validate(config)
    assert tiers == set(TIERS)  # one worked example per tier


def test_the_shipped_agents_load():
    from assay.agent import load_agents

    agents = load_agents(str(EXAMPLES / "agents"))
    assert len(agents) >= 3
