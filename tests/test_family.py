from assay import family
from assay.publish import family_section


def test_every_published_model_is_in_the_family():
    names = [m.name for m in family.FAMILY]
    assert len(names) == len(set(names))
    assert all(m.run and m.backbone and m.size for m in family.FAMILY)


def test_tables_have_one_row_per_model(tmp_path):
    short = family.short_table().splitlines()
    full = family.full_table().splitlines()
    assert len(short) == len(family.FAMILY) + 2  # header and separator
    assert len(full) == len(family.FAMILY) + 2
    for m in family.FAMILY:
        assert f"[{m.name}]" in family.short_table()


def test_missing_runs_show_a_dash(tmp_path):
    """A machine without the run directories still renders the table, visibly empty."""
    assert family.cells(str(tmp_path), "holdout") == "-"
    assert family.abstention(str(tmp_path)) == "-"
    assert family.latency(str(tmp_path)) == "-"


def test_the_card_section_carries_the_same_table():
    card = family_section()
    assert family.short_table() in card
    assert "## Serving" in card and "## Train one on your own data" in card
