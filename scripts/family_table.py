"""Write docs/models.md from the run directories.

    uv run python scripts/family_table.py > docs/models.md

The family itself lives in `assay.family`, which the model cards read as well.
"""

from assay.family import document

if __name__ == "__main__":
    print(document(), end="")
