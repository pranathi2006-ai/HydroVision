from __future__ import annotations

from pathlib import Path

import pytest

from training.prepare_yolo_dataset import prepare_output


def test_empty_output_directory_is_safe_to_reuse(tmp_path: Path) -> None:
    output = tmp_path / "hydrovision"
    output.mkdir()

    prepare_output(output, overwrite=False)

    assert output.is_dir()


def test_nonempty_output_requires_explicit_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "hydrovision"
    output.mkdir()
    existing = output / "manifest.json"
    existing.write_text("keep me")

    with pytest.raises(FileExistsError):
        prepare_output(output, overwrite=False)

    assert existing.read_text() == "keep me"
