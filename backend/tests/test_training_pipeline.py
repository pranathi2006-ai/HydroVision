from __future__ import annotations

from pathlib import Path

import yaml

from training.prepare_yolo_dataset import (
    merge_source,
    read_names,
    require_complete_source,
    source_class_ids,
)


def make_source(root: Path, names: list[str], label: str) -> None:
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": names}))
    (root / "train" / "images" / "sample.jpg").write_bytes(b"image-placeholder")
    (root / "train" / "labels" / "sample.txt").write_text(label)


def test_class_lookup_accepts_dataset_spellings(tmp_path: Path) -> None:
    source = tmp_path / "source"
    make_source(source, ["OIL LEAKAGE", "rust"], "0 0.5 0.5 0.2 0.3\n")
    names = read_names(source)
    assert source_class_ids(names, "oil_leak") == {0}
    assert source_class_ids(names, "corrosion") == {1}
    assert source_class_ids(["leakage"], "oil_leak") == {0}


def test_merge_remaps_target_and_keeps_other_classes_as_negatives(tmp_path: Path) -> None:
    corrosion = tmp_path / "corrosion"
    output = tmp_path / "combined"
    make_source(corrosion, ["Slippage", "corrosion", "crack"], "1 0.5 0.5 0.2 0.3\n")

    stats = merge_source(corrosion, output, "corrosion", "corrosion", 1)

    assert stats["train"] == {"images": 1, "boxes": 1, "negative_images": 0}
    assert (output / "labels" / "train" / "corrosion_sample.txt").read_text() == (
        "1 0.5 0.5 0.2 0.3\n"
    )


def test_incomplete_source_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "partial"
    make_source(source, ["corrosion"], "0 0.5 0.5 0.2 0.3\n")

    try:
        require_complete_source(source, "corrosion")
    except ValueError as error:
        assert "Incomplete corrosion source" in str(error)
    else:
        raise AssertionError("partial data was accepted")
