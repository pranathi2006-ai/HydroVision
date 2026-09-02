from __future__ import annotations

from pathlib import Path

import yaml

from training.download_datasets import normalize_ship_label
from training.prepare_yolo_dataset import (
    merge_source,
    normalized_bbox,
    read_names,
    replace_output,
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
    assert source_class_ids(
        ["rust", "corrosion", "moderate corrosion", "severe corrosion"],
        "corrosion",
    ) == {0, 1, 2, 3}


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


def test_ship_label_normalization_repairs_reversed_box_edges(tmp_path: Path) -> None:
    normalized, repaired = normalize_ship_label(
        "1 0.544509 0.205354 -0.0350877 0.19802",
        tmp_path / "sample.txt",
    )
    assert repaired is True
    assert normalized == "1 0.544509 0.205354 0.0350877 0.19802"


def test_segmentation_polygon_is_converted_to_tight_bbox(tmp_path: Path) -> None:
    class_id, bbox = normalized_bbox(
        "2 0.2 0.1 0.8 0.1 0.8 0.7 0.2 0.7".split(),
        tmp_path / "polygon.txt",
    )
    assert class_id == 2
    assert [float(value) for value in bbox] == [0.5, 0.4, 0.6, 0.6]


def test_merge_accepts_polygon_corrosion_annotations(tmp_path: Path) -> None:
    corrosion = tmp_path / "corrosion"
    output = tmp_path / "combined"
    make_source(
        corrosion,
        ["rust"],
        "0 0.2 0.1 0.8 0.1 0.8 0.7 0.2 0.7\n",
    )

    stats = merge_source(corrosion, output, "rgb", "corrosion", 1)

    assert stats["train"]["boxes"] == 1
    assert (output / "labels" / "train" / "rgb_sample.txt").read_text() == (
        "1 0.5 0.4 0.6 0.6\n"
    )


def test_completed_staging_dataset_replaces_previous_output(tmp_path: Path) -> None:
    output = tmp_path / "hydrovision"
    staging = tmp_path / ".hydrovision-building"
    output.mkdir()
    staging.mkdir()
    (output / "old.txt").write_text("old")
    (staging / "data.yaml").write_text("names: [oil_leak, corrosion]\n")

    replace_output(staging, output)

    assert not (output / "old.txt").exists()
    assert (output / "data.yaml").is_file()
    assert not staging.exists()
