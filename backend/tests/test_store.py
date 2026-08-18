from __future__ import annotations

from pathlib import Path

from backend.store import Store


def test_clear_all_hides_results_but_retains_hash_cache(tmp_path: Path) -> None:
    store = Store(tmp_path / "hydrovision.sqlite3")
    media_id = store.insert_media(
        content_hash="abc123",
        original_name="leak.jpg",
        stored_name="abc123.jpg",
        media_type="image",
        location_id="turbine-a",
        width=640,
        height=480,
        inference_engine="test",
    )
    store.insert_findings(media_id, [{
        "defect_type": "leak",
        "confidence": 0.9,
        "severity": "critical",
        "affected_area_pct": 10.0,
        "bbox": [10, 20, 100, 120],
    }])

    cleared = store.clear_all()

    assert cleared == {
        "findings_cleared": 1,
        "media_cleared": 1,
        "cache_retained": True,
    }
    assert store.findings() == []
    cached = store.cached_media("abc123")
    assert cached is not None
    assert cached["cleared_at"] is not None
    assert store.snapshot()["metrics"]["active_findings"] == 0

    store.restore_media(media_id)

    assert len(store.findings()) == 1
    assert store.cached_media("abc123")["cleared_at"] is None
