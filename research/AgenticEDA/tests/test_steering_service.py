from backend.services.steering_service import (
    clear_steering,
    drain_steering,
    enqueue_steering,
    get_steering_items,
)


def test_enqueue_and_drain_steering_fifo():
    clear_steering("s1")

    first = enqueue_steering("s1", "focus on anomalies")
    second = enqueue_steering("s1", "compare weekends")

    assert first["status"] == "queued"
    assert second["status"] == "queued"

    drained = drain_steering("s1")

    assert [item["content"] for item in drained] == [
        "focus on anomalies",
        "compare weekends",
    ]
    assert all(item["status"] == "read" for item in drained)
    assert all(item["read_at"] for item in drained)
    assert drain_steering("s1") == []


def test_get_steering_items_returns_read_and_queued_items():
    clear_steering("s2")

    first = enqueue_steering("s2", "first")
    second = enqueue_steering("s2", "second")
    drain_steering("s2", limit=1)

    items = get_steering_items("s2")

    assert [item["id"] for item in items] == [first["id"], second["id"]]
    assert [item["content"] for item in items] == ["first", "second"]
    assert [item["status"] for item in items] == ["read", "queued"]


def test_steering_endpoint_rejects_when_agent_not_running(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import backend.routers.run as run_router
    from backend.app import app

    clear_steering("s3")
    session_dir = tmp_path / "s3"
    session_dir.mkdir()
    monkeypatch.setattr(run_router, "get_session_dir", lambda session_id: session_dir)
    run_router._running.pop("s3", None)

    response = TestClient(app).post(
        "/api/run/s3/steering",
        json={"content": "focus on anomalies", "message_id": "msg-1"},
    )

    assert response.status_code == 409
    assert get_steering_items("s3") == []
