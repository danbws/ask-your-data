import base64
import io

import pytest
from fastapi.testclient import TestClient

from ask_your_data.api import create_app, default_provider_factory
from ask_your_data.providers import demo_provider


@pytest.fixture()
def client(tmp_path):
    return TestClient(create_app(tmp_path, provider_factory=lambda ds: demo_provider()))


def test_sample_dataset_then_ask(client):
    ds = client.post("/datasets/sample").json()
    assert set(ds["tables"]) == {"machines", "production_orders", "dye_batches"}

    r = client.post("/ask", json={"dataset_id": ds["dataset_id"], "question": "re-dye rate?"})
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "fake" and body["stop_reason"] == "end_turn"
    assert [s["tool"] for s in body["steps"]] == ["run_sql", "run_sql", "run_python", "make_chart"]
    assert body["steps"][0]["is_error"] is True
    assert len(body["sql"]) == 4
    assert base64.b64decode(body["chart_base64"])[:4] == b"\x89PNG"
    assert "JET-07" in body["answer"]


def test_upload_csv_creates_a_queryable_table(client):
    csv_bytes = b"Order ID,Weight (kg)\n1,10.5\n2,20\n"
    r = client.post("/datasets", files=[("files", ("My Orders.csv", io.BytesIO(csv_bytes)))])
    assert r.status_code == 201
    ds = r.json()
    assert ds["tables"] == ["my_orders"]
    assert client.get(f"/datasets/{ds['dataset_id']}").json()["tables"] == ["my_orders"]


def test_upload_rejects_non_csv(client):
    r = client.post("/datasets", files=[("files", ("x.exe", io.BytesIO(b"MZ")))])
    assert r.status_code == 400


def test_upload_rejects_garbage_csv(client):
    r = client.post("/datasets", files=[("files", ("x.csv", io.BytesIO(b"")))])
    assert r.status_code == 400


def test_ask_unknown_dataset_404(client):
    r = client.post("/ask", json={"dataset_id": "nope", "question": "anything?"})
    assert r.status_code == 404


def test_ask_validates_input(client):
    ds = client.post("/datasets/sample").json()
    assert client.post("/ask", json={"dataset_id": ds["dataset_id"], "question": "x"}).status_code == 422


def test_connect_is_disabled_by_default(client, monkeypatch):
    monkeypatch.delenv("ASK_YOUR_DATA_ALLOW_DB_URLS", raising=False)
    r = client.post("/datasets/connect", json={"url": "sqlite:///whatever.db"})
    assert r.status_code == 403


def test_connect_when_enabled(client, monkeypatch, sample_db_path):
    monkeypatch.setenv("ASK_YOUR_DATA_ALLOW_DB_URLS", "1")
    r = client.post("/datasets/connect", json={"url": f"sqlite:///{sample_db_path}"})
    assert r.status_code == 201 and "machines" in r.json()["tables"]
    missing = client.post("/datasets/connect", json={"url": "sqlite:///does_not_exist.db"})
    assert missing.status_code == 400


def test_default_provider_without_key_only_serves_the_sample(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = TestClient(create_app(tmp_path, provider_factory=default_provider_factory))
    sample = client.post("/datasets/sample").json()
    assert client.post("/ask", json={"dataset_id": sample["dataset_id"],
                                     "question": "re-dye rate?"}).status_code == 200
    upload = client.post("/datasets", files=[("files", ("a.csv", io.BytesIO(b"x,y\n1,2\n")))]).json()
    r = client.post("/ask", json={"dataset_id": upload["dataset_id"], "question": "sum of x?"})
    assert r.status_code == 503 and "ANTHROPIC_API_KEY" in r.json()["detail"]


def test_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "ask-your-data" in r.text
