import os
import io
import json
import base64
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from engine.main import app

client = TestClient(app)

def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_list_templates():
    response = client.get("/api/templates")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_list_templates_filtering():
    response = client.get("/api/templates?category=tech")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    for tpl in response.json():
        assert tpl["category"] == "tech"

def test_history_api():
    # Fetch history
    response = client.get("/api/history")
    assert response.status_code == 200
    history = response.json()
    assert isinstance(history, list)

    # Try deleting a non-existent or dummy ID (it shouldn't crash)
    response = client.delete("/api/history/999999")
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_design_upload_validation():
    # 1. Reject invalid signature / mime-type
    response = client.post(
        "/api/designs/upload",
        files={"file": ("test.txt", b"some plain text data that is not an image", "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]

    # 2. Upload valid PNG
    img = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/api/designs/upload",
        files={"file": ("test.png", buf, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "design_id" in data
    assert data["has_alpha"] is True

    # 3. Preview uploaded image
    design_id = data["design_id"]
    response = client.get(f"/api/designs/{design_id}/preview")
    assert response.status_code == 200

    # 4. Remove background of uploaded design (returns transparent png)
    # Let's create an image without alpha (opaque) first
    img_opaque = Image.new("RGB", (100, 100), (255, 255, 255))
    buf_opaque = io.BytesIO()
    img_opaque.save(buf_opaque, format="JPEG")
    buf_opaque.seek(0)

    response_opaque = client.post(
        "/api/designs/upload",
        files={"file": ("test.jpg", buf_opaque, "image/jpeg")}
    )
    assert response_opaque.status_code == 200
    opaque_data = response_opaque.json()
    assert opaque_data["has_alpha"] is False

    # Remove background on the opaque design
    opaque_id = opaque_data["design_id"]
    bg_response = client.post(f"/api/designs/{opaque_id}/remove-bg")
    assert bg_response.status_code == 200
    assert bg_response.json()["design_id"] == opaque_id
