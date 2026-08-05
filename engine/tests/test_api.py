import io
import os
import base64
import shutil
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from engine.main import app

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Make sure we clean up test files under data/uploads after testing
    os.makedirs("data/uploads", exist_ok=True)
    yield
    # Cleanup data/uploads of any temporary pngs
    for filename in os.listdir("data/uploads"):
        filepath = os.path.join("data/uploads", filename)
        if os.path.isfile(filepath) and filename.endswith(".png"):
            os.remove(filepath)

def create_dummy_image(w=100, h=100, format="PNG", color=(255, 0, 0, 255)):
    img = Image.new("RGBA", (w, h), color=color)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_upload_design():
    # Create a small valid PNG
    img_bytes = create_dummy_image(200, 200)
    response = client.post(
        "/api/designs/upload",
        files={"file": ("test_design.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "design_id" in data
    assert "preview" in data
    assert data["preview"].startswith("data:image/png;base64,")

def test_upload_design_oversized():
    # Create an image larger than 8000x8000
    img_bytes = create_dummy_image(8001, 100)
    response = client.post(
        "/api/designs/upload",
        files={"file": ("test_huge.png", img_bytes, "image/png")}
    )
    assert response.status_code == 400
    assert "exceeds" in response.json()["detail"]

def test_upload_invalid_format():
    response = client.post(
        "/api/designs/upload",
        files={"file": ("test.txt", b"plain text", "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]

def test_remove_background():
    # Upload a design with white background
    img_bytes = create_dummy_image(100, 100, color=(255, 255, 255, 255))
    upload_res = client.post(
        "/api/designs/upload",
        files={"file": ("design_to_remove.png", img_bytes, "image/png")}
    )
    design_id = upload_res.json()["design_id"]

    # Remove bg
    bg_res = client.post(f"/api/designs/{design_id}/remove-bg")
    assert bg_res.status_code == 200
    data = bg_res.json()
    assert data["design_id"] == design_id
    assert "preview" in data

    # Check path traversal block
    traversal_res = client.post("/api/designs/../remove-bg")
    assert traversal_res.status_code == 404 # route not matching or invalid

def test_upload_suggest_bg_removal():
    # 1. Upload an opaque design (returns prompt_bg_removal = True)
    img_bytes_opaque = create_dummy_image(100, 100, color=(255, 0, 0, 255))
    res_opaque = client.post(
        "/api/designs/upload",
        files={"file": ("opaque.png", img_bytes_opaque, "image/png")}
    )
    assert res_opaque.status_code == 200
    assert res_opaque.json()["prompt_bg_removal"] is True

    # 2. Upload a transparent design (returns prompt_bg_removal = False)
    img_bytes_transparent = create_dummy_image(100, 100, color=(255, 0, 0, 0)) # fully transparent
    res_trans = client.post(
        "/api/designs/upload",
        files={"file": ("transparent.png", img_bytes_transparent, "image/png")}
    )
    assert res_trans.status_code == 200
    assert res_trans.json()["prompt_bg_removal"] is False

def test_render_warnings():
    # Upload a small opaque design to trigger multiple warnings on apparel template
    img_bytes_opaque = create_dummy_image(50, 50, color=(0, 255, 0, 255)) # small and opaque
    upload_res = client.post(
        "/api/designs/upload",
        files={"file": ("small_opaque.png", img_bytes_opaque, "image/png")}
    )
    design_id = upload_res.json()["design_id"]

    # Trigger render on 'tshirt_01' which expects larger transparent design
    render_payload = {
        "template_id": "tshirt_01",
        "design_id": design_id,
        "blend_mode": "multiply",
        "color_correct": False,
        "feather_radius": 3,
        "linear_blend": True,
        "physical_size_mm": [1000.0, 1000.0], # massive physical size to trigger resolution clamping
        "dpi": 600
    }
    render_res = client.post("/api/render", json=render_payload)
    assert render_res.status_code == 200
    data = render_res.json()
    assert "warnings" in data
    warnings = data["warnings"]
    assert len(warnings) > 0
    # Should warning about upscaling/resolution
    assert any("resolution" in w or "upscaling" in w for w in warnings)
    # Should warn about opaque design on apparel
    assert any("opaque" in w or "transparency" in w for w in warnings)
    # Should warn about clamping
    assert any("clamp" in w or "exceed" in w for w in warnings)

def test_render_using_uploaded_id():
    # Upload design first
    img_bytes = create_dummy_image(100, 100, color=(0, 255, 0, 255))
    upload_res = client.post(
        "/api/designs/upload",
        files={"file": ("design.png", img_bytes, "image/png")}
    )
    design_id = upload_res.json()["design_id"]

    # Trigger render using template 'tshirt_01'
    render_payload = {
        "template_id": "tshirt_01",
        "design_id": design_id,
        "blend_mode": "multiply",
        "color_correct": False,
        "feather_radius": 3
    }
    render_res = client.post("/api/render", json=render_payload)
    assert render_res.status_code == 200
    data = render_res.json()
    assert "mockup_base64" in data
    assert data["format"] == "png"

def test_delete_history():
    # Fetch history first to check count or entries
    res_history = client.get("/api/history")
    assert res_history.status_code == 200
    history = res_history.json()
    if len(history) > 0:
        item_id = history[0]["id"]
        del_res = client.delete(f"/api/history/{item_id}")
        assert del_res.status_code == 200
        assert del_res.json()["status"] == "success"

def test_template_ingest_apparel():
    # Create mock apparel base photo
    img_bytes = create_dummy_image(300, 300, color=(255, 255, 255, 255))
    response = client.post(
        "/api/templates/ingest",
        data={
            "template_id": "test_tshirt_ingested",
            "category": "apparel",
            "subtype": "t-shirt",
            "label": "Test Custom T-Shirt"
        },
        files={"file": ("tshirt_base.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    meta = response.json()
    assert meta["id"] == "test_tshirt_ingested"
    assert meta["category"] == "apparel"
    assert meta["subtype"] == "t-shirt"
    assert os.path.exists("templates/test_tshirt_ingested/metadata.json")
    assert os.path.exists("templates/test_tshirt_ingested/base.png")
    assert os.path.exists("templates/test_tshirt_ingested/mask.png")
    assert os.path.exists("templates/test_tshirt_ingested/displacement.png")
    assert os.path.exists("templates/test_tshirt_ingested/lighting.png")

    # Cleanup ingested template directory
    shutil.rmtree("templates/test_tshirt_ingested", ignore_errors=True)
