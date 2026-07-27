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

def test_template_category_filtering():
    # Test GET template category filtering
    get_res = client.get("/api/templates?category=tech")
    assert get_res.status_code == 200
    templates_get = get_res.json()
    for t in templates_get:
        assert t["category"] == "tech"

    # Test POST template category filtering
    post_res = client.post("/api/templates", json={"category": "apparel"})
    assert post_res.status_code == 200
    templates_post = post_res.json()
    for t in templates_post:
        assert t["category"] == "apparel"

def test_upload_prompt_bg_removal():
    # Fully opaque image should prompt for background removal
    opaque_bytes = create_dummy_image(100, 100, color=(255, 0, 0, 255))
    res = client.post(
        "/api/designs/upload",
        files={"file": ("opaque.png", opaque_bytes, "image/png")}
    )
    assert res.status_code == 200
    assert res.json()["prompt_bg_removal"] is True

    # Image with transparent background should not prompt
    transparent_bytes = create_dummy_image(100, 100, color=(255, 0, 0, 100))
    res2 = client.post(
        "/api/designs/upload",
        files={"file": ("transparent.png", transparent_bytes, "image/png")}
    )
    assert res2.status_code == 200
    assert res2.json()["prompt_bg_removal"] is False

def test_render_warnings():
    # Upload an opaque and small design to trigger warnings when rendering on apparel
    opaque_small_bytes = create_dummy_image(10, 10, color=(255, 255, 255, 255))
    upload_res = client.post(
        "/api/designs/upload",
        files={"file": ("small.png", opaque_small_bytes, "image/png")}
    )
    design_id = upload_res.json()["design_id"]

    # Render with custom physical size that forces clamping
    render_payload = {
        "template_id": "tshirt_01",
        "design_id": design_id,
        "physical_size_mm": [9999.0, 9999.0], # massive physical size
        "dpi": 600
    }
    render_res = client.post("/api/render", json=render_payload)
    assert render_res.status_code == 200
    data = render_res.json()
    assert "warnings" in data
    warnings = data["warnings"]
    assert warnings is not None
    # Should have upscaling and missing transparency and clamping warnings
    assert any("transparency" in w for w in warnings)
    assert any("upscaling" in w for w in warnings)
    assert any("clamped" in w for w in warnings)

def test_render_base64_decompression_bomb_limits():
    # Test base64 payload length security validation
    huge_payload = "a" * (36 * 1024 * 1024)
    render_payload = {
        "template_id": "tshirt_01",
        "design_base64": huge_payload
    }
    render_res = client.post("/api/render", json=render_payload)
    assert render_res.status_code == 400
    assert "exceeds" in render_res.json()["detail"]
