import os
import io
import cv2
import base64
import sqlite3
import numpy as np
import pytest
from fastapi.testclient import TestClient
from engine.main import app, DATABASE_PATH, UPLOADS_DIR

client = TestClient(app)

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_list_templates():
    response = client.get("/api/templates")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_design_upload_and_bg_removal():
    # 1. Create a dummy solid image as PNG bytes using Pillow
    from PIL import Image
    img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 255))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # 2. Upload file
    response = client.post(
        "/api/designs/upload",
        files={"file": ("test_design.png", img_byte_arr, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "design_id" in data
    design_id = data["design_id"]

    # 3. Retrieve design image
    response = client.get(f"/api/designs/{design_id}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

    # 4. Remove background
    response = client.post(f"/api/designs/{design_id}/remove-bg")
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    # 5. Check if the corners of the processed image are transparent (alpha = 0)
    saved_img_path = os.path.join(UPLOADS_DIR, f"{design_id}.png")
    img_processed = cv2.imread(saved_img_path, cv2.IMREAD_UNCHANGED)
    assert img_processed is not None
    assert img_processed.shape[2] == 4  # Should have alpha channel
    # Since it is a solid color, the whole image gets flood filled from corners, so all alpha should be 0
    assert img_processed[0, 0, 3] == 0
    assert img_processed[99, 99, 3] == 0

def test_invalid_design_upload_oversized():
    # Attempt upload of file too large (26MB)
    huge_data = b"0" * (26 * 1024 * 1024)
    response = client.post(
        "/api/designs/upload",
        files={"file": ("test_huge.png", huge_data, "image/png")}
    )
    assert response.status_code == 400
    assert "exceeds" in response.json()["detail"]

def test_invalid_design_upload_resolution():
    from PIL import Image
    # Create image 8001x10
    img = Image.new("RGBA", (8001, 10), color=(255, 0, 0, 255))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    response = client.post(
        "/api/designs/upload",
        files={"file": ("test_res.png", img_byte_arr, "image/png")}
    )
    assert response.status_code == 400
    assert "exceeds" in response.json()["detail"]

def test_delete_history_and_render_by_id():
    # First make sure database record is present
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO render_history (template_id, export_format, width, height) VALUES (?, ?, ?, ?)",
        ("laptop_01", "png", 100, 100)
    )
    history_id = cursor.lastrowid
    conn.commit()
    conn.close()

    # Get history list and confirm
    response = client.get("/api/history")
    assert response.status_code == 200
    initial_len = len(response.json())
    assert initial_len > 0

    # Delete history item
    response = client.delete(f"/api/history/{history_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    # Confirm it was deleted
    response = client.get("/api/history")
    assert response.status_code == 200
    assert len(response.json()) == initial_len - 1

def test_templates_ingest():
    # Trigger ingest endpoint
    response = client.post("/api/templates/ingest")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
