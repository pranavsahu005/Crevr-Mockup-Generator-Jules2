import os
import re
import json
import base64
import cv2
import numpy as np
import sqlite3
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from typing import Optional, List

from engine.api.schemas import RenderRequest, RenderResponse, TransformParams
from engine.pipeline.render import render_mockup_pipeline, calculate_export_dimensions

app = FastAPI(
    title="Crevr Mockup Generator Engine",
    description="Deterministic high-performance product mockup compositing engine.",
    version="1.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_PATH = "data/crevr.db"

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS render_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            template_id TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            export_format TEXT,
            width INTEGER,
            height INTEGER
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Mount templates directory so they are accessible statically (e.g. for base.png or mask.png)
if os.path.exists("templates"):
    app.mount("/templates", StaticFiles(directory="templates"), name="templates")

def validate_template_id(template_id: str) -> str:
    """
    Validates template_id format and checks for directory traversal.
    Returns the sanitized absolute directory path.
    """
    if not re.match(r"^[a-zA-Z0-9_-]+$", template_id):
        raise HTTPException(status_code=400, detail="Invalid template ID format")

    base_dir = os.path.abspath("templates")
    target_dir = os.path.abspath(os.path.join(base_dir, template_id))

    # Ensure target_dir is strictly within the base templates directory
    if not target_dir.startswith(base_dir):
        raise HTTPException(status_code=400, detail="Directory traversal attempt detected")

    return target_dir

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "engine": "Crevr Mockup Engine v1.0"}

@app.get("/api/templates")
def list_templates():
    """
    Lists all ingested and ready-to-use mockup templates on disk.
    """
    templates_dir = "templates"
    if not os.path.exists(templates_dir):
        return []

    templates = []
    for t_id in os.listdir(templates_dir):
        try:
            # Safely validate template dir path
            target_dir = validate_template_id(t_id)
            meta_path = os.path.join(target_dir, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    templates.append(meta)
        except Exception as e:
            # Skip invalid or unauthorized directories
            continue
    return templates

@app.get("/api/templates/{template_id}")
def get_template_by_id(template_id: str):
    target_dir = validate_template_id(template_id)
    meta_path = os.path.join(target_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

def decode_base64_image(b64_str: str) -> np.ndarray:
    """Decodes base64 string (including Data URL scheme) to OpenCV BGR/BGRA array."""
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    img_bytes = base64.b64decode(b64_str)
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Invalid image or corrupted bytes")
    return img

@app.post("/api/render", response_model=RenderResponse)
def render_mockup(req: RenderRequest):
    # Secure path validation
    template_dir = validate_template_id(req.template_id)
    if not os.path.exists(template_dir):
        raise HTTPException(status_code=404, detail=f"Template {req.template_id} not found")

    # Read template's metadata
    with open(os.path.join(template_dir, "metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Load support files
    base_img = cv2.imread(os.path.join(template_dir, meta["base_image"]))
    mask_img = cv2.imread(os.path.join(template_dir, meta["mask_image"]), cv2.IMREAD_GRAYSCALE)

    disp_path = os.path.join(template_dir, meta["displacement_image"])
    disp_img = cv2.imread(disp_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(disp_path) else None

    light_path = os.path.join(template_dir, meta["lighting_image"])
    light_img = cv2.imread(light_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(light_path) else None

    # Extract coordinates
    # Use client supplied dynamic high-res dst_corners if provided, otherwise default to metadata
    dst_corners = req.dst_corners if req.dst_corners is not None else meta.get("design_zone_corners")

    # Decode user's custom design image
    try:
        design_img = decode_base64_image(req.design_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode design image: {str(e)}")

    h_ds, w_ds = design_img.shape[:2]
    src_corners = [[0, 0], [w_ds, 0], [w_ds, h_ds], [0, h_ds]]

    # Set default fold intensity if not custom supplied
    fold_intensity = req.fold_intensity if req.fold_intensity is not None else meta.get("fold_intensity", 15.0)

    # Transform options mapping
    t_params = None
    if req.transform_params and req.dst_corners is None:
        t_params = {
            "scale": req.transform_params.scale,
            "rotation": req.transform_params.rotation,
            "offset_x": req.transform_params.offset_x,
            "offset_y": req.transform_params.offset_y
        }

    # Process through pipeline
    try:
        composite = render_mockup_pipeline(
            base_image=base_img,
            design_image=design_img,
            mask_image=mask_img,
            displacement_image=disp_img,
            lighting_image=light_img,
            src_corners=src_corners,
            dst_corners=dst_corners,
            fold_intensity=fold_intensity,
            feather_radius=req.feather_radius,
            blend_mode=req.blend_mode,
            color_correct=req.color_correct,
            transform_params=t_params,

            # Phase 2 forwardings
            garment_color=req.garment_color,
            corner_radius=req.corner_radius,
            perspective_bend=req.perspective_bend,
            remove_background=req.remove_background
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Render pipeline failed: {str(e)}")

    # Optional print DPI export resizing
    if req.physical_size_mm and len(req.physical_size_mm) == 2:
        export_w, export_h = calculate_export_dimensions(
            (req.physical_size_mm[0], req.physical_size_mm[1]),
            req.dpi
        )
        # Check max resolution bounds
        max_res = meta.get("export_max_resolution_px", [4000, 4000])
        export_w = min(export_w, max_res[0])
        export_h = min(export_h, max_res[1])
        composite = cv2.resize(composite, (export_w, export_h), interpolation=cv2.INTER_CUBIC)

    h_out, w_out = composite.shape[:2]

    # Encode output to base64
    ext = f".{req.export_format.lower()}"
    if ext not in [".png", ".jpg", ".jpeg", ".webp"]:
        ext = ".png"

    encode_params = []
    if ext in [".jpg", ".jpeg"]:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    elif ext == ".png":
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 4]

    success, buffer = cv2.imencode(ext, composite, encode_params)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode resulting composite image")

    b64_out = base64.b64encode(buffer).decode("utf-8")
    mime_type = "image/png" if "png" in ext else "image/jpeg" if "jpg" in ext or "jpeg" in ext else "image/webp"
    data_url = f"data:{mime_type};base64,{b64_out}"

    # Save to rendering job history SQLite
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO render_history (template_id, export_format, width, height) VALUES (?, ?, ?, ?)",
            (req.template_id, req.export_format, w_out, h_out)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Failed to record render history: {e}")

    return RenderResponse(
        mockup_base64=data_url,
        format=req.export_format,
        width=w_out,
        height=h_out
    )

@app.get("/api/history")
def get_render_history():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, template_id, timestamp, export_format, width, height FROM render_history ORDER BY timestamp DESC LIMIT 50")
    rows = cursor.fetchall()
    history = [dict(row) for row in rows]
    conn.close()
    return history

# Fallback index endpoint to serve static frontend/index.html directly
@app.get("/")
def get_frontend():
    frontend_index = "frontend/index.html"
    if os.path.exists(frontend_index):
        with open(frontend_index, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    else:
        return HTMLResponse("<h1>Crevr Frontend Not Ready Yet</h1>")
