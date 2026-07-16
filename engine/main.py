import os
import re
import json
import base64
import cv2
import numpy as np
import sqlite3
import uuid
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from typing import Optional, List
from PIL import Image
import io

from engine.api.schemas import RenderRequest, RenderResponse, TransformParams
from engine.pipeline.render import render_mockup_pipeline, calculate_export_dimensions
from engine.pipeline.ingest import generate_displacement_map, generate_lighting_overlay, convert_to_serializable

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
UPLOADS_DIR = "data/uploads"

os.makedirs(UPLOADS_DIR, exist_ok=True)

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
def list_templates(category: Optional[str] = Query(None)):
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
                    if category is None or category.lower() == "all" or meta.get("category", "").lower() == category.lower():
                        templates.append(meta)
        except Exception as e:
            # Skip invalid or unauthorized directories
            continue
    return templates

@app.post("/api/templates")
def create_template(
    id: str = Form(...),
    category: str = Form(...),
    subtype: str = Form(...),
    label: str = Form(...),
    fold_intensity: float = Form(15.0),
    base_file: UploadFile = File(...)
):
    """
    (Admin) Ingest/create a new template with a raw base photo.
    """
    if not re.match(r"^[a-zA-Z0-9_-]+$", id):
        raise HTTPException(status_code=400, detail="Invalid template ID format")

    target_dir = os.path.join("templates", id)
    if os.path.exists(target_dir):
        raise HTTPException(status_code=400, detail="Template ID already exists")

    os.makedirs(target_dir, exist_ok=True)

    try:
        content = base_file.file.read()

        # Verify and strip EXIF
        try:
            img_pil = Image.open(io.BytesIO(content))
            # Strip EXIF
            data = list(img_pil.getdata())
            img_clean = Image.new(img_pil.mode, img_pil.size)
            img_clean.putdata(data)

            # Save base image
            base_path = os.path.join(target_dir, "base.png")
            img_clean.save(base_path, "PNG")
        except Exception as e:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=f"Invalid base image: {str(e)}")

        base_img = cv2.imread(base_path)
        h, w = base_img.shape[:2]

        # Sane defaults: auto mask the central region of the image
        # Let's create a central rectangular design zone mask (similar to ingest.py fallback)
        mask_img = np.zeros((h, w), dtype=np.uint8)
        z_w, z_h = int(w * 0.4), int(h * 0.5)
        cx, cy = w // 2, h // 2
        z_x1, z_y1 = cx - z_w // 2, cy - z_h // 2
        z_x2, z_y2 = cx + z_w // 2, cy + z_h // 2
        cv2.rectangle(mask_img, (z_x1, z_y1), (z_x2, z_y2), 255, -1)

        mask_path = os.path.join(target_dir, "mask.png")
        cv2.imwrite(mask_path, mask_img)

        # Generate displacement and lighting
        disp_img = generate_displacement_map(base_img, mask_img, fold_intensity)
        light_img = generate_lighting_overlay(base_img, mask_img)

        cv2.imwrite(os.path.join(target_dir, "displacement.png"), disp_img)
        cv2.imwrite(os.path.join(target_dir, "lighting.png"), light_img)

        corners = [[z_x1, z_y1], [z_x2, z_y1], [z_x2, z_y2], [z_x1, z_y2]]

        metadata = {
            "id": id,
            "category": category,
            "subtype": subtype,
            "label": label,
            "base_image": "base.png",
            "mask_image": "mask.png",
            "displacement_image": "displacement.png",
            "lighting_image": "lighting.png",
            "design_zone_corners": corners,
            "fold_intensity": fold_intensity,
            "allow_rotation": True,
            "rotation_limits_deg": [-15, 15],
            "allow_perspective_adjust": True,
            "recommended_design_resolution_px": [1500, 1500],
            "min_upload_resolution_px": [300, 300],
            "max_upload_resolution_px": [6000, 6000],
            "supported_formats": ["png", "jpg", "webp"],
            "export_default_format": "png",
            "export_max_resolution_px": [4096, 4096],
            "created_at": "2026-07-15",
            "engine_version": "1.0"
        }

        metadata = convert_to_serializable(metadata)
        with open(os.path.join(target_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        return metadata
    except Exception as e:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to ingest template: {str(e)}")

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

@app.post("/api/designs/upload")
def upload_design(file: UploadFile = File(...)):
    """
    Secure design file upload with EXIF metadata stripping and mime-type verification.
    """
    # 1. Size constraint
    content = file.file.read()
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 25MB)")

    # 2. Signature verification
    # Supported magic bytes: PNG, JPG, WebP
    is_png = content.startswith(b"\x89PNG\r\n\x1a\n")
    is_jpg = content.startswith(b"\xff\xd8\xff")
    is_webp = content.startswith(b"RIFF") and b"WEBP" in content[8:16]

    if not (is_png or is_jpg or is_webp):
        raise HTTPException(status_code=400, detail="Unsupported file format (only PNG, JPG, WebP allowed)")

    # 3. Stripping EXIF metadata & Pixel check
    try:
        img_pil = Image.open(io.BytesIO(content))
        w, h = img_pil.size
        if w > 8000 or h > 8000:
            raise HTTPException(status_code=400, detail="Dimensions exceed maximum limits (max 8000x8000)")

        # Strip EXIF
        data = list(img_pil.getdata())
        img_clean = Image.new(img_pil.mode, img_pil.size)
        img_clean.putdata(data)

        design_id = str(uuid.uuid4())
        # Standardize file extension
        ext = "png" if is_png or img_pil.mode == "RGBA" else "jpg"
        filename = f"{design_id}.{ext}"
        filepath = os.path.join(UPLOADS_DIR, filename)

        img_clean.save(filepath)

        # Quick check for alpha channel
        has_alpha = img_pil.mode == "RGBA" or "transparency" in img_pil.info

        return {
            "design_id": design_id,
            "filename": filename,
            "width": w,
            "height": h,
            "has_alpha": has_alpha,
            "preview_url": f"/api/designs/{design_id}/preview"
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

@app.get("/api/designs/{design_id}/preview")
def get_design_preview(design_id: str):
    """
    Safely serves the uploaded design preview.
    """
    if not re.match(r"^[a-zA-Z0-9_-]+$", design_id):
        raise HTTPException(status_code=400, detail="Invalid design ID")

    for ext in ["png", "jpg", "jpeg", "webp"]:
        filepath = os.path.join(UPLOADS_DIR, f"{design_id}.{ext}")
        if os.path.exists(filepath):
            return FileResponse(filepath)

    raise HTTPException(status_code=404, detail="Design not found")

@app.post("/api/designs/{design_id}/remove-bg")
def remove_background(design_id: str):
    """
    Applies classical computer vision flood fill to segment and remove the design background.
    Assumes the border/corners represent the background color.
    """
    if not re.match(r"^[a-zA-Z0-9_-]+$", design_id):
        raise HTTPException(status_code=400, detail="Invalid design ID")

    filepath = None
    file_ext = None
    for ext in ["png", "jpg", "jpeg", "webp"]:
        test_path = os.path.join(UPLOADS_DIR, f"{design_id}.{ext}")
        if os.path.exists(test_path):
            filepath = test_path
            file_ext = ext
            break

    if not filepath:
        raise HTTPException(status_code=404, detail="Design not found")

    try:
        # Load the image
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not read the design image")

        # Convert to BGRA if it doesn't already have alpha
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        h, w = img.shape[:2]

        # Extract 3-channel BGR from our image
        bgr = img[:, :, :3].copy()

        # Prepare a mask 2 pixels larger as required by cv2.floodFill
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

        # Flood fill parameters
        # loDiff and upDiff control the color tolerance (sane default: 15)
        lo_diff = (15, 15, 15)
        up_diff = (15, 15, 15)
        new_val = (0, 0, 0)

        # Fill from 4 corners
        corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
        for pt in corners:
            cv2.floodFill(bgr, mask, pt, new_val, loDiff=lo_diff, upDiff=up_diff, flags=cv2.FLOODFILL_FIXED_RANGE)

        # filled_mask represents anywhere the flood fill was applied (value 1)
        filled_mask = mask[1:h+1, 1:w+1]

        # Set alpha to 0 (completely transparent) for all filled background regions
        img[filled_mask == 1, 3] = 0

        # Save as PNG to preserve alpha channel
        new_filepath = os.path.join(UPLOADS_DIR, f"{design_id}.png")
        cv2.imwrite(new_filepath, img)

        # If original file wasn't .png, remove it to prevent clutter
        if file_ext != "png":
            os.remove(filepath)

        return {
            "design_id": design_id,
            "filename": f"{design_id}.png",
            "message": "Background removed successfully",
            "preview_url": f"/api/designs/{design_id}/preview"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove background: {str(e)}")

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

    # Handle design image (from base64 or upload design_id)
    design_img = None
    if req.design_id:
        if not re.match(r"^[a-zA-Z0-9_-]+$", req.design_id):
            raise HTTPException(status_code=400, detail="Invalid design ID format")
        # Try to locate the file
        found_path = None
        for ext in ["png", "jpg", "jpeg", "webp"]:
            test_path = os.path.join(UPLOADS_DIR, f"{req.design_id}.{ext}")
            if os.path.exists(test_path):
                found_path = test_path
                break
        if not found_path:
            raise HTTPException(status_code=404, detail="Design file not found")
        design_img = cv2.imread(found_path, cv2.IMREAD_UNCHANGED)
    elif req.design_base64:
        try:
            design_img = decode_base64_image(req.design_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode design image: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Either design_base64 or design_id must be provided")

    if design_img is None:
        raise HTTPException(status_code=400, detail="Could not read or process design image")

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
            transform_params=t_params
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

@app.delete("/api/history/{id}")
def delete_history_item(id: int):
    """
    Deletes a history item by ID from the SQLite database.
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM render_history WHERE id = ?", (id,))
        conn.commit()
        conn.close()
        return {"status": "success", "message": f"History item {id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete history item: {str(e)}")

# Fallback index endpoint to serve static frontend/index.html directly
@app.get("/")
def get_frontend():
    frontend_index = "frontend/index.html"
    if os.path.exists(frontend_index):
        with open(frontend_index, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    else:
        return HTMLResponse("<h1>Crevr Frontend Not Ready Yet</h1>")
