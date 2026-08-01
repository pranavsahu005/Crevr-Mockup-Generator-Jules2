import os
import re
import json
import base64
import cv2
import numpy as np
import sqlite3
import uuid
import io
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from typing import Optional, List

from engine.api.schemas import RenderRequest, RenderResponse, TransformParams
from engine.pipeline.render import render_mockup_pipeline, calculate_export_dimensions
from engine.pipeline.ingest import generate_lighting_overlay, generate_displacement_map, convert_to_serializable

app = FastAPI(
    title="Crevr Mockup Generator Engine",
    description="Deterministic high-performance product mockup compositing engine.",
    version="1.0"
)

# Ensure uploads directory exists
os.makedirs("data/uploads", exist_ok=True)

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
    if req.design_id:
        target_path = validate_design_id(req.design_id)
        if not os.path.exists(target_path):
            raise HTTPException(status_code=404, detail=f"Design {req.design_id} not found")
        design_img = cv2.imread(target_path, cv2.IMREAD_UNCHANGED)
        if design_img is None:
            raise HTTPException(status_code=400, detail="Failed to load design image from disk")
    else:
        try:
            design_img = decode_base64_image(req.design_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode design image: {str(e)}")

    h_ds, w_ds = design_img.shape[:2]
    src_corners = [[0, 0], [w_ds, 0], [w_ds, h_ds], [0, h_ds]]

    # Warnings collection
    warnings = []

    # 1. Check for upscaling (design dimensions smaller than template recommended dimensions)
    rec_res = meta.get("recommended_design_resolution_px", [1000, 1000])
    if w_ds < rec_res[0] or h_ds < rec_res[1]:
        warnings.append("upscaling")

    # 2. Check for missing transparency on apparel templates (design is fully opaque)
    if meta.get("category") == "apparel":
        is_fully_opaque = True
        if design_img.shape[2] == 4:
            alpha_channel = design_img[:, :, 3]
            min_val, _, _, _ = cv2.minMaxLoc(alpha_channel)
            if min_val < 255:
                is_fully_opaque = False
        if is_fully_opaque:
            warnings.append("missing_transparency")

    # Optional print DPI export resizing
    clamped = False
    target_export_w, target_export_h = None, None
    if req.physical_size_mm and len(req.physical_size_mm) == 2:
        export_w, export_h = calculate_export_dimensions(
            (req.physical_size_mm[0], req.physical_size_mm[1]),
            req.dpi
        )
        # Check max resolution bounds
        max_res = meta.get("export_max_resolution_px", [4000, 4000])
        if export_w > max_res[0] or export_h > max_res[1]:
            clamped = True
            export_w = min(export_w, max_res[0])
            export_h = min(export_h, max_res[1])
        target_export_w, target_export_h = export_w, export_h

    # 3. Check if resolution gets clamped
    if clamped:
        warnings.append("resolution_clamped")

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
            linear_blend=req.linear_blend or False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Render pipeline failed: {str(e)}")

    if target_export_w is not None and target_export_h is not None:
        composite = cv2.resize(composite, (target_export_w, target_export_h), interpolation=cv2.INTER_CUBIC)

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
        height=h_out,
        warnings=warnings
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

def sniff_and_sanitize_image(file_bytes: bytes) -> tuple:
    """
    Sniffs magic bytes to determine mime type, checks constraints,
    strips EXIF metadata, and returns a PIL Image and MIME type.
    """
    if len(file_bytes) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds the 25MB limit")

    # Sniff magic bytes
    mime_type = None
    if file_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        mime_type = "image/png"
    elif file_bytes.startswith(b"\xff\xd8\xff"):
        mime_type = "image/jpeg"
    elif file_bytes.startswith(b"RIFF") and b"WEBP" in file_bytes[8:16]:
        mime_type = "image/webp"
    else:
        # Extra check: allow general image formats if PIL can open them, but enforce magic signature or reject
        raise HTTPException(status_code=400, detail="Unsupported or corrupt image format")

    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()  # Corrupt image check
        # Re-open after verify()
        img = Image.open(io.BytesIO(file_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Corrupt or invalid image: {str(e)}")

    w, h = img.size
    if w > 8000 or h > 8000:
        raise HTTPException(status_code=400, detail="Image resolution exceeds the 8000x8000 limit")

    return img, mime_type

@app.post("/api/designs/upload")
async def upload_design(file: UploadFile = File(...)):
    """
    Uploads a user design, sniffs MIME-type, enforces security limits,
    strips EXIF metadata, stores it under a random UUID, and returns ID + preprocessed preview.
    """
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    img, mime_type = sniff_and_sanitize_image(file_bytes)

    # Check if the uploaded design is fully opaque to set prompt_bg_removal flag
    prompt_bg_removal = True
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        alpha = img.getchannel("A") if "A" in img.getbands() else None
        if alpha:
            extrema = alpha.getextrema()
            # If the minimum alpha value is less than 255, there is transparency
            if extrema and extrema[0] < 255:
                prompt_bg_removal = False
        else:
            if img.mode == "P" and "transparency" in img.info:
                prompt_bg_removal = False

    # Convert to RGBA to preserve transparency if it exists or if we save as PNG
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")

    upload_id = str(uuid.uuid4())
    filepath = os.path.join("data/uploads", f"{upload_id}.png")

    try:
        # Saving as PNG with Pillow naturally strips EXIF metadata unless specifically supplied
        img.save(filepath, format="PNG")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save design image: {str(e)}")

    # Generate a preprocessed preview
    # To keep network payload small, we can create a fast preview
    preview_img = img.copy()
    preview_img.thumbnail((800, 800))
    buffered = io.BytesIO()
    preview_img.save(buffered, format="PNG")
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    preview_url = f"data:image/png;base64,{b64_str}"

    return {
        "design_id": upload_id,
        "preview": preview_url,
        "prompt_bg_removal": prompt_bg_removal
    }

def validate_design_id(design_id: str) -> str:
    """
    Validates design_id format and checks for directory traversal.
    Returns the sanitized absolute file path.
    """
    if not re.match(r"^[a-zA-Z0-9_-]+$", design_id):
        raise HTTPException(status_code=400, detail="Invalid design ID format")

    base_dir = os.path.abspath("data/uploads")
    target_path = os.path.abspath(os.path.join(base_dir, f"{design_id}.png"))

    if not target_path.startswith(base_dir):
        raise HTTPException(status_code=400, detail="Directory traversal attempt detected")

    return target_path

@app.post("/api/designs/{design_id}/remove-bg")
async def remove_background(design_id: str):
    """
    Runs background removal using a classical OpenCV flood-fill algorithm starting from the four corners.
    Separates 4-channel BGRA to BGR for flood filling and modifies the alpha channel of matching pixels.
    """
    target_path = validate_design_id(design_id)
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail="Design file not found")

    # Read image with alpha channel
    img = cv2.imread(target_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(status_code=400, detail="Failed to load or corrupt design image")

    h, w = img.shape[:2]

    # Convert to BGRA if it's 1 or 3-channel
    if len(img.shape) == 2:  # Grayscale
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    elif img.shape[2] == 3:
        img_bgr = img.copy()
        img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    else:
        # 4-channel BGRA
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img_bgra = img.copy()

    # Create flood fill mask (H+2, W+2)
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # We will flood-fill from all 4 corners
    corners = [
        (0, 0),          # Top-Left
        (w - 1, 0),      # Top-Right
        (0, h - 1),      # Bottom-Left
        (w - 1, h - 1)   # Bottom-Right
    ]

    # Use a small tolerance for flood-fill (e.g. loDiff=10, upDiff=10 per channel)
    lo_diff = (10, 10, 10)
    up_diff = (10, 10, 10)
    flags = cv2.FLOODFILL_MASK_ONLY | (255 << 8)

    for pt in corners:
        cv2.floodFill(img_bgr, ff_mask, pt, 0, lo_diff, up_diff, flags)

    # Extract the image mask part from ff_mask
    bg_mask = ff_mask[1:h+1, 1:w+1]

    # Set background pixels' alpha channel to 0 (fully transparent)
    img_bgra[bg_mask == 255, 3] = 0

    # Save the background removed image as a transparent PNG
    try:
        cv2.imwrite(target_path, img_bgra)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save background-removed image: {str(e)}")

    # Return updated preview
    success, buffer = cv2.imencode(".png", img_bgra)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image to PNG preview")

    b64_str = base64.b64encode(buffer).decode("utf-8")
    preview_url = f"data:image/png;base64,{b64_str}"

    return {
        "design_id": design_id,
        "preview": preview_url
    }

@app.post("/api/templates/ingest")
async def ingest_template(
    template_id: str = Form(...),
    category: str = Form(...),
    subtype: str = Form(...),
    label: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Admin endpoint to ingest a new base photo as a template.
    Generates mask, displacement map, lighting overlay, and metadata.json dynamically.
    """
    # Validate template_id format
    target_dir = validate_template_id(template_id)
    os.makedirs(target_dir, exist_ok=True)

    try:
        file_bytes = await file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image file")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse base photo: {str(e)}")

    h_img, w_img = img.shape[:2]

    # Save base.png
    base_path = os.path.join(target_dir, "base.png")
    cv2.imwrite(base_path, img)

    # Initialize supporting variables
    mask_img = None
    disp_img = None
    light_img = None
    corners = []
    fold_intensity = 0.0

    category_clean = category.lower().strip()
    subtype_clean = subtype.lower().strip()

    if category_clean == "tech" or subtype_clean == "laptop":
        # Screen detection logic
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            screen_contour = contours[0]
            epsilon = 0.02 * cv2.arcLength(screen_contour, True)
            approx = cv2.approxPolyDP(screen_contour, epsilon, True)
            if len(approx) == 4:
                corners_l = approx.reshape(4, 2).tolist()
                # Sort corners programmatically: TL, TR, BR, BL
                corners_l = sorted(corners_l, key=lambda p: p[0] + p[1])
                tl = corners_l[0]
                br = corners_l[3]
                other = sorted(corners_l[1:3], key=lambda p: p[0])
                bl = other[0]
                tr = other[1]
                corners = [tl, tr, br, bl]
            else:
                corners = [[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]]
        else:
            corners = [[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]]

        mask_img = binary
        disp_img = np.ones_like(gray) * 128
        light_img = np.ones_like(gray) * 128
        fold_intensity = 0.0

    elif category_clean == "apparel" or subtype_clean == "t-shirt":
        # Apparel/t-shirt detection logic
        b, g, r = cv2.split(img)
        is_gray = (np.abs(r.astype(int) - g.astype(int)) < 15) & (np.abs(g.astype(int) - b.astype(int)) < 15)
        is_light = (r > 130) & (g > 130) & (b > 130)
        tshirt_mask = (is_gray & is_light).astype(np.uint8) * 255

        # Check if t-shirt area is detected, otherwise fallback to entire image
        ys, xs = np.where(tshirt_mask > 0)
        if len(ys) > 0 and len(xs) > 0:
            min_y, max_y = ys.min(), ys.max()
            min_x, max_x = xs.min(), xs.max()

            mask_img = np.zeros_like(tshirt_mask)
            zone_w = int((max_x - min_x) * 0.45)
            zone_h = int((max_y - min_y) * 0.55)
            cx, cy = (min_x + max_x) // 2, (min_y + max_y) // 2 - 30

            z_x1, z_y1 = cx - zone_w // 2, cy - zone_h // 2
            z_x2, z_y2 = cx + zone_w // 2, cy + zone_h // 2
            cv2.rectangle(mask_img, (z_x1, z_y1), (z_x2, z_y2), 255, -1)
            mask_img = cv2.bitwise_and(mask_img, tshirt_mask)
            corners = [[z_x1, z_y1], [z_x2, z_y1], [z_x2, z_y2], [z_x1, z_y2]]
        else:
            mask_img = np.ones((h_img, w_img), dtype=np.uint8) * 255
            corners = [[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]]

        fold_intensity = 15.0
        disp_img = generate_displacement_map(img, mask_img, fold_intensity)
        light_img = generate_lighting_overlay(img, mask_img)

    else:
        # Generic/other category fallback
        # Define a centered zone (center 50% of the image)
        mask_img = np.zeros((h_img, w_img), dtype=np.uint8)
        z_w = int(w_img * 0.5)
        z_h = int(h_img * 0.5)
        z_x1 = (w_img - z_w) // 2
        z_y1 = (h_img - z_h) // 2
        z_x2 = z_x1 + z_w
        z_y2 = z_y1 + z_h
        cv2.rectangle(mask_img, (z_x1, z_y1), (z_x2, z_y2), 255, -1)
        corners = [[z_x1, z_y1], [z_x2, z_y1], [z_x2, z_y2], [z_x1, z_y2]]

        disp_img = np.ones((h_img, w_img), dtype=np.uint8) * 128
        light_img = np.ones((h_img, w_img), dtype=np.uint8) * 128
        fold_intensity = 0.0

    # Write files
    cv2.imwrite(os.path.join(target_dir, "mask.png"), mask_img)
    cv2.imwrite(os.path.join(target_dir, "displacement.png"), disp_img)
    cv2.imwrite(os.path.join(target_dir, "lighting.png"), light_img)

    metadata = {
        "id": template_id,
        "category": category_clean,
        "subtype": subtype_clean,
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
        "min_upload_resolution_px": [400, 400],
        "max_upload_resolution_px": [8000, 8000],
        "supported_formats": ["png", "jpg", "webp"],
        "export_default_format": "png",
        "export_max_resolution_px": [4096, 4096],
        "created_at": "2026-07-15",
        "engine_version": "1.0"
    }

    metadata = convert_to_serializable(metadata)
    # Ensure encoding="utf-8" explicitly as required by directives!
    with open(os.path.join(target_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    return metadata

@app.delete("/api/history/{id}")
def delete_render_history(id: int):
    """
    Deletes past render log from the SQLite database.
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
