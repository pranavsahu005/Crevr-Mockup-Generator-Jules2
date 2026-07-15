from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class TransformParams(BaseModel):
    scale: float = 1.0
    rotation: float = 0.0
    offset_x: int = 0
    offset_y: int = 0

class RenderRequest(BaseModel):
    template_id: str
    design_base64: str  # Data URL or standard base64 string
    transform_params: Optional[TransformParams] = None
    dst_corners: Optional[List[List[float]]] = None  # High-res coordinates computed dynamically from client
    blend_mode: Optional[str] = "multiply"
    color_correct: Optional[bool] = True
    feather_radius: Optional[int] = 5
    fold_intensity: Optional[float] = None  # Will default to template's setting if None
    export_format: Optional[str] = "png"
    dpi: Optional[int] = 300
    physical_size_mm: Optional[List[float]] = None

    # Phase 2 Enhancements
    garment_color: Optional[str] = None  # HEX code (e.g. "#FF5733") or None
    corner_radius: Optional[float] = 0.0  # Rounds corners, e.g. 0 to 50 pixels
    perspective_bend: Optional[float] = 0.0  # Adjusts top vs bottom width tilt factor (-50 to 50)
    remove_background: Optional[bool] = False  # Auto-chromakey solid background for designs

class RenderResponse(BaseModel):
    mockup_base64: str  # Returned as base64-encoded PNG/JPEG data
    format: str
    width: int
    height: int
