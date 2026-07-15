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

class RenderResponse(BaseModel):
    mockup_base64: str  # Returned as base64-encoded PNG/JPEG data
    format: str
    width: int
    height: int
