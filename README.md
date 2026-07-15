# 🎨 Crevr Mockup Generator — Single Source of Truth

Welcome to **Crevr**, a local-first, highly-deterministic **image compositing engine** designed for high-resolution product mockups (t-shirts, laptop screens, mugs, and more).

This tool uses **deterministic OpenCV and NumPy pixel math** (perspective warping, Sobel-gradient displacement mapping, photometric blending, and LAB color mapping) to produce realistic mockup renders instantly and entirely offline.

---

## 🚀 Quick Start (Chalu Kaise Karein)

### Step 1: Virtual Environment Set Karein (First-time only)
Apne terminal mein yeh commands run karke environment setup karein:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

---

### Step 2: Running the Ingestion Pipeline (Template Kaise Banayein)
Agar aapko blank product photos ko valid ready-to-use templates mein convert karna hai (base masks, neutral height maps, lighting layers and metadata generator), toh simply yeh command run karein:

```bash
PYTHONPATH=. python3 engine/pipeline/ingest.py
```
*Yeh automatic templates directory generate karke saare images load kar dega.*

---

### Step 3: Run the FastAPI Server (Backend Start Karein)
Local backend engine ko port **8015** par start karne ke liye:

```bash
PYTHONPATH=. uvicorn engine.main:app --host 0.0.0.0 --port 8015
```

Uvicorn server running state mein aa jayega.

---

### Step 4: Open the Frontend (Frontend Kaise Chalayein)
Apna browser open karein aur is URL par chale jao:
👉 **[http://localhost:8015](http://localhost:8015)**

Yahan par aapko complete UI mil jayega jahan se aap:
- Grid view se template select kar sakte hain.
- Custom artwork images (.png, .webp, .jpg) drag-drop ya browse karke upload kar sakte hain.
- Sliders use karke: **Corner Radius (Screens)**, **Perspective Tilt / Bend**, **Fold Intensity (Displacement)**, **Edge Feathering** controls set kar sakte hain.
- **Garment Color Swatches** aur custom hex color picker se white t-shirts ka background color non-destructively badal sakte hain.
- **Remove Solid Background** toggle on karke JPEG graphics se solid background automatic chroma-key algorithm se clean kar sakte hain.
- **Undo / Redo / Reset** handles se layers manipulate kar sakte hain.
- **Crop Tool Modal** open karke design crop kar sakte hain.
- **High-Resolution Export** option se format (PNG, JPG, WEBP) select karke directly print-quality DPI choose karke output download kar sakte hain.

---

## 🧪 Testing Suite (Test Kaise Run Karein)

Humaare 10 mathematical test cases aur endpoint integration checks run karne ke liye commands follow karein:

```bash
PYTHONPATH=. pytest engine/tests/
```

Saare test cases automatically pass ho jayenge.

---

## 📂 Folder Structure Overview

- **`engine/`**: FastAPI app (`main.py`) aur mathematical processing modules (`engine/pipeline/` - warp, displacement, blend, mask, render, ingest).
- **`frontend/`**: Pure Tailwind/React/Fabric.js SPA (`index.html`) served statically via backend.
- **`templates/`**: Ingested ready-to-composite templates directory with assets.
- **`data/`**: Past rendering logs history database SQLite (`crevr.db`).
