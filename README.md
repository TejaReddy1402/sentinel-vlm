---
title: Sentinel VLM Fusion
emoji: 🛰️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.20.0
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
---

# Sentinel-2 Optical Fusion + VLM Analysis

> Real satellite intelligence — download, process, and AI-analyse any location on Earth for free.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-blue)](https://sentinel-vlm.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Gradio](https://img.shields.io/badge/UI-Gradio%206-orange)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## What It Does

Pick any location on Earth and a date range. The pipeline:

1. **Acquires** real Sentinel-2 L2A multispectral satellite bands (Blue, Green, Red, NIR, SWIR1, SWIR2) from Microsoft / AWS Earth Search — completely free
2. **Computes** true spectral indices — NDVI (vegetation), NDWI (water), NDBI (built-up surfaces)
3. **Generates** scientific false-colour composites and a fused evidence overlay
4. **Analyses** the imagery with Llama 4 Scout (via Groq) — a real vision model that has never seen this scene before
5. **Renders** a self-contained HTML dashboard with all outputs

All free. No paid APIs. No fake data.

---

## Demo

| True Colour RGB | Fused Evidence Overlay |
|---|---|
| Real Sentinel-2 scene | Water (blue) · Vegetation (green) · Port candidates (orange) |

**Preset locations included:** Rotterdam Port · Singapore · Port of LA · Shanghai · Dubai · Amazon · Sahel

---

## Architecture

```
User Input (bbox + date range)
        │
        ▼
┌─────────────────────────────┐
│  ACQUIRE                    │
│  Earth Search STAC (AWS)    │  ← free, no auth, no IP blocks
│  6 Sentinel-2 L2A bands     │
│  COG windowed read @ 512px  │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  PROCESS                    │
│  NDVI = (NIR-Red)/(NIR+Red) │  ← true spectral science
│  NDWI = (Green-NIR)/(G+NIR) │
│  NDBI = (SWIR-NIR)/(S+NIR)  │
│  7 output images generated  │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  VLM ANALYSIS               │
│  Llama 4 Scout via Groq     │  ← free vision AI
│  5 images → JSON findings   │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  DASHBOARD                  │
│  Self-contained HTML report │
│  Base64 embedded images     │
└─────────────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Satellite data | Sentinel-2 L2A (ESA) |
| Data access | Earth Search STAC API (Element84 / AWS) |
| Band reading | Rasterio + GDAL (COG windowed reads) |
| Spectral science | NumPy |
| Visualisation | Pillow + Matplotlib |
| Vision AI | Llama 4 Scout 17B via Groq API |
| Web UI | Gradio 6 |
| Hosting | Render (free tier) |

---

## Run Locally

```bash
git clone https://github.com/TejaReddy1402/sentinel-vlm
cd sentinel-vlm

pip install -r requirements.txt

# Add your free Groq API key (https://console.groq.com)
echo "GROQ_API_KEY=gsk_..." > .env

# Run Gradio UI
python app.py

# Or run the CLI pipeline directly
python run_pipeline.py --bbox 4.0 51.8 4.6 52.0 --start-date 2023-06-01 --end-date 2023-08-31
```

---

## CLI Usage

```bash
# Rotterdam port (default)
python run_pipeline.py

# Any location — get bbox from bboxfinder.com
python run_pipeline.py --bbox -118.3 33.7 -118.1 33.8 --start-date 2024-01-01 --end-date 2024-06-30

# Skip VLM call (fast test)
python run_pipeline.py --skip-vlm
```

---

## Real-World Use Cases

| Use Case | How |
|---|---|
| Port / infrastructure monitoring | Detect new construction, vessel presence |
| Flood mapping | NDWI spike shows inundation extent |
| Deforestation tracking | NDVI drop over time shows forest loss |
| Urban sprawl analysis | NDBI growth shows new development |
| Crop stress monitoring | NDVI anomalies reveal stressed fields |
| Disaster response | Before/after scene comparison |

---

## Project Structure

```
sentinel-vlm/
├── app.py               # Gradio web UI
├── run_pipeline.py      # CLI entry point
├── requirements.txt
├── render.yaml          # Render deployment config
└── pipeline/
    ├── acquire.py       # Sentinel-2 data download (Earth Search)
    ├── process.py       # Spectral indices + image generation
    ├── vlm_analyze.py   # Groq / Llama 4 Scout vision analysis
    ├── report.py        # HTML dashboard generation
    └── config.py        # Default configuration
```

---

## Deployment

### Render (free)
1. Fork this repo
2. Create a new Web Service on [render.com](https://render.com) connected to your fork
3. Add environment variable: `GROQ_API_KEY=your_key`
4. Deploy — `render.yaml` handles the rest

### Get a free Groq API key
Sign up at [console.groq.com](https://console.groq.com) — no credit card required.

---

## License

MIT — free to use, modify, and deploy.
