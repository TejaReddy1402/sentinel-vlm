import base64
import io
import os
import json
from pathlib import Path
from typing import Dict, Any

from groq import Groq
from PIL import Image


SYSTEM_PROMPT = (
    "You are an expert remote sensing and geospatial analyst. "
    "You will be shown Sentinel-2 satellite imagery products: "
    "true-colour RGB, false-colour NIR composite, NDVI, NDWI, NDBI, and a fused evidence overlay. "
    "Analyse the scene carefully and return ONLY a valid JSON object — no markdown fences, no explanation. "
    "The JSON must have exactly these keys: "
    "findings (list of {type, confidence, description}), "
    "dominant_landcover (string), "
    "infrastructure_detected (list of strings), "
    "vlm_summary (2-4 sentence narrative), "
    "limitations (list of strings). "
    "Confidence values must be floats between 0 and 1."
)


def _build_prompt(metrics: Dict, meta: Dict) -> str:
    return (
        f"Scene metadata: {json.dumps(meta, indent=2)}\n\n"
        f"Computed spectral metrics:\n"
        f"  Water share (NDWI>0.2): {metrics['water_share']:.1%}\n"
        f"  Vegetation share (NDVI>0.3): {metrics['vegetation_share']:.1%}\n"
        f"  Built-up share (NDBI>0): {metrics['built_share']:.1%}\n"
        f"  Port/industrial candidate share: {metrics['port_candidate_share']:.1%}\n"
        f"  Road candidate share: {metrics['road_candidate_share']:.1%}\n"
        f"  Mean NDVI: {metrics['ndvi_mean']:.3f}\n"
        f"  Mean NDWI: {metrics['ndwi_mean']:.3f}\n"
        f"  Mean NDBI: {metrics['ndbi_mean']:.3f}\n\n"
        "The images (in order) are: True Colour RGB, NDVI, NDWI, NDBI, Fused Evidence Overlay.\n"
        "Return ONLY the JSON object."
    )


def _b64_png(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _extract_json(text: str) -> Dict:
    text = text.strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip().lstrip("json").strip()
            try:
                return json.loads(part)
            except json.JSONDecodeError:
                continue
    return json.loads(text)


def analyze_scene(
    image_paths: Dict[str, Path],
    metrics: Dict,
    scene_meta: Dict,
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
) -> Dict[str, Any]:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set — add it to your .env file")

    client = Groq(api_key=api_key)

    display_order = ["rgb", "ndvi", "ndwi", "ndbi", "fused"]  # max 5 for Groq
    content = []
    img_count = 0
    for key in display_order:
        p = image_paths.get(key)
        if p and Path(p).exists():
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_b64_png(Path(p))}"},
            })
            img_count += 1

    content.append({"type": "text", "text": _build_prompt(metrics, scene_meta)})

    print(f"  Sending {img_count} images to {model} …")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        max_tokens=4096,
        temperature=0.2,
    )

    text = response.choices[0].message.content or ""
    try:
        result = _extract_json(text)
    except json.JSONDecodeError:
        result = {
            "findings": [],
            "dominant_landcover": "unknown",
            "infrastructure_detected": [],
            "vlm_summary": text[:600],
            "limitations": ["JSON parse failed — raw text stored in vlm_summary"],
        }

    result["metrics"] = metrics
    result["scene_meta"] = scene_meta
    return result
