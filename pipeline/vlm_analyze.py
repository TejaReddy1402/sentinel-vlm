import os
import json
from typing import Dict, Any

from groq import Groq


SYSTEM_PROMPT = (
    "You are an expert remote sensing and geospatial analyst. "
    "You will receive Sentinel-2 L2A spectral metrics and scene metadata for a location on Earth. "
    "Using the quantitative spectral indices (NDVI, NDWI, NDBI) and derived land-cover fractions, "
    "produce a rigorous scientific analysis of the scene. "
    "Return ONLY a valid JSON object — no markdown fences, no explanation. "
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
        f"Scene metadata:\n{json.dumps(meta, indent=2)}\n\n"
        f"Spectral metrics derived from Sentinel-2 L2A bands:\n"
        f"  Water fraction (NDWI > 0.2):          {metrics['water_share']:.1%}\n"
        f"  Vegetation fraction (NDVI > 0.3):     {metrics['vegetation_share']:.1%}\n"
        f"  Built-up fraction (NDBI > 0):         {metrics['built_share']:.1%}\n"
        f"  Port/industrial candidate fraction:   {metrics['port_candidate_share']:.1%}\n"
        f"  Road/linear feature fraction:         {metrics['road_candidate_share']:.1%}\n"
        f"  Mean NDVI (vegetation index):         {metrics['ndvi_mean']:.4f}\n"
        f"  Mean NDWI (water index):              {metrics['ndwi_mean']:.4f}\n"
        f"  Mean NDBI (built-up index):           {metrics['ndbi_mean']:.4f}\n"
        f"  Estimated analysis confidence:        {metrics['estimated_confidence']:.4f}\n\n"
        "NDVI interpretation: >0.6 dense forest/crops, 0.3-0.6 sparse veg/grassland, "
        "0.1-0.3 bare soil, <0.1 water/urban/rock.\n"
        "NDWI interpretation: >0.2 open water, -0.2 to 0.2 mixed, <-0.2 dry land.\n"
        "NDBI interpretation: >0.1 high-density urban/industrial, 0-0.1 suburban, <0 vegetated.\n\n"
        "Based on these spectral signatures and the known geography of the bounding box, "
        "provide a detailed scientific analysis. Return ONLY the JSON object."
    )


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
    image_paths: Dict,
    metrics: Dict,
    scene_meta: Dict,
    model: str = "llama-3.3-70b-versatile",
) -> Dict[str, Any]:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set — add it to your .env file")

    client = Groq(api_key=api_key)

    prompt = _build_prompt(metrics, scene_meta)
    print(f"  Sending spectral metrics to {model} ...")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
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
