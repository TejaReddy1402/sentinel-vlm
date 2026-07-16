import warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Dict, Tuple


def _norm(arr: np.ndarray, low_pct: float = 2, high_pct: float = 98) -> np.ndarray:
    lo, hi = np.percentile(arr[arr > 0], [low_pct, high_pct]) if np.any(arr > 0) else (0, 1)
    if hi == lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0, 1).astype(np.float32)


def make_rgb(bands: Dict[str, np.ndarray], target_size: Tuple[int, int] = (800, 800)) -> np.ndarray:
    r = _norm(bands["B04"])
    g = _norm(bands["B03"])
    b = _norm(bands["B02"])
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    img = Image.fromarray(rgb).resize(target_size, Image.LANCZOS)
    return np.array(img)


def make_false_colour(bands: Dict[str, np.ndarray], target_size: Tuple[int, int] = (800, 800)) -> np.ndarray:
    """NIR-Red-Green false colour (vegetation appears red)."""
    r = _norm(bands["B08"])
    g = _norm(bands["B04"])
    b = _norm(bands["B03"])
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    img = Image.fromarray(rgb).resize(target_size, Image.LANCZOS)
    return np.array(img)


def compute_ndvi(bands: Dict[str, np.ndarray]) -> np.ndarray:
    nir = bands["B08"].astype(np.float64)
    red = bands["B04"].astype(np.float64)
    denom = nir + red
    return np.where(denom > 0, (nir - red) / denom, 0.0).astype(np.float32)


def compute_ndwi(bands: Dict[str, np.ndarray]) -> np.ndarray:
    green = bands["B03"].astype(np.float64)
    nir = bands["B08"].astype(np.float64)
    denom = green + nir
    return np.where(denom > 0, (green - nir) / denom, 0.0).astype(np.float32)


def compute_ndbi(bands: Dict[str, np.ndarray]) -> np.ndarray:
    swir = bands["B11"].astype(np.float64)
    nir = bands["B08"].astype(np.float64)
    denom = swir + nir
    return np.where(denom > 0, (swir - nir) / denom, 0.0).astype(np.float32)


def compute_metrics(ndvi: np.ndarray, ndwi: np.ndarray, ndbi: np.ndarray) -> Dict:
    water_mask = ndwi > 0.2
    veg_mask = ndvi > 0.3
    built_mask = (ndbi > 0.0) & ~water_mask
    port_mask = (ndbi > 0.05) & (ndvi < 0.2) & ~water_mask
    road_mask = (ndbi > 0.1) & (ndwi < 0.0) & (ndvi < 0.1)
    return {
        "water_share": float(np.mean(water_mask)),
        "vegetation_share": float(np.mean(veg_mask)),
        "built_share": float(np.mean(built_mask)),
        "port_candidate_share": float(np.mean(port_mask)),
        "road_candidate_share": float(np.mean(road_mask)),
        "ndvi_mean": float(np.nanmean(ndvi)),
        "ndwi_mean": float(np.nanmean(ndwi)),
        "ndbi_mean": float(np.nanmean(ndbi)),
        "estimated_confidence": min(0.95, 0.6 + 0.35 * float(np.mean(water_mask | veg_mask | built_mask))),
    }, water_mask, veg_mask, built_mask, port_mask


def _index_to_image(index: np.ndarray, cmap_name: str, target_size: Tuple[int, int]) -> np.ndarray:
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    rgba = (cmap(norm(index)) * 255).astype(np.uint8)
    img = Image.fromarray(rgba[..., :3]).resize(target_size, Image.LANCZOS)
    return np.array(img)


def _mask_overlay(base_rgb: np.ndarray, mask: np.ndarray, colour: Tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = base_rgb.copy().astype(np.float32)
    resized_mask = np.array(
        Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (base_rgb.shape[1], base_rgb.shape[0]), Image.NEAREST
        )
    ) > 127
    for c, v in enumerate(colour):
        out[resized_mask, c] = out[resized_mask, c] * (1 - alpha) + v * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def generate_images(
    bands: Dict[str, np.ndarray],
    ndvi: np.ndarray,
    ndwi: np.ndarray,
    ndbi: np.ndarray,
    water_mask: np.ndarray,
    veg_mask: np.ndarray,
    built_mask: np.ndarray,
    port_mask: np.ndarray,
    output_dir: Path,
    size: Tuple[int, int] = (800, 800),
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    # 1. True-colour RGB
    rgb = make_rgb(bands, size)
    p = output_dir / "01_true_colour_rgb.png"
    Image.fromarray(rgb).save(p)
    paths["rgb"] = p

    # 2. False-colour NIR composite
    fc = make_false_colour(bands, size)
    p = output_dir / "02_false_colour_nir.png"
    Image.fromarray(fc).save(p)
    paths["false_colour"] = p

    # 3. NDVI
    ndvi_img = _index_to_image(ndvi, "RdYlGn", size)
    p = output_dir / "03_ndvi.png"
    Image.fromarray(ndvi_img).save(p)
    paths["ndvi"] = p

    # 4. NDWI
    ndwi_img = _index_to_image(ndwi, "Blues", size)
    p = output_dir / "04_ndwi.png"
    Image.fromarray(ndwi_img).save(p)
    paths["ndwi"] = p

    # 5. NDBI
    ndbi_img = _index_to_image(ndbi, "YlOrRd", size)
    p = output_dir / "05_ndbi.png"
    Image.fromarray(ndbi_img).save(p)
    paths["ndbi"] = p

    # 6. Fused evidence overlay (RGB + coloured mask overlays)
    fused = _mask_overlay(rgb, water_mask, (30, 144, 255))      # blue = water
    fused = _mask_overlay(fused, veg_mask, (34, 139, 34))       # green = vegetation
    fused = _mask_overlay(fused, port_mask, (255, 140, 0))      # orange = port candidates
    p = output_dir / "06_fused_evidence_overlay.png"
    Image.fromarray(fused).save(p)
    paths["fused"] = p

    # 7. Layer stack panel (all 6 thumbnails in a 2×3 grid)
    thumb_size = (380, 380)
    labels = ["True Colour", "False Colour (NIR)", "NDVI", "NDWI", "NDBI", "Fused Evidence"]
    imgs = [rgb, fc, ndvi_img, ndwi_img, ndbi_img, fused]
    panel_w, panel_h = 2 * thumb_size[0] + 30, 3 * thumb_size[1] + 60
    panel = Image.new("RGB", (panel_w, panel_h), (10, 16, 30))
    draw = ImageDraw.Draw(panel)
    for idx, (im_arr, label) in enumerate(zip(imgs, labels)):
        col, row = idx % 2, idx // 2
        x = col * (thumb_size[0] + 15) + 8
        y = row * (thumb_size[1] + 20) + 8
        thumb = Image.fromarray(im_arr).resize(thumb_size, Image.LANCZOS)
        panel.paste(thumb, (x, y))
        draw.text((x + 4, y + thumb_size[1] - 20), label, fill=(200, 230, 255))
    p = output_dir / "07_layer_stack_panel.png"
    panel.save(p)
    paths["layer_stack"] = p

    return paths
