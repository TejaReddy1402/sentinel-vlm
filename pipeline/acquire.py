import os
import pystac_client
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import numpy as np
from typing import List, Dict, Tuple, Any

# GDAL network settings for cloud environments
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
os.environ.setdefault("GDAL_HTTP_CONNECTTIMEOUT", "30")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff")

# Earth Search (AWS/Element84) — no auth, no IP restrictions
STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

# Earth Search uses spectral names instead of band numbers
BAND_MAP = {
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B08": "nir",
    "B11": "swir16",
    "B12": "swir22",
}
BANDS_20M = {"B11", "B12"}


def find_best_scene(bbox: List[float], date_range: Tuple[str, str], max_cloud: int = 20) -> Any:
    catalog = pystac_client.Client.open(STAC_URL, timeout=60)
    search = catalog.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=f"{date_range[0]}/{date_range[1]}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
        sortby="+properties.eo:cloud_cover",
        max_items=10,
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(
            f"No Sentinel-2 scenes found for bbox={bbox} "
            f"between {date_range[0]} and {date_range[1]} with cloud<{max_cloud}%"
        )
    best = min(items, key=lambda i: i.properties.get("eo:cloud_cover", 100))
    print(f"  Scene: {best.id}  cloud={best.properties.get('eo:cloud_cover', '?')}%  date={best.datetime.date()}")
    return best


def _read_band(href: str, bbox_wgs84: List[float], target_px: int = 512) -> np.ndarray:
    with rasterio.Env(
        GDAL_HTTP_TIMEOUT=60,
        GDAL_HTTP_CONNECTTIMEOUT=30,
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_CHUNK_SIZE=10485760,   # 10 MB chunks
    ):
        with rasterio.open(href) as src:
            bbox_native = transform_bounds("EPSG:4326", src.crs, *bbox_wgs84)
            window = from_bounds(*bbox_native, transform=src.transform)
            # Downsample to at most target_px — reads from COG overview, much faster
            win_h = max(1, int(round(window.height)))
            win_w = max(1, int(round(window.width)))
            scale = min(target_px / max(win_h, win_w), 1.0)
            out_h = max(1, int(win_h * scale))
            out_w = max(1, int(win_w * scale))
            arr = src.read(
                1, window=window,
                out_shape=(out_h, out_w),
                resampling=rasterio.enums.Resampling.average,
                boundless=True, fill_value=0,
            )
    return arr.astype(np.float32)


def download_bands(item: Any, bbox: List[float], bands: List[str]) -> Tuple[Dict[str, np.ndarray], Dict]:
    print(f"  Downloading {len(bands)} bands from AWS S3 ...")
    data: Dict[str, np.ndarray] = {}
    for band in bands:
        asset_key = BAND_MAP.get(band, band)
        if asset_key not in item.assets:
            raise RuntimeError(f"Asset '{asset_key}' not found in scene. Available: {list(item.assets.keys())}")
        href = item.assets[asset_key].href
        print(f"    {band} ({asset_key}) ...", flush=True)
        arr = _read_band(href, bbox)
        data[band] = arr
        print(f"    {band} done: {arr.shape}")

    # Upsample 20m bands to match 10m resolution of B04
    ref_shape = data["B04"].shape
    for band in bands:
        if band in BANDS_20M and data[band].shape != ref_shape:
            from PIL import Image
            pil = Image.fromarray(data[band]).resize(
                (ref_shape[1], ref_shape[0]), resample=Image.BILINEAR
            )
            data[band] = np.array(pil, dtype=np.float32)

    meta = {
        "scene_id": item.id,
        "scene_date": str(item.datetime.date()),
        "cloud_cover": item.properties.get("eo:cloud_cover", 0),
        "bbox": bbox,
        "platform": item.properties.get("platform", "sentinel-2"),
        "tile": item.properties.get("s2:mgrs_tile", ""),
    }
    return data, meta


def load_scene(bbox: List[float], date_range: Tuple[str, str], max_cloud: int = 20, bands: List[str] = None) -> Tuple[Dict, Dict]:
    if bands is None:
        bands = ["B02", "B03", "B04", "B08", "B11", "B12"]
    item = find_best_scene(bbox, date_range, max_cloud)
    return download_bands(item, bbox, bands)
