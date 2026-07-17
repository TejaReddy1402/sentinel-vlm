import os
import pystac_client
import planetary_computer
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import numpy as np
from typing import List, Dict, Tuple, Any

# Set GDAL/CURL network timeouts for cloud environments
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
os.environ.setdefault("GDAL_HTTP_CONNECTTIMEOUT", "30")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

BANDS_10M = {"B02", "B03", "B04", "B08"}
BANDS_20M = {"B11", "B12"}
STAC_TIMEOUT = 60  # seconds


def find_best_scene(bbox: List[float], date_range: Tuple[str, str], max_cloud: int = 20) -> Any:
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
        timeout=STAC_TIMEOUT,
    )
    search = catalog.search(
        collections=["sentinel-2-l2a"],
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
    print(f"  Selected scene: {best.id}  cloud={best.properties.get('eo:cloud_cover', '?')}%  date={best.datetime.date()}")
    return best


def _read_band(href: str, bbox_wgs84: List[float]) -> np.ndarray:
    env = rasterio.Env(
        GDAL_HTTP_TIMEOUT=60,
        GDAL_HTTP_CONNECTTIMEOUT=30,
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
    )
    with env:
        with rasterio.open(href) as src:
            bbox_native = transform_bounds("EPSG:4326", src.crs, *bbox_wgs84)
            window = from_bounds(*bbox_native, transform=src.transform)
            arr = src.read(1, window=window, boundless=True, fill_value=0)
    return arr.astype(np.float32)


def download_bands(item: Any, bbox: List[float], bands: List[str]) -> Tuple[Dict[str, np.ndarray], Dict]:
    print(f"  Downloading {len(bands)} bands ...")
    data: Dict[str, np.ndarray] = {}
    for band in bands:
        print(f"    {band} ...", flush=True)
        href = item.assets[band].href
        arr = _read_band(href, bbox)
        data[band] = arr
        print(f"    {band} done: {arr.shape}")

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
