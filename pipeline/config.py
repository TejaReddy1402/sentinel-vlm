from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class PipelineConfig:
    # Area of interest [west, south, east, north] in WGS-84
    bbox: List[float] = field(default_factory=lambda: [4.0, 51.8, 4.6, 52.0])
    # Date range for scene search
    start_date: str = "2023-06-01"
    end_date: str = "2023-08-31"
    # Maximum cloud cover percentage
    max_cloud: int = 20
    # Sentinel-2 bands to download
    bands: List[str] = field(default_factory=lambda: ["B02", "B03", "B04", "B08", "B11", "B12"])
    # Claude model
    model: str = "gemini-1.5-flash"
    # Output directory
    output_dir: str = "outputs"
    # Target image size for VLM
    image_size: Tuple[int, int] = (800, 800)

DEFAULT_CONFIG = PipelineConfig()
