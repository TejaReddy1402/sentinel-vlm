"""
Sentinel-2 real-data VLM pipeline.

Usage:
  python run_pipeline.py
  python run_pipeline.py --bbox 4.0 51.8 4.6 52.0 --start-date 2023-06-01 --end-date 2023-08-31
  python run_pipeline.py --bbox -87.7 41.8 -87.5 41.95 --start-date 2023-07-01 --end-date 2023-09-30
"""
import argparse
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


def parse_args():
    p = argparse.ArgumentParser(description="Sentinel-2 Optical Fusion + VLM Pipeline")
    p.add_argument(
        "--bbox", nargs=4, type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        default=[4.0, 51.8, 4.6, 52.0],
        help="Bounding box in WGS-84 (default: Rotterdam port)",
    )
    p.add_argument("--start-date", default="2023-06-01")
    p.add_argument("--end-date", default="2023-08-31")
    p.add_argument("--max-cloud", type=int, default=20, help="Max cloud cover %%")
    p.add_argument("--model", default="llama-3.3-70b-versatile")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--skip-vlm", action="store_true", help="Skip Claude VLM call (faster testing)")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== [1/4] Acquiring Sentinel-2 scene from Earth Search (AWS) ===")
    from pipeline.acquire import load_scene
    bands, meta = load_scene(
        bbox=args.bbox,
        date_range=(args.start_date, args.end_date),
        max_cloud=args.max_cloud,
        bands=["B02", "B03", "B04", "B08", "B11", "B12"],
    )
    print(f"  Acquired: {meta['scene_id']}")

    print("\n=== [2/4] Computing spectral indices and generating images ===")
    from pipeline.process import compute_ndvi, compute_ndwi, compute_ndbi, compute_metrics, generate_images
    ndvi = compute_ndvi(bands)
    ndwi = compute_ndwi(bands)
    ndbi = compute_ndbi(bands)
    metrics, water_mask, veg_mask, built_mask, port_mask = compute_metrics(ndvi, ndwi, ndbi)

    print(f"  Water: {metrics['water_share']:.1%}  Vegetation: {metrics['vegetation_share']:.1%}  "
          f"Built: {metrics['built_share']:.1%}  Port candidates: {metrics['port_candidate_share']:.1%}")

    image_paths = generate_images(
        bands=bands,
        ndvi=ndvi, ndwi=ndwi, ndbi=ndbi,
        water_mask=water_mask, veg_mask=veg_mask, built_mask=built_mask, port_mask=port_mask,
        output_dir=output_dir,
    )
    print(f"  Generated {len(image_paths)} images in {output_dir}/")

    if args.skip_vlm:
        print("\n=== [3/4] Skipping VLM analysis (--skip-vlm) ===")
        analysis = {
            "findings": [],
            "dominant_landcover": "N/A (skipped)",
            "infrastructure_detected": [],
            "vlm_summary": "VLM analysis skipped.",
            "limitations": ["VLM analysis was skipped via --skip-vlm flag."],
            "metrics": metrics,
            "scene_meta": meta,
        }
    else:
        print(f"\n=== [3/4] Running Claude VLM analysis ({args.model}) ===")
        from pipeline.vlm_analyze import analyze_scene
        analysis = analyze_scene(
            image_paths=image_paths,
            metrics=metrics,
            scene_meta=meta,
            model=args.model,
        )

    # Save JSON analysis
    json_path = output_dir / "analysis.json"
    json_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"  Analysis saved: {json_path}")

    print("\n=== [4/4] Generating HTML dashboard ===")
    from pipeline.report import generate_dashboard
    dashboard_path = generate_dashboard(
        analysis=analysis,
        image_paths=image_paths,
        output_dir=output_dir,
    )

    print(f"\n=== Done! ===")
    print(f"  Dashboard: {dashboard_path.resolve()}")
    print(f"  Open in browser:  http://localhost:8787/outputs/dashboard.html")
    print(f"  Or directly:      {dashboard_path.resolve().as_uri()}")


if __name__ == "__main__":
    main()
