import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from PIL import Image

PRESETS = {
    "Rotterdam Port, Netherlands":    [4.0,    51.8,  4.6,   52.0],
    "Port of Singapore":              [103.7,   1.2, 104.0,   1.4],
    "Port of Los Angeles, USA":      [-118.3,  33.7,-118.1,  33.8],
    "Port of Shanghai, China":        [121.8,  31.1, 122.1,  31.4],
    "Dubai Port, UAE":                 [55.1,  25.0,  55.4,  25.2],
    "Amazon Deforestation, Brazil":   [-55.5,  -5.0, -55.0,  -4.5],
    "Sahel Drylands, Niger":           [13.0,  13.5,  13.5,  14.0],
}


def apply_preset(name):
    b = PRESETS.get(name, [4.0, 51.8, 4.6, 52.0])
    return b[0], b[1], b[2], b[3]


def run_analysis(west, south, east, north, start_date, end_date, max_cloud, progress=gr.Progress(track_tqdm=True)):
    from pipeline.acquire import load_scene
    from pipeline.process import compute_ndvi, compute_ndwi, compute_ndbi, compute_metrics, generate_images
    from pipeline.vlm_analyze import analyze_scene
    from pipeline.report import generate_dashboard

    output_dir = Path(tempfile.mkdtemp(prefix="sentinel_"))
    bbox = [float(west), float(south), float(east), float(north)]

    # --- Stage 1: acquire ---
    progress(0.05, desc="Searching Earth Search (AWS) for best Sentinel-2 scene...")
    try:
        bands, meta = load_scene(
            bbox=bbox,
            date_range=(start_date, end_date),
            max_cloud=int(max_cloud),
        )
    except RuntimeError as e:
        raise gr.Error(str(e))

    scene_info = (
        f"**Scene:** {meta['scene_id']}  \n"
        f"**Date:** {meta['scene_date']}  |  "
        f"**Cloud:** {meta['cloud_cover']:.1f}%  |  "
        f"**Tile:** {meta.get('tile','')}"
    )

    # --- Stage 2: process ---
    progress(0.35, desc="Computing NDVI / NDWI / NDBI ...")
    ndvi = compute_ndvi(bands)
    ndwi = compute_ndwi(bands)
    ndbi = compute_ndbi(bands)
    metrics, water_mask, veg_mask, built_mask, port_mask = compute_metrics(ndvi, ndwi, ndbi)

    progress(0.50, desc="Generating spectral images ...")
    image_paths = generate_images(
        bands=bands, ndvi=ndvi, ndwi=ndwi, ndbi=ndbi,
        water_mask=water_mask, veg_mask=veg_mask,
        built_mask=built_mask, port_mask=port_mask,
        output_dir=output_dir,
    )

    # --- Stage 3: VLM ---
    progress(0.65, desc="Running Llama 4 Scout vision analysis via Groq ...")
    analysis = analyze_scene(image_paths=image_paths, metrics=metrics, scene_meta=meta)

    # --- Stage 4: dashboard ---
    progress(0.90, desc="Building HTML dashboard ...")
    dashboard_path = generate_dashboard(
        analysis=analysis, image_paths=image_paths, output_dir=output_dir
    )

    progress(1.0, desc="Done!")

    metrics_display = {
        "Water share":            f"{metrics['water_share']:.1%}",
        "Vegetation share":       f"{metrics['vegetation_share']:.1%}",
        "Built-up share":         f"{metrics['built_share']:.1%}",
        "Port candidates":        f"{metrics['port_candidate_share']:.1%}",
        "Road candidates":        f"{metrics['road_candidate_share']:.1%}",
        "Mean NDVI":              f"{metrics['ndvi_mean']:.3f}",
        "Mean NDWI":              f"{metrics['ndwi_mean']:.3f}",
        "Mean NDBI":              f"{metrics['ndbi_mean']:.3f}",
        "Confidence":             f"{metrics['estimated_confidence']:.2f}",
    }

    findings_text = ""
    for f in analysis.get("findings", []):
        bar = int(f.get("confidence", 0) * 10)
        findings_text += (
            f"**{f.get('type','')}** — conf: {f.get('confidence',0):.2f}\n"
            f"{'█' * bar}{'░' * (10 - bar)}\n"
            f"{f.get('description','')}\n\n"
        )

    infra = ", ".join(analysis.get("infrastructure_detected", [])) or "None detected"
    lc = analysis.get("dominant_landcover", "—")

    summary_md = (
        f"### Dominant Land Cover\n{lc}\n\n"
        f"### Infrastructure Detected\n{infra}\n\n"
        f"### VLM Summary\n{analysis.get('vlm_summary','')}\n\n"
        f"### Limitations\n" +
        "\n".join(f"- {l}" for l in analysis.get("limitations", []))
    )

    dashboard_html = dashboard_path.read_text(encoding="utf-8")

    def load_img(key):
        p = image_paths.get(key)
        if p and Path(p).exists():
            return Image.open(p).convert("RGB")
        return None

    return (
        scene_info,
        load_img("rgb"),
        load_img("fused"),
        load_img("ndvi"),
        load_img("ndwi"),
        load_img("ndbi"),
        metrics_display,
        findings_text,
        summary_md,
        dashboard_html,
    )


# ── UI ────────────────────────────────────────────────────────────────────────
theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate",
).set(
    body_background_fill="#050914",
    body_text_color="#edf7ff",
    block_background_fill="#0a1425",
    block_border_color="#28425f",
    input_background_fill="#0e1b2e",
)

custom_css = """
/* dropdown list items */
.gr-dropdown ul, .gr-dropdown li,
ul[role="listbox"], ul[role="listbox"] li,
div[data-testid="dropdown"] ul li,
.svelte-select-list, .svelte-select-list li,
.option { color: #edf7ff !important; background-color: #0e1b2e !important; }

/* dropdown selected value */
.svelte-select input, .svelte-select .value-container,
div[data-testid="dropdown"] input,
div[data-testid="dropdown"] .wrap { color: #edf7ff !important; }

/* dropdown container */
div[data-testid="dropdown"] .wrap,
.svelte-select { background-color: #0e1b2e !important; border-color: #28425f !important; }

/* hover state */
ul[role="listbox"] li:hover,
.option:hover { background-color: #1e3a5f !important; }
"""

with gr.Blocks(title="Sentinel-2 VLM Fusion") as demo:
    gr.Markdown(
        """
        # Sentinel-2 Optical Fusion + VLM Analysis
        Real Sentinel-2 L2A data from Microsoft Planetary Computer
        Vision analysis by **Llama 4 Scout** via Groq (free)
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Area of Interest")

            preset = gr.Dropdown(
                choices=list(PRESETS.keys()),
                label="Quick Presets",
                value="Rotterdam Port, Netherlands",
            )

            with gr.Row():
                west  = gr.Number(label="West",  value=4.0,  precision=4)
                south = gr.Number(label="South", value=51.8, precision=4)
            with gr.Row():
                east  = gr.Number(label="East",  value=4.6,  precision=4)
                north = gr.Number(label="North", value=52.0, precision=4)

            preset.change(apply_preset, inputs=preset, outputs=[west, south, east, north])

            gr.Markdown("### Date Range")
            with gr.Row():
                start_date = gr.Textbox(label="Start Date", value="2023-06-01", placeholder="YYYY-MM-DD")
                end_date   = gr.Textbox(label="End Date",   value="2023-08-31", placeholder="YYYY-MM-DD")

            max_cloud = gr.Slider(0, 60, value=20, step=5, label="Max Cloud Cover %")

            run_btn = gr.Button("Run Analysis", variant="primary", size="lg")

        with gr.Column(scale=2):
            scene_info = gr.Markdown("*Run the pipeline to see scene metadata.*")
            metrics_out = gr.JSON(label="Spectral Metrics")

    gr.Markdown("---")
    gr.Markdown("### Imagery")
    with gr.Row():
        img_rgb   = gr.Image(label="True Colour RGB",      show_label=True)
        img_fused = gr.Image(label="Fused Evidence Overlay", show_label=True)
    with gr.Row():
        img_ndvi = gr.Image(label="NDVI (Vegetation)", show_label=True)
        img_ndwi = gr.Image(label="NDWI (Water)",      show_label=True)
        img_ndbi = gr.Image(label="NDBI (Built-up)",   show_label=True)

    gr.Markdown("---")
    gr.Markdown("### VLM Analysis — Llama 4 Scout")
    with gr.Row():
        findings_out = gr.Markdown(label="Findings")
        summary_out  = gr.Markdown(label="Summary & Limitations")

    gr.Markdown("---")
    gr.Markdown("### Full Dashboard")
    dashboard_out = gr.HTML(label="Dashboard")

    run_btn.click(
        fn=run_analysis,
        inputs=[west, south, east, north, start_date, end_date, max_cloud],
        outputs=[
            scene_info,
            img_rgb, img_fused,
            img_ndvi, img_ndwi, img_ndbi,
            metrics_out,
            findings_out,
            summary_out,
            dashboard_out,
        ],
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=theme,
        css=custom_css,
    )
