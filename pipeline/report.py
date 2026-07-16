import base64
import json
from pathlib import Path
from typing import Dict, Any


def _b64_img(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _conf(v: float) -> str:
    return f"{v:.2f}"


def generate_dashboard(
    analysis: Dict[str, Any],
    image_paths: Dict[str, Path],
    output_dir: Path,
) -> Path:
    output_dir = Path(output_dir)
    metrics = analysis.get("metrics", {})
    meta = analysis.get("scene_meta", {})
    findings = analysis.get("findings", [])

    # Embed images as base64 data URIs
    def img_tag(key: str, alt: str) -> str:
        p = image_paths.get(key)
        if p and Path(p).exists():
            return f'<img src="data:image/png;base64,{_b64_img(Path(p))}" alt="{alt}">'
        return f'<p style="color:#f87">Image not found: {key}</p>'

    findings_html = ""
    for f in findings:
        c = f.get("confidence", 0)
        bar_color = "#22c55e" if c >= 0.7 else "#f59e0b" if c >= 0.4 else "#ef4444"
        findings_html += f"""
        <div class="finding">
          <div class="finding-header">
            <span class="finding-type">{f.get('type', 'Unknown')}</span>
            <span class="finding-conf" style="color:{bar_color}">{_conf(c)}</span>
          </div>
          <div class="conf-bar"><div class="conf-fill" style="width:{c*100:.0f}%;background:{bar_color}"></div></div>
          <p class="finding-desc">{f.get('description', '')}</p>
        </div>"""

    limitations = analysis.get("limitations", [])
    lim_html = "".join(f"<li>{l}</li>" for l in limitations)

    infra = analysis.get("infrastructure_detected", [])
    infra_html = "".join(f'<span class="tag">{i}</span>' for i in infra)

    analysis_json = json.dumps(analysis, indent=2)

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Sentinel-2 VLM Analysis — {meta.get('scene_date','')}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#050914;color:#edf7ff;font-family:Inter,system-ui,Arial,sans-serif;line-height:1.5}}
.wrap{{max-width:1560px;margin:auto;padding:28px 24px}}
h1{{font-size:1.7rem;font-weight:700;margin-bottom:4px}}
h2{{font-size:1.1rem;font-weight:600;margin-bottom:12px;color:#93c5fd}}
.subtitle{{color:#7ea8c9;font-size:.92rem;margin-bottom:28px}}
.meta-bar{{display:flex;gap:24px;flex-wrap:wrap;background:#0a1424;border:1px solid #1e3a5f;border-radius:14px;padding:14px 20px;margin-bottom:24px;font-size:.88rem}}
.meta-item b{{color:#60a5fa}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:14px;margin-bottom:28px}}
.metric{{background:#0e1b2e;border:1px solid #2d4b6d;border-radius:14px;padding:16px;text-align:center}}
.metric small{{display:block;color:#7ea8c9;font-size:.8rem;margin-bottom:4px}}
.metric b{{font-size:1.6rem;font-weight:700;color:#e2f0ff}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:24px}}
.grid3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px;margin-bottom:24px}}
.card{{background:#0a1425;border:1px solid #28425f;border-radius:18px;padding:16px}}
.card img{{width:100%;border-radius:12px;border:1px solid #1e3a5f;display:block}}
.wide{{grid-column:1/-1}}
.finding{{background:#0d1a2e;border:1px solid #1e3a5f;border-radius:10px;padding:14px;margin-bottom:10px}}
.finding-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}}
.finding-type{{font-weight:600;font-size:.95rem}}
.finding-conf{{font-size:1rem;font-weight:700}}
.conf-bar{{height:5px;background:#1e2d44;border-radius:3px;margin-bottom:8px}}
.conf-fill{{height:100%;border-radius:3px;transition:width .3s}}
.finding-desc{{color:#8ea7c4;font-size:.87rem}}
.tag{{display:inline-block;background:#162033;border:1px solid #2d4b6d;border-radius:8px;padding:3px 10px;margin:3px;font-size:.83rem;color:#7dd3fc}}
.summary-box{{background:#07101d;border:1px solid #1e3a5f;border-radius:14px;padding:20px;margin-bottom:16px;color:#b8d7f8;font-size:.95rem}}
pre{{white-space:pre-wrap;background:#07101d;border:1px solid #1e3a5f;border-radius:14px;padding:18px;color:#b8d7f8;font-size:.82rem;overflow:auto;max-height:500px}}
ul{{padding-left:18px;color:#8ea7c4;font-size:.88rem}}
li{{margin-bottom:4px}}
.section-title{{font-size:1.05rem;font-weight:600;color:#93c5fd;margin:24px 0 12px}}
</style></head>
<body><div class="wrap">
<h1>Sentinel-2 Optical Fusion + VLM Analysis</h1>
<div class="subtitle">Real Sentinel-2 L2A · Planetary Computer · Claude {meta.get('platform','sentinel-2').upper()} VLM</div>

<div class="meta-bar">
  <div class="meta-item"><small>Scene ID</small><b>{meta.get('scene_id','—')}</b></div>
  <div class="meta-item"><small>Date</small><b>{meta.get('scene_date','—')}</b></div>
  <div class="meta-item"><small>Cloud Cover</small><b>{meta.get('cloud_cover','—')}%</b></div>
  <div class="meta-item"><small>Tile</small><b>{meta.get('tile','—')}</b></div>
  <div class="meta-item"><small>Dominant Land Cover</small><b>{analysis.get('dominant_landcover','—')}</b></div>
</div>

<div class="metrics">
  <div class="metric"><small>Water (NDWI)</small><b>{_pct(metrics.get('water_share',0))}</b></div>
  <div class="metric"><small>Vegetation (NDVI)</small><b>{_pct(metrics.get('vegetation_share',0))}</b></div>
  <div class="metric"><small>Built-up (NDBI)</small><b>{_pct(metrics.get('built_share',0))}</b></div>
  <div class="metric"><small>Port Candidates</small><b>{_pct(metrics.get('port_candidate_share',0))}</b></div>
  <div class="metric"><small>Road Candidates</small><b>{_pct(metrics.get('road_candidate_share',0))}</b></div>
  <div class="metric"><small>Confidence</small><b>{_conf(metrics.get('estimated_confidence',0))}</b></div>
  <div class="metric"><small>Mean NDVI</small><b>{metrics.get('ndvi_mean',0):.3f}</b></div>
  <div class="metric"><small>Mean NDWI</small><b>{metrics.get('ndwi_mean',0):.3f}</b></div>
  <div class="metric"><small>Mean NDBI</small><b>{metrics.get('ndbi_mean',0):.3f}</b></div>
</div>

<div class="section-title">Layer Stack</div>
<div class="card wide" style="margin-bottom:24px">
  {img_tag('layer_stack', 'Layer Stack Panel')}
</div>

<div class="section-title">Key Images</div>
<div class="grid2">
  <div class="card"><h2>True Colour RGB</h2>{img_tag('rgb','True Colour')}</div>
  <div class="card"><h2>Fused Evidence Overlay</h2>{img_tag('fused','Fused Evidence')}</div>
</div>

<div class="grid3">
  <div class="card"><h2>NDVI (Vegetation)</h2>{img_tag('ndvi','NDVI')}</div>
  <div class="card"><h2>NDWI (Water)</h2>{img_tag('ndwi','NDWI')}</div>
  <div class="card"><h2>NDBI (Built-up)</h2>{img_tag('ndbi','NDBI')}</div>
</div>

<div class="section-title">VLM Analysis — Llama 4 via Groq</div>
<div class="summary-box">{analysis.get('vlm_summary','')}</div>

<div class="grid2" style="margin-bottom:24px">
  <div class="card">
    <h2>Findings</h2>
    {findings_html if findings_html else '<p style="color:#7ea8c9">No findings</p>'}
  </div>
  <div class="card">
    <h2>Infrastructure Detected</h2>
    <div style="margin-bottom:16px">{infra_html if infra_html else '<span style="color:#7ea8c9">None detected</span>'}</div>
    <h2>Limitations</h2>
    <ul>{lim_html}</ul>
  </div>
</div>

<div class="section-title">Full Analysis JSON</div>
<pre>{analysis_json}</pre>

</div></body></html>"""

    out_path = output_dir / "dashboard.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"  Dashboard written: {out_path}")
    return out_path
