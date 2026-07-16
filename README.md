---
title: Sentinel VLM Fusion
emoji: 🛰️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
---

# Sentinel-2 Optical Fusion + VLM Analysis

Real Sentinel-2 L2A satellite imagery fused with spectral indices and analysed by Llama 4 Scout vision model via Groq.

## How it works
1. Downloads real Sentinel-2 L2A bands from Microsoft Planetary Computer (free)
2. Computes true NDVI, NDWI, NDBI spectral indices
3. Generates fused evidence overlays
4. Sends imagery to Llama 4 Scout (Groq free tier) for VLM analysis
5. Renders a live HTML dashboard

## Setup (HF Spaces)
Add your Groq API key as a **Space Secret** named `GROQ_API_KEY`.
Get one free at https://console.groq.com
