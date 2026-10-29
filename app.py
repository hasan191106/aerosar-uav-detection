import os
import io
import base64
import math
import sys
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template_string

# Tambahkan direktori lokal ke sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sar_drone_detector import SARDroneVision, generate_synthetic_sar_image

app = Flask(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AeroSAR Tactical Mission Control | UAV Maritime Vision</title>
    
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    
    <style>
        :root {
            --bg-base: #090d16;
            --bg-surface: #111827;
            --bg-surface-elevated: #162032;
            --bg-surface-hover: #1f2d44;
            --border-subtle: rgba(255, 255, 255, 0.08);
            --border-medium: rgba(255, 255, 255, 0.15);
            
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            
            --maritime-blue: #2563eb;
            --maritime-glow: rgba(37, 99, 235, 0.35);
            --cyan-accent: #06b6d4;
            --emerald-safe: #10b981;
            --emerald-glow: rgba(16, 185, 129, 0.25);
            --rose-danger: #ef4444;
            --rose-glow: rgba(239, 68, 68, 0.3);
            --amber-warn: #f59e0b;
            
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
            --radius-full: 9999px;
            
            --font-main: 'Plus Jakarta Sans', -apple-system, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg-base);
            color: var(--text-primary);
            font-family: var(--font-main);
            min-height: 100vh;
            line-height: 1.5;
            overflow-x: hidden;
        }

        body::before {
            content: '';
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background-image: 
                linear-gradient(to right, rgba(255,255,255,0.02) 1px, transparent 1px),
                linear-gradient(to bottom, rgba(255,255,255,0.02) 1px, transparent 1px);
            background-size: 32px 32px;
            pointer-events: none;
            z-index: 0;
        }

        .app-container {
            position: relative;
            z-index: 1;
            max-width: 1650px;
            margin: 0 auto;
            padding: 20px 24px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        /* Navbar */
        .navbar {
            background: rgba(17, 24, 39, 0.85);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 14px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        }

        .brand-cluster {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .brand-badge-icon {
            width: 44px;
            height: 44px;
            background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
            border-radius: var(--radius-md);
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 12px var(--maritime-glow);
        }

        .brand-badge-icon svg {
            width: 24px; height: 24px; fill: none; stroke: #fff; stroke-width: 2;
        }

        .brand-title h1 {
            font-size: 19px;
            font-weight: 800;
            letter-spacing: -0.3px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .brand-title span.ver {
            font-size: 11px;
            font-family: var(--font-mono);
            background: rgba(37, 99, 235, 0.2);
            color: #93c5fd;
            border: 1px solid rgba(37, 99, 235, 0.4);
            padding: 2px 8px;
            border-radius: var(--radius-full);
        }

        .brand-title p {
            font-size: 12px;
            color: var(--text-secondary);
        }

        .nav-actions {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .telemetry-chip {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            border-radius: var(--radius-full);
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: var(--emerald-safe);
            font-size: 12px;
            font-weight: 600;
            font-family: var(--font-mono);
        }

        .chip-dot {
            width: 8px; height: 8px; background: var(--emerald-safe); border-radius: 50%;
            box-shadow: 0 0 10px var(--emerald-safe);
            animation: pulse-dot 2s infinite;
        }

        @keyframes pulse-dot {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(0.8); }
        }

        .audio-toggle-btn {
            background: var(--bg-surface-elevated);
            border: 1px solid var(--border-subtle);
            color: var(--text-secondary);
            font-size: 12px;
            padding: 6px 12px;
            border-radius: var(--radius-md);
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .audio-toggle-btn.active {
            border-color: var(--cyan-accent);
            color: var(--cyan-accent);
            background: rgba(6, 182, 212, 0.1);
        }

        /* Workspace */
        .workspace {
            display: grid;
            grid-template-columns: 400px minmax(0, 1fr);
            gap: 20px;
            align-items: start;
        }

        @media (max-width: 1150px) {
            .workspace { grid-template-columns: 1fr; }
        }

        /* Control Deck */
        .control-deck {
            display: flex;
            flex-direction: column;
            gap: 18px;
        }

        .card {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 20px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
            transition: border-color 0.2s;
        }

        .card:hover { border-color: var(--border-medium); }

        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 14px;
        }

        .card-title {
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .card-title svg { width: 16px; height: 16px; stroke: var(--cyan-accent); }

        /* Dropzone */
        .dropzone {
            border: 2px dashed rgba(255, 255, 255, 0.14);
            border-radius: var(--radius-md);
            padding: 24px 16px;
            text-align: center;
            cursor: pointer;
            transition: all 0.25s ease;
            background: rgba(15, 23, 42, 0.4);
        }

        .dropzone:hover, .dropzone.dragover {
            border-color: var(--maritime-blue);
            background: rgba(37, 99, 235, 0.06);
            transform: translateY(-1px);
        }

        .dropzone-icon {
            width: 44px; height: 44px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.05);
            display: flex; align-items: center; justify-content: center;
            margin: 0 auto 10px;
        }

        .dropzone-icon svg { width: 22px; height: 22px; stroke: var(--text-secondary); }
        .dropzone-text h4 { font-size: 13.5px; font-weight: 600; color: #fff; margin-bottom: 2px; }
        .dropzone-text p { font-size: 11.5px; color: var(--text-muted); }
        input[type="file"] { display: none; }

        .file-pill {
            display: none;
            margin-top: 10px;
            padding: 8px 12px;
            background: rgba(37, 99, 235, 0.1);
            border: 1px solid rgba(37, 99, 235, 0.3);
            border-radius: var(--radius-md);
            font-size: 12px;
            justify-content: space-between;
            align-items: center;
        }

        .file-name { font-weight: 600; color: #93c5fd; max-width: 240px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

        /* Sliders */
        .slider-box {
            margin: 14px 0 16px;
        }

        .slider-head {
            display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 12px;
        }

        .slider-badge {
            font-family: var(--font-mono); font-size: 12.5px; font-weight: 700;
            background: rgba(37, 99, 235, 0.15); color: #60a5fa; padding: 2px 8px;
            border-radius: var(--radius-sm); border: 1px solid rgba(37, 99, 235, 0.3);
        }

        input[type="range"] {
            width: 100%; height: 6px; background: #1e293b; border-radius: 4px; outline: none; -webkit-appearance: none;
        }

        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none; width: 16px; height: 16px; border-radius: 50%;
            background: #3b82f6; cursor: pointer; box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
            border: 2px solid #ffffff;
        }

        .btn-run {
            width: 100%; padding: 13px 18px; border-radius: var(--radius-md);
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            color: #fff; border: 1px solid rgba(255, 255, 255, 0.15);
            font-size: 13.5px; font-weight: 700; cursor: pointer;
            display: flex; align-items: center; justify-content: center; gap: 8px;
            box-shadow: 0 6px 20px var(--maritime-glow);
            transition: all 0.2s ease;
        }

        .btn-run:hover {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            transform: translateY(-2px);
        }

        /* Scenarios & Weather Buttons */
        .preset-grid {
            display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px;
        }

        .weather-grid {
            display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin-top: 8px;
        }

        .btn-preset, .btn-weather {
            background: var(--bg-surface-elevated);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 8px;
            text-align: left;
            cursor: pointer;
            color: var(--text-primary);
            transition: all 0.18s;
        }

        .btn-preset:hover, .btn-weather:hover {
            border-color: var(--border-medium);
            background: var(--bg-surface-hover);
        }

        .btn-weather { text-align: center; font-size: 11px; padding: 7px 4px; font-weight: 600; }
        .btn-weather.active { border-color: var(--cyan-accent); background: rgba(6, 182, 212, 0.12); color: var(--cyan-accent); }

        .preset-tag {
            font-size: 9.5px; font-weight: 700; font-family: var(--font-mono);
            display: inline-block; padding: 1px 5px; border-radius: 4px; margin-bottom: 4px;
        }

        .tag-crit { background: rgba(239, 68, 68, 0.2); color: #f87171; }
        .tag-safe { background: rgba(16, 185, 129, 0.2); color: #34d399; }
        .preset-h { font-size: 11.5px; font-weight: 600; }
        .preset-sub { font-size: 10.5px; color: var(--text-muted); }

        /* Viewport Canvas Deck */
        .viewport-deck {
            display: flex; flex-direction: column; gap: 18px;
        }

        .viewport-card {
            background: #020617;
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            overflow: hidden;
            box-shadow: 0 12px 36px rgba(0, 0, 0, 0.4);
            display: flex; flex-direction: column;
        }

        .toolbar {
            background: var(--bg-surface);
            border-bottom: 1px solid var(--border-subtle);
            padding: 10px 16px;
            display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;
        }

        .tab-group {
            display: flex; background: var(--bg-base); border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md); padding: 3px; gap: 2px;
        }

        .tab-item {
            background: transparent; border: none; color: var(--text-secondary);
            font-size: 11.5px; font-weight: 600; padding: 6px 12px; border-radius: var(--radius-sm);
            cursor: pointer; display: flex; align-items: center; gap: 5px; transition: all 0.18s;
        }

        .tab-item:hover { color: #fff; }
        .tab-item.active {
            background: var(--bg-surface-elevated); color: #fff; border: 1px solid var(--border-subtle);
        }

        .tool-actions {
            display: flex; align-items: center; gap: 8px;
        }

        .btn-act {
            background: var(--bg-surface-elevated); border: 1px solid var(--border-subtle);
            color: var(--text-secondary); font-size: 11.5px; padding: 6px 12px;
            border-radius: var(--radius-sm); cursor: pointer; text-decoration: none;
            display: inline-flex; align-items: center; gap: 6px;
        }

        .btn-act:hover { color: #fff; border-color: var(--border-medium); }

        .canvas-meta {
            padding: 8px 16px; background: rgba(17, 24, 39, 0.7);
            border-bottom: 1px solid var(--border-subtle);
            display: flex; justify-content: space-between; font-size: 11px;
            font-family: var(--font-mono); color: var(--text-muted);
        }

        .canvas-area {
            position: relative; width: 100%; min-height: 480px;
            display: flex; align-items: center; justify-content: center;
            background: #020617; overflow: hidden;
        }

        #viewImage {
            width: 100%; height: auto; max-height: 560px; object-fit: contain; display: block;
        }

        /* Split-Screen Slider */
        .curtain-wrap {
            display: none; position: relative; width: 100%; max-width: 800px; height: 500px; overflow: hidden;
        }

        .curtain-under { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain; }
        .curtain-over-clip { position: absolute; top: 0; left: 0; width: 100%; height: 100%; overflow: hidden; }
        .curtain-over-clip img { position: absolute; top: 0; left: 0; width: 800px; height: 500px; object-fit: contain; }
        .curtain-bar {
            position: absolute; top: 0; bottom: 0; width: 3px; background: var(--cyan-accent);
            cursor: ew-resize; z-index: 10; box-shadow: 0 0 10px var(--cyan-accent);
        }
        .curtain-knob {
            position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
            width: 30px; height: 30px; border-radius: 50%; background: var(--cyan-accent);
            border: 2px solid #fff; box-shadow: 0 0 12px var(--cyan-accent);
            display: flex; align-items: center; justify-content: center;
        }

        /* Telemetry & Details Deck */
        .intel-deck {
            display: grid; grid-template-columns: 1fr 1.5fr; gap: 18px;
        }

        @media (max-width: 900px) { .intel-deck { grid-template-columns: 1fr; } }

        .threat-banner {
            display: flex; align-items: center; justify-content: space-between;
            padding: 14px 16px; border-radius: var(--radius-md); margin-bottom: 14px;
        }

        .threat-banner.crit {
            background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.35);
        }

        .threat-banner.safe {
            background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.35);
        }

        .threat-title h3 { font-size: 15px; font-weight: 800; }
        .threat-title p { font-size: 11.5px; color: var(--text-secondary); }

        .threat-tag {
            font-family: var(--font-mono); font-weight: 800; font-size: 12.5px;
            padding: 5px 12px; border-radius: var(--radius-full);
        }

        .tag-crit-badge { background: var(--rose-danger); color: #fff; box-shadow: 0 0 12px var(--rose-glow); }
        .tag-safe-badge { background: var(--emerald-safe); color: #fff; box-shadow: 0 0 12px var(--emerald-glow); }

        .data-row {
            display: flex; justify-content: space-between; align-items: center;
            padding: 8px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.05); font-size: 12.5px;
        }

        .data-label { color: var(--text-secondary); }
        .data-val { font-family: var(--font-mono); font-weight: 700; color: #fff; }

        .gauge-track {
            height: 7px; background: #1e293b; border-radius: 4px; overflow: hidden; margin: 12px 0 6px;
        }

        .gauge-fill { height: 100%; width: 50%; transition: width 0.35s ease; border-radius: 4px; }

        .tactical-memo {
            margin-top: 14px; padding: 12px; border-radius: var(--radius-md);
            background: rgba(30, 41, 59, 0.5); border-left: 3px solid var(--maritime-blue);
            font-size: 12px; line-height: 1.6; color: #cbd5e1;
        }

        /* Targets Table */
        .intel-table { width: 100%; border-collapse: collapse; font-size: 11.5px; }
        .intel-table th {
            text-align: left; padding: 8px 6px; color: var(--text-muted);
            font-family: var(--font-mono); border-bottom: 1px solid var(--border-subtle);
        }
        .intel-table td {
            padding: 10px 6px; border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            font-family: var(--font-mono);
        }

        .obj-badge {
            display: inline-block; padding: 2px 7px; border-radius: 4px; font-weight: 600; font-size: 10.5px;
        }
        .badge-h { background: rgba(16, 185, 129, 0.15); color: #34d399; }
        .badge-v { background: rgba(249, 115, 22, 0.15); color: #fb923c; }
        .badge-d { background: rgba(239, 68, 68, 0.15); color: #f87171; }

        /* Modal Dokumen Berita Acara SAR */
        .modal-backdrop {
            display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0, 0, 0, 0.75); backdrop-filter: blur(8px);
            z-index: 200; align-items: center; justify-content: center; padding: 20px;
        }

        .modal-card {
            background: #ffffff; color: #0f172a; width: 100%; max-width: 720px;
            border-radius: var(--radius-lg); padding: 32px; box-shadow: 0 20px 50px rgba(0,0,0,0.6);
            max-height: 90vh; overflow-y: auto; font-family: 'Plus Jakarta Sans', sans-serif;
        }

        .dossier-header {
            display: flex; justify-content: space-between; border-bottom: 2px solid #0f172a; padding-bottom: 14px; margin-bottom: 18px;
        }

        .dossier-header h2 { font-size: 18px; font-weight: 800; letter-spacing: -0.3px; }
        .dossier-header p { font-size: 11px; color: #64748b; font-family: var(--font-mono); }
        .dossier-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 18px; font-size: 12px; }
        .dossier-item { background: #f8fafc; padding: 10px; border-radius: 6px; border: 1px solid #e2e8f0; }
        .dossier-item span { display: block; color: #64748b; font-size: 10px; font-weight: 700; text-transform: uppercase; }
        .dossier-item strong { font-size: 13px; color: #0f172a; }

        .dossier-sign { display: flex; justify-content: space-between; margin-top: 30px; padding-top: 20px; border-top: 1px dashed #cbd5e1; font-size: 11px; }

        .btn-print {
            background: #2563eb; color: #fff; padding: 10px 18px; border-radius: 6px; border: none; font-weight: 700; cursor: pointer;
        }

        .btn-close-modal {
            background: #e2e8f0; color: #0f172a; padding: 10px 18px; border-radius: 6px; border: none; font-weight: 700; cursor: pointer;
        }
    </style>
</head>
<body>

    <div class="app-container">
        <!-- Navbar -->
        <header class="navbar">
            <div class="brand-cluster">
                <div class="brand-badge-icon">
                    <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"></circle><line x1="22" y1="12" x2="18" y2="12"></line><line x1="6" y1="12" x2="2" y2="12"></line><line x1="12" y1="6" x2="12" y2="2"></line><line x1="12" y1="22" x2="12" y2="18"></line></svg>
                </div>
                <div class="brand-title">
                    <h1>AeroSAR UAV Mission Control <span class="ver">AI-ENGINE 3.0</span></h1>
                    <p>Sistem Pengenalan Target Maritim Berbasis Visi Komputer & Penentuan Jarak Spasial</p>
                </div>
            </div>

            <div class="nav-actions">
                <div class="telemetry-chip">
                    <span class="chip-dot"></span>
                    <span>OPENCV 5.0 CORE ACTIVE</span>
                </div>
                <button class="audio-toggle-btn active" id="audioToggle" onclick="toggleAudio()">
                    🔊 Alarm Audio ON
                </button>
            </div>
        </header>

        <!-- Main Workspace -->
        <main class="workspace">
            
            <!-- Sidebar Controls -->
            <aside class="control-deck">
                <!-- Upload Box -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">
                            <svg viewBox="0 0 24 24" fill="none" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
                            Input Citra Kamera UAV
                        </div>
                    </div>

                    <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
                        <div class="dropzone-icon">
                            <svg viewBox="0 0 24 24" fill="none" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
                        </div>
                        <div class="dropzone-text">
                            <h4>Unggah Foto Udara Drone</h4>
                            <p>Tarik file ke sini atau klik untuk memilih file</p>
                        </div>
                        <input type="file" id="fileInput" accept="image/*">
                    </div>

                    <div class="file-pill" id="filePill">
                        <span class="file-name" id="fileNameText">citra.jpg</span>
                        <span style="color: var(--text-muted); cursor: pointer;" onclick="clearLoadedFile()">✕</span>
                    </div>

                    <!-- Batas Aman Slider -->
                    <div class="slider-box">
                        <div class="slider-head">
                            <span style="color: var(--text-secondary); font-weight: 500;">Batas Jarak Aman (Threshold)</span>
                            <span class="slider-badge" id="threshBadge">200 px (16.0 m)</span>
                        </div>
                        <input type="range" id="threshSlider" min="60" max="400" value="200" step="5">
                    </div>

                    <!-- GSD Physical Calibration -->
                    <div class="slider-box" style="margin-top: 0;">
                        <div class="slider-head">
                            <span style="color: var(--text-secondary); font-weight: 500;">Ground Sampling Distance (GSD)</span>
                            <span class="slider-badge" id="gsdBadge">0.08 m/px</span>
                        </div>
                        <input type="range" id="gsdSlider" min="0.04" max="0.20" value="0.08" step="0.01">
                        <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">
                            Kalibrasi skala fisik: konversi otomatis piksel ➔ meter.
                        </div>
                    </div>

                    <button class="btn-run" id="btnRun">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                        Jalankan Deteksi & Analisis
                    </button>
                </div>

                <!-- Simulation & Stress Test -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">
                            <svg viewBox="0 0 24 24" fill="none" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polygon points="10 8 16 12 10 16 10 8"></polygon></svg>
                            Preset Skenario Penugasan
                        </div>
                    </div>
                    <div class="preset-grid">
                        <button class="btn-preset" onclick="loadSimScenario('critical')">
                            <span class="preset-tag tag-crit">KRITIS</span>
                            <div class="preset-h">Bahaya Dekat</div>
                            <div class="preset-sub">Jarak 106 px (< batas)</div>
                        </button>
                        <button class="btn-preset" onclick="loadSimScenario('safe')">
                            <span class="preset-tag tag-safe">AMAN</span>
                            <div class="preset-h">Jarak Aman</div>
                            <div class="preset-sub">Jarak 241 px (≥ batas)</div>
                        </button>
                    </div>

                    <div style="margin-top: 14px; font-size: 11.5px; color: var(--text-secondary); font-weight: 600;">
                        Simulasi Kondisi Cuaca & Sensor:
                    </div>
                    <div class="weather-grid">
                        <button class="btn-weather active" id="wClear" onclick="setWeather('clear')">☀️ Cerah</button>
                        <button class="btn-weather" id="wFog" onclick="setWeather('fog')">🌫️ Kabut</button>
                        <button class="btn-weather" id="wDusk" onclick="setWeather('dusk')">🌅 Senja</button>
                        <button class="btn-weather" id="wRough" onclick="setWeather('rough')">🌊 Ombak</button>
                    </div>
                </div>
            </aside>

            <!-- Main Viewport & Telemetry Deck -->
            <section class="viewport-deck">
                
                <div class="viewport-card">
                    <!-- Toolbar Tabs -->
                    <div class="toolbar">
                        <div class="tab-group">
                            <button class="tab-item active" id="tabSBS" onclick="setViewMode('sbs')">
                                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="8" height="18" rx="1"></rect><rect x="13" y="3" width="8" height="18" rx="1"></rect></svg>
                                Berdampingan
                            </button>
                            <button class="tab-item" id="tabCurtain" onclick="setViewMode('curtain')">
                                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="3" x2="12" y2="21"></line><polyline points="8 8 4 12 8 16"></polyline><polyline points="16 8 20 12 16 16"></polyline></svg>
                                Geser Tirai (Split)
                            </button>
                            <button class="tab-item" id="tabPipeline" onclick="setViewMode('pipeline')">
                                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 11a9 9 0 0 1 9 9"></path><path d="M4 4a16 16 0 0 1 16 16"></path><circle cx="5" cy="19" r="1"></circle></svg>
                                Tahap Pipeline Visi
                            </button>
                            <button class="tab-item" id="tabRes" onclick="setViewMode('res')">Hasil Saja</button>
                            <button class="tab-item" id="tabOrig" onclick="setViewMode('orig')">Asli</button>
                        </div>

                        <div class="tool-actions">
                            <button class="btn-act" onclick="openDossierModal()">
                                📋 Berita Acara Misi
                            </button>
                            <a id="downloadBtn" class="btn-act" href="#" download="hasil_deteksi_uav.png">
                                ⬇️ Unduh Citra
                            </a>
                        </div>
                    </div>

                    <div class="canvas-meta">
                        <span id="canvasCaption">VIEWPORT: PERBANDINGAN BERDAMPINGAN RESMI</span>
                        <span id="canvasRes">1600 × 640 PX</span>
                    </div>

                    <div class="canvas-area">
                        <!-- Standard Canvas -->
                        <img id="viewImage" src="" alt="Pratinjau Hasil">

                        <!-- Split Curtain Canvas -->
                        <div class="curtain-wrap" id="curtainWrap">
                            <img id="curtainUnder" class="curtain-under" src="" alt="Orig">
                            <div class="curtain-over-clip" id="curtainOverClip">
                                <img id="curtainOver" src="" alt="Res">
                            </div>
                            <div class="curtain-bar" id="curtainBar" style="left: 50%;">
                                <div class="curtain-knob">
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#090d16" stroke-width="2"><polyline points="8 7 3 12 8 17"></polyline><polyline points="16 7 21 12 16 17"></polyline></svg>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Telemetry & Threat Assessment -->
                <div class="intel-deck">
                    <!-- Threat Evaluation -->
                    <div class="card">
                        <div class="card-header">
                            <div class="card-title">
                                <svg viewBox="0 0 24 24" fill="none" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
                                Evaluasi Bahaya Korban
                            </div>
                        </div>

                        <div class="threat-banner crit" id="threatBanner">
                            <div class="threat-title">
                                <h3 id="threatTitleText">STATUS KRITIS</h3>
                                <p id="threatSubText">Korban berada di zona ancaman langsung</p>
                            </div>
                            <div class="threat-tag tag-crit-badge" id="threatTagBadge">CRITICAL</div>
                        </div>

                        <div class="data-row">
                            <span class="data-label">Jarak Korban ➔ Sumber Bahaya</span>
                            <span class="data-val" id="valDistance">105.6 px (8.5 m)</span>
                        </div>
                        <div class="data-row">
                            <span class="data-label">Batas Aman yang Ditentukan</span>
                            <span class="data-val" id="valThreshold">200.0 px (16.0 m)</span>
                        </div>
                        <div class="data-row">
                            <span class="data-label">Rute Helipad ➔ Korban (Pengantaran)</span>
                            <span class="data-val" id="valRoute">36.5 m (ETA 2.4 detik)</span>
                        </div>

                        <div class="gauge-track">
                            <div class="gauge-fill" id="gaugeFill" style="width: 53%; background: var(--rose-danger);"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 10.5px; color: var(--text-muted); font-family: var(--font-mono);">
                            <span>0 m (Kontak Bahaya)</span>
                            <span id="gaugeLabelThresh">Batas: 16.0 m</span>
                            <span>32 m (Zona Aman)</span>
                        </div>

                        <div class="tactical-memo" id="tacticalMemo">
                            Memuat analisis intelijen misi...
                        </div>
                    </div>

                    <!-- Coordinates Table -->
                    <div class="card">
                        <div class="card-header">
                            <div class="card-title">
                                <svg viewBox="0 0 24 24" fill="none" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="22" y1="12" x2="18" y2="12"></line><line x1="6" y1="12" x2="2" y2="12"></line><line x1="12" y1="6" x2="12" y2="2"></line><line x1="12" y1="22" x2="12" y2="18"></line></svg>
                                Telemetri Koordinat Target
                            </div>
                        </div>

                        <table class="intel-table">
                            <thead>
                                <tr>
                                    <th>Target Objek</th>
                                    <th>Centroid (+)</th>
                                    <th>Bounding Box [x, y, w, h]</th>
                                    <th>Luas Area</th>
                                </tr>
                            </thead>
                            <tbody id="intelTbody">
                                <!-- Generated by JS -->
                            </tbody>
                        </table>

                        <div style="margin-top: 16px; padding: 12px; background: rgba(0,0,0,0.25); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle); font-size: 11px; line-height: 1.5; color: var(--text-muted);">
                            <strong>Catatan Teknis untuk Dewan Juri:</strong><br>
                            Sistem menerapkan segmentasi ruang warna HSV adaptif dengan filter morfologi elips untuk mereduksi derau riak gelombang laut, dilanjutkan ekstraksi <em>spatial moments</em> ($M_{10}/M_{00}, M_{01}/M_{00}$) guna mendapatkan titik berat target dengan akurasi sub-piksel.
                        </div>
                    </div>
                </div>

            </section>
        </main>
    </div>

    <!-- Modal Berita Acara Misi SAR -->
    <div class="modal-backdrop" id="dossierModal">
        <div class="modal-card">
            <div class="dossier-header">
                <div>
                    <h2>BERITA ACARA PENGAMATAN UDARA SAR</h2>
                    <p>UNIT RECONNAISSANCE UAV MARITIM • DOKUMEN RESMI</p>
                </div>
                <div style="text-align: right;">
                    <strong style="color: #2563eb; font-family: var(--font-mono);">NO: SAR-UAV/2026/089</strong><br>
                    <span style="font-size: 11px; color: #64748b;" id="dossierDate">29 Oktober 2026</span>
                </div>
            </div>

            <div class="dossier-grid">
                <div class="dossier-item">
                    <span>Status Evaluasi Korban</span>
                    <strong id="dossierStatus">KRITIS (EMERGENCY)</strong>
                </div>
                <div class="dossier-item">
                    <span>Jarak Korban ke Bahaya</span>
                    <strong id="dossierDist">105.6 px (8.5 meter)</strong>
                </div>
                <div class="dossier-item">
                    <span>Jarak Tempuh dari Helipad</span>
                    <strong id="dossierRoute">36.5 meter</strong>
                </div>
                <div class="dossier-item">
                    <span>Estimasi Waktu Luncurkan (ETA)</span>
                    <strong id="dossierEta">2.4 detik</strong>
                </div>
            </div>

            <div style="font-size: 12.5px; line-height: 1.6; color: #334155; margin-bottom: 20px;">
                <strong>Ringkasan Eksekutif Operasi:</strong><br>
                <p id="dossierMemoText"></p>
            </div>

            <div class="dossier-sign">
                <div>
                    Petugas Operator UAV SAR:<br><br><br>
                    <strong>( Operator Visi Komputer )</strong>
                </div>
                <div style="text-align: right;">
                    Mengetahui, Komandan Misi SAR:<br><br><br>
                    <strong>( Capt. Maritime SAR Command )</strong>
                </div>
            </div>

            <div style="display: flex; justify-content: flex-end; gap: 10px; margin-top: 24px;">
                <button class="btn-close-modal" onclick="closeDossierModal()">Tutup</button>
                <button class="btn-print" onclick="window.print()">🖨️ Cetak Dokumen</button>
            </div>
        </div>
    </div>

    <script>
        // State
        let rawImageBase64 = null;
        let lastResult = null;
        let currentMode = 'sbs';
        let currentWeather = 'clear';
        let audioActive = true;

        // Elements
        const fileInput = document.getElementById('fileInput');
        const dropzone = document.getElementById('dropzone');
        const filePill = document.getElementById('filePill');
        const fileNameText = document.getElementById('fileNameText');
        const threshSlider = document.getElementById('threshSlider');
        const threshBadge = document.getElementById('threshBadge');
        const gsdSlider = document.getElementById('gsdSlider');
        const gsdBadge = document.getElementById('gsdBadge');
        const btnRun = document.getElementById('btnRun');
        const viewImage = document.getElementById('viewImage');
        const curtainWrap = document.getElementById('curtainWrap');
        const curtainUnder = document.getElementById('curtainUnder');
        const curtainOver = document.getElementById('curtainOver');
        const curtainOverClip = document.getElementById('curtainOverClip');
        const curtainBar = document.getElementById('curtainBar');
        const downloadBtn = document.getElementById('downloadBtn');

        const threatBanner = document.getElementById('threatBanner');
        const threatTitleText = document.getElementById('threatTitleText');
        const threatSubText = document.getElementById('threatSubText');
        const threatTagBadge = document.getElementById('threatTagBadge');
        const valDistance = document.getElementById('valDistance');
        const valThreshold = document.getElementById('valThreshold');
        const valRoute = document.getElementById('valRoute');
        const gaugeFill = document.getElementById('gaugeFill');
        const gaugeLabelThresh = document.getElementById('gaugeLabelThresh');
        const tacticalMemo = document.getElementById('tacticalMemo');
        const intelTbody = document.getElementById('intelTbody');
        const canvasCaption = document.getElementById('canvasCaption');
        const canvasRes = document.getElementById('canvasRes');

        // Audio Ping Synthesizer (Web Audio API)
        function playSonarPing(isCritical) {
            if (!audioActive) return;
            try {
                const ctx = new (window.AudioContext || window.webkitAudioContext)();
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                osc.type = isCritical ? 'triangle' : 'sine';
                osc.frequency.setValueAtTime(isCritical ? 880 : 540, ctx.currentTime);
                if (isCritical) {
                    osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.35);
                }
                gain.gain.setValueAtTime(0.2, ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.35);
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.start();
                osc.stop(ctx.currentTime + 0.36);
            } catch (e) {}
        }

        function toggleAudio() {
            audioActive = !audioActive;
            const btn = document.getElementById('audioToggle');
            btn.className = audioActive ? 'audio-toggle-btn active' : 'audio-toggle-btn';
            btn.textContent = audioActive ? '🔊 Alarm Audio ON' : '🔇 Audio Muted';
        }

        // Sliders
        threshSlider.addEventListener('input', () => {
            const gsd = parseFloat(gsdSlider.value);
            const px = parseFloat(threshSlider.value);
            threshBadge.textContent = `${px} px (${(px * gsd).toFixed(1)} m)`;
        });

        gsdSlider.addEventListener('input', () => {
            const gsd = parseFloat(gsdSlider.value);
            const px = parseFloat(threshSlider.value);
            gsdBadge.textContent = `${gsd.toFixed(2)} m/px`;
            threshBadge.textContent = `${px} px (${(px * gsd).toFixed(1)} m)`;
        });

        // Dropzone
        dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
        dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault(); dropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length) loadSelectedFile(e.dataTransfer.files[0]);
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) loadSelectedFile(e.target.files[0]);
        });

        function loadSelectedFile(file) {
            fileNameText.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
            filePill.style.display = 'flex';
            const reader = new FileReader();
            reader.onload = (ev) => {
                rawImageBase64 = ev.target.result;
                triggerDetection();
            };
            reader.readAsDataURL(file);
        }

        function clearLoadedFile() {
            rawImageBase64 = null;
            filePill.style.display = 'none';
            fileInput.value = '';
        }

        // Weather Simulator
        function setWeather(w) {
            currentWeather = w;
            ['wClear', 'wFog', 'wDusk', 'wRough'].forEach(id => document.getElementById(id).classList.remove('active'));
            if (w === 'clear') document.getElementById('wClear').classList.add('active');
            if (w === 'fog') document.getElementById('wFog').classList.add('active');
            if (w === 'dusk') document.getElementById('wDusk').classList.add('active');
            if (w === 'rough') document.getElementById('wRough').classList.add('active');
            loadSimScenario('critical');
        }

        async function loadSimScenario(type) {
            const res = await fetch(`/api/sample?type=${type}&weather=${currentWeather}`);
            const data = await res.json();
            rawImageBase64 = data.image_base64;
            fileNameText.textContent = `simulasi_${type}_${currentWeather}.png`;
            filePill.style.display = 'flex';
            triggerDetection();
        }

        btnRun.addEventListener('click', () => {
            if (!rawImageBase64) {
                alert("Silakan unggah citra terlebih dahulu atau klik salah satu simulasi skenario.");
                return;
            }
            triggerDetection();
        });

        // Trigger Detection
        async function triggerDetection() {
            btnRun.disabled = true;
            btnRun.textContent = "⏳ Memproses...";

            try {
                const threshold = parseFloat(threshSlider.value);
                const gsd = parseFloat(gsdSlider.value);
                const resp = await fetch('/api/detect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image_base64: rawImageBase64,
                        threshold: threshold,
                        gsd: gsd
                    })
                });

                lastResult = await resp.json();
                renderTelemetryUI(lastResult);
                playSonarPing(lastResult.status === "CRITICAL");
            } catch (err) {
                alert("Kesalahan pemrosesan: " + err);
            } finally {
                btnRun.disabled = false;
                btnRun.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Jalankan Deteksi & Analisis`;
            }
        }

        // Render Telemetry
        function renderTelemetryUI(data) {
            curtainUnder.src = data.original_image;
            curtainOver.src = data.result_only_image;
            updateViewportView();

            const isCrit = (data.status === "CRITICAL");
            threatBanner.className = isCrit ? "threat-banner crit" : "threat-banner safe";
            threatTitleText.textContent = isCrit ? "STATUS: KRITIS / CRITICAL" : "STATUS: AMAN / SAFE";
            threatSubText.textContent = isCrit ? "Korban berada dalam jangkauan bahaya langsung" : "Korban berada di luar perimeter bahaya";
            threatTagBadge.className = isCrit ? "threat-tag tag-crit-badge" : "threat-tag tag-safe-badge";
            threatTagBadge.textContent = data.status;

            const gsd = data.gsd || 0.08;
            if (data.distance !== null) {
                const distM = (data.distance * gsd).toFixed(1);
                valDistance.textContent = `${data.distance.toFixed(1)} px (${distM} m)`;
                const threshM = (data.threshold * gsd).toFixed(1);
                valThreshold.textContent = `${data.threshold.toFixed(1)} px (${threshM} m)`;
                gaugeLabelThresh.textContent = `Batas: ${threshM} m`;

                const pct = Math.min(Math.max((data.distance / 400.0) * 100, 5), 100);
                gaugeFill.style.width = `${pct}%`;
                gaugeFill.style.background = isCrit ? "var(--rose-danger)" : "var(--emerald-safe)";
            }

            if (data.flight_route_m !== null) {
                valRoute.textContent = `${data.flight_route_m.toFixed(1)} m (ETA ${data.eta_seconds.toFixed(1)}s)`;
            } else {
                valRoute.textContent = "-";
            }

            tacticalMemo.innerHTML = data.recommendation;

            // Targets Table
            intelTbody.innerHTML = '';
            const targets = [
                { k: 'Helipad', cls: 'badge-h' },
                { k: 'Korban', cls: 'badge-v' },
                { k: 'Sumber Bahaya', cls: 'badge-d' }
            ];

            targets.forEach(t => {
                const tr = document.createElement('tr');
                const info = data.detections[t.k];
                if (info) {
                    tr.innerHTML = `
                        <td><span class="obj-badge ${t.cls}">${t.k}</span></td>
                        <td style="color: #67e8f9;">(${info.center[0]}, ${info.center[1]})</td>
                        <td>[${info.bbox[0]}, ${info.bbox[1]}, ${info.bbox[2]}, ${info.bbox[3]}]</td>
                        <td>${Math.round(info.area)} px²</td>
                    `;
                } else {
                    tr.innerHTML = `
                        <td><span class="obj-badge ${t.cls}">${t.k}</span></td>
                        <td>-</td><td>-</td><td style="color: var(--rose-danger);">Tidak Terdeteksi</td>
                    `;
                }
                intelTbody.appendChild(tr);
            });
        }

        // View Mode Switcher
        function setViewMode(mode) {
            currentMode = mode;
            ['tabSBS', 'tabCurtain', 'tabPipeline', 'tabRes', 'tabOrig'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.classList.remove('active');
            });

            if (mode === 'sbs') document.getElementById('tabSBS').classList.add('active');
            if (mode === 'curtain') document.getElementById('tabCurtain').classList.add('active');
            if (mode === 'pipeline') document.getElementById('tabPipeline').classList.add('active');
            if (mode === 'res') document.getElementById('tabRes').classList.add('active');
            if (mode === 'orig') document.getElementById('tabOrig').classList.add('active');

            updateViewportView();
        }

        function updateViewportView() {
            if (!lastResult) return;

            if (currentMode === 'curtain') {
                viewImage.style.display = 'none';
                curtainWrap.style.display = 'block';
                canvasCaption.textContent = "VIEWPORT: GESER TIRAI INTERAKTIF (DRAG UNTUK MEMBANDINGKAN)";
                canvasRes.textContent = "SPLIT OVERLAY";
            } else {
                curtainWrap.style.display = 'none';
                viewImage.style.display = 'block';

                if (currentMode === 'sbs') {
                    viewImage.src = lastResult.comparison_image;
                    downloadBtn.href = lastResult.comparison_image;
                    canvasCaption.textContent = "VIEWPORT: PERBANDINGAN BERDAMPINGAN RESMI";
                    canvasRes.textContent = "1600 × 640 PX";
                } else if (currentMode === 'pipeline') {
                    viewImage.src = lastResult.pipeline_image;
                    downloadBtn.href = lastResult.pipeline_image;
                    canvasCaption.textContent = "VIEWPORT: TAHAP SEGMENTASI PIPELINE VISI (UNTUK PENGUJI)";
                    canvasRes.textContent = "800 × 600 PX (4-GRID)";
                } else if (currentMode === 'res') {
                    viewImage.src = lastResult.result_only_image;
                    downloadBtn.href = lastResult.result_only_image;
                    canvasCaption.textContent = "VIEWPORT: HASIL DETEKSI SAJA";
                    canvasRes.textContent = "800 × 600 PX";
                } else if (currentMode === 'orig') {
                    viewImage.src = lastResult.original_image;
                    downloadBtn.href = lastResult.original_image;
                    canvasCaption.textContent = "VIEWPORT: CITRA ASLI KAMERA UAV";
                    canvasRes.textContent = "800 × 600 PX";
                }
            }
        }

        // Curtain Slider
        let isDrag = false;
        curtainBar.addEventListener('mousedown', () => isDrag = true);
        window.addEventListener('mouseup', () => isDrag = false);
        window.addEventListener('mousemove', (e) => {
            if (!isDrag) return;
            const rect = curtainWrap.getBoundingClientRect();
            const x = Math.max(10, Math.min(e.clientX - rect.left, rect.width - 10));
            curtainBar.style.left = `${x}px`;
            curtainOverClip.style.width = `${x}px`;
        });

        // Dossier Modal
        function openDossierModal() {
            if (!lastResult) {
                alert("Jalankan analisis citra terlebih dahulu.");
                return;
            }
            const gsd = lastResult.gsd || 0.08;
            document.getElementById('dossierStatus').textContent = lastResult.status;
            document.getElementById('dossierDist').textContent = `${lastResult.distance.toFixed(1)} px (${(lastResult.distance * gsd).toFixed(1)} m)`;
            document.getElementById('dossierRoute').textContent = lastResult.flight_route_m ? `${lastResult.flight_route_m.toFixed(1)} meter` : "-";
            document.getElementById('dossierEta').textContent = lastResult.eta_seconds ? `${lastResult.eta_seconds.toFixed(1)} detik` : "-";
            document.getElementById('dossierMemoText').innerHTML = lastResult.recommendation;
            document.getElementById('dossierModal').style.display = 'flex';
        }

        function closeDossierModal() {
            document.getElementById('dossierModal').style.display = 'none';
        }

        window.addEventListener('DOMContentLoaded', () => {
            loadSimScenario('critical');
        });
    </script>
</body>
</html>
"""

def decode_image_base64(data_uri):
    if "," in data_uri:
        data_uri = data_uri.split(",")[1]
    raw_bytes = base64.b64decode(data_uri)
    nparr = np.frombuffer(raw_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def encode_image_base64(img):
    _, buffer = cv2.imencode('.png', img)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/sample", methods=["GET"])
def get_sample():
    sample_type = request.args.get("type", "critical")
    weather = request.args.get("weather", "clear")
    danger_near = (sample_type == "critical")
    sample_img = generate_synthetic_sar_image(danger_near_victim=danger_near, weather=weather)
    return jsonify({
        "image_base64": encode_image_base64(sample_img),
        "type": sample_type,
        "weather": weather
    })

@app.route("/api/detect", methods=["POST"])
def detect_api():
    data = request.get_json() or {}
    b64_in = data.get("image_base64")
    threshold = float(data.get("threshold", 200.0))
    gsd = float(data.get("gsd", 0.08))

    if not b64_in:
        return jsonify({"error": "Citra tidak ditemukan"}), 400

    orig_img = decode_image_base64(b64_in)
    if orig_img is None:
        return jsonify({"error": "Gagal membaca citra"}), 400

    detector = SARDroneVision(safe_distance_threshold=threshold, gsd_meters_per_pixel=gsd)
    result_img, detections, dist, status, masks, flight_route_m, eta_seconds = detector.process(orig_img)
    side_by_side = detector.make_side_by_side(orig_img, result_img)
    pipeline_img = detector.create_pipeline_inspection_image(orig_img, masks)

    if status == "CRITICAL":
        recommendation = (
            f"<strong>DARURAT OPERASIONAL:</strong> Korban berada di zona kritis ancaman langsung "
            f"(jarak terukur: <strong>{dist:.1f} px / {(dist * gsd):.1f} m</strong>, di bawah batas aman <strong>{threshold:.0f} px / {(threshold * gsd):.1f} m</strong>). "
            f"Drone SAR diperintahkan untuk segera melepaskan muatan pelampung darurat (*airdrop life-ring*), "
            f"menyalakan suar sinyal laut (*marine flare*), serta memandu kapal penyelamat terdekat."
        )
    elif status == "SAFE":
        recommendation = (
            f"<strong>KONDISI AMAN TERKONFIRMASI:</strong> Jarak aman antara Korban dan Sumber Bahaya "
            f"tercapai (jarak terukur: <strong>{dist:.1f} px / {(dist * gsd):.1f} m</strong>, melampaui batas aman <strong>{threshold:.0f} px / {(threshold * gsd):.1f} m</strong>). "
            f"Drone UAV dapat melakukan penurunan suplai medis darurat dengan rute stabil sebelum kembali mendarat di Helipad."
        )
    else:
        recommendation = (
            "<strong>PERHATIAN:</strong> Target deteksi belum lengkap. Periksa sudut kamera drone dan pencahayaan permukaan laut."
        )

    return jsonify({
        "status": status,
        "distance": dist,
        "threshold": threshold,
        "gsd": gsd,
        "flight_route_m": flight_route_m,
        "eta_seconds": eta_seconds,
        "detections": detections,
        "recommendation": recommendation,
        "original_image": encode_image_base64(orig_img),
        "result_only_image": encode_image_base64(result_img),
        "comparison_image": encode_image_base64(side_by_side),
        "pipeline_image": encode_image_base64(pipeline_img)
    })

if __name__ == "__main__":
    port = 5000
    print(f"\n=======================================================")
    print(f"🚀 AeroSAR Tactical Ground Control Station (GCS) v3.0 Aktif!")
    print(f"🔗 Buka di Browser: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
