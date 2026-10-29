#!/usr/bin/env python3
"""
================================================================================
AeroSAR Drone Vision System - Advanced Engineering Edition
Deteksi Helipad, Korban, dan Sumber Bahaya Menggunakan OpenCV
================================================================================
"""

import cv2
import numpy as np
import math
import argparse
import os
import sys

# ==============================================================================
# 1. GENERATOR CITRA SIMULASI AERIAL MARITIM MULTI-KONDISI
# ==============================================================================
def generate_synthetic_sar_image(width=800, height=600, danger_near_victim=True, weather="clear"):
    """
    Membuat citra simulasi kamera udara drone SAR maritim:
    - weather: 'clear' (terang), 'fog' (kabut laut), 'dusk' (senja/redup), 'rough' (ombak berombak)
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # 1. Background Laut (gradasi kedalaman laut)
    for y in range(height):
        if weather == "dusk":
            b = int(70 + 25 * (y / height))
            g = int(45 + 15 * (y / height))
            r = int(25 + 10 * (y / height))
        else:
            b = int(140 + 40 * (y / height))
            g = int(80 + 20 * (y / height))
            r = int(20 + 10 * (y / height))
        image[y, :] = [b, g, r]

    # Tekstur riak air laut
    np.random.seed(42)
    noise_std = 18 if weather == "rough" else 7
    noise = np.random.normal(0, noise_std, (height, width, 3)).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Ombak buih jika cuaca rough
    if weather == "rough":
        for _ in range(12):
            ox = np.random.randint(150, width)
            oy = np.random.randint(50, height)
            cv2.ellipse(image, (ox, oy), (35, 6), 15, 0, 360, (190, 180, 160), -1)

    # 2. Dermaga Pelabuhan (Dock / Pier)
    dock_color = (130, 130, 130) if weather != "dusk" else (90, 90, 90)
    cv2.rectangle(image, (0, 0), (220, 240), dock_color, -1)
    cv2.rectangle(image, (0, 0), (220, 240), (70, 70, 70), 3)

    # --- Objek 1: Helipad di dermaga ---
    helipad_center = (110, 120)
    h_col = (30, 200, 30) if weather != "dusk" else (20, 150, 20)
    cv2.circle(image, helipad_center, 45, h_col, -1)       # Hijau BGR
    cv2.circle(image, helipad_center, 45, (255, 255, 255), 3)
    cv2.putText(image, "H", (helipad_center[0] - 18, helipad_center[1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4, cv2.LINE_AA)

    # --- Objek 2: Korban di perairan ---
    victim_center = (520, 320)
    v_col = (0, 120, 255) if weather != "dusk" else (0, 95, 205)
    cv2.circle(image, victim_center, 26, v_col, -1)           # Oranye BGR
    cv2.circle(image, victim_center, 12, (255, 255, 255), -1) # Tengah
    cv2.circle(image, victim_center, 7, v_col, -1)
    cv2.circle(image, victim_center, 26, (0, 80, 200), 2)

    # --- Objek 3: Sumber Bahaya ---
    if danger_near_victim:
        hazard_center = (600, 390)  # ~106 px (< batas aman -> CRITICAL)
    else:
        hazard_center = (680, 140)  # ~241 px (≥ batas aman -> SAFE)

    haz_col = (20, 20, 240) if weather != "dusk" else (15, 15, 185)
    cv2.circle(image, hazard_center, 32, haz_col, -1)  # Merah BGR
    cv2.circle(image, hazard_center, 32, (0, 0, 180), 3)
    cv2.putText(image, "!", (hazard_center[0] - 7, hazard_center[1] + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

    # Efek Kabut Maritim (Sea Fog)
    if weather == "fog":
        fog_layer = np.full((height, width, 3), 190, dtype=np.uint8)
        image = cv2.addWeighted(image, 0.65, fog_layer, 0.35, 0)

    return image


# ==============================================================================
# 2. ADVANCED SAR DRONE VISION PIPELINE ENGINE
# ==============================================================================
class SARDroneVision:
    def __init__(self, safe_distance_threshold=200.0, gsd_meters_per_pixel=0.08, drone_speed_mps=15.0):
        """
        :param safe_distance_threshold: Batas aman minimum (piksel)
        :param gsd_meters_per_pixel: Ground Sampling Distance (meter per piksel) untuk konversi fisik
        :param drone_speed_mps: Kecepatan jelajah UAV dalam meter per detik
        """
        self.safe_threshold = float(safe_distance_threshold)
        self.gsd = float(gsd_meters_per_pixel)
        self.drone_speed = float(drone_speed_mps)

        # Ambang batas HSV Adaptif
        self.hsv_helipad_low = np.array([30, 60, 60])
        self.hsv_helipad_high = np.array([90, 255, 255])
        self.color_helipad = (0, 200, 0) # Hijau

        self.hsv_victim_low = np.array([8, 80, 80])
        self.hsv_victim_high = np.array([28, 255, 255])
        self.color_victim = (0, 140, 255) # Oranye

        self.hsv_hazard_low1 = np.array([0, 90, 80])
        self.hsv_hazard_high1 = np.array([10, 255, 255])
        self.hsv_hazard_low2 = np.array([168, 90, 80])
        self.hsv_hazard_high2 = np.array([180, 255, 255])
        self.color_hazard = (0, 0, 230) # Merah

    def _get_hazard_mask(self, hsv):
        m1 = cv2.inRange(hsv, self.hsv_hazard_low1, self.hsv_hazard_high1)
        m2 = cv2.inRange(hsv, self.hsv_hazard_low2, self.hsv_hazard_high2)
        return cv2.bitwise_or(m1, m2)

    def _clean_and_find_contour(self, mask, min_area=70):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_DILATE, kernel, iterations=1)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) >= min_area]
        return max(valid, key=cv2.contourArea) if valid else None, cleaned

    def _extract_object(self, contour, label, color):
        x, y, w, h = cv2.boundingRect(contour)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2

        return {
            "label": label,
            "bbox": (int(x), int(y), int(w), int(h)),
            "center": (int(cx), int(cy)),
            "color": color,
            "area": float(cv2.contourArea(contour))
        }

    def detect_all_objects(self, bgr_image):
        """Mendeteksi seluruh target dan mengembalikan info objek beserta mask biner"""
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

        results = {}
        masks = {}

        # 1. Helipad
        mask_h_raw = cv2.inRange(blurred, self.hsv_helipad_low, self.hsv_helipad_high)
        cnt_h, mask_h_clean = self._clean_and_find_contour(mask_h_raw, min_area=180)
        masks["Helipad"] = mask_h_clean
        if cnt_h is not None:
            results["Helipad"] = self._extract_object(cnt_h, "Helipad", self.color_helipad)

        # 2. Korban
        mask_v_raw = cv2.inRange(blurred, self.hsv_victim_low, self.hsv_victim_high)
        cnt_v, mask_v_clean = self._clean_and_find_contour(mask_v_raw, min_area=60)
        masks["Korban"] = mask_v_clean
        if cnt_v is not None:
            results["Korban"] = self._extract_object(cnt_v, "Korban", self.color_victim)

        # 3. Sumber Bahaya
        mask_d_raw = self._get_hazard_mask(blurred)
        cnt_d, mask_d_clean = self._clean_and_find_contour(mask_d_raw, min_area=80)
        masks["Sumber Bahaya"] = mask_d_clean
        if cnt_d is not None:
            results["Sumber Bahaya"] = self._extract_object(cnt_d, "Sumber Bahaya", self.color_hazard)

        return results, masks

    def draw_cross_marker(self, img, center, color, size=14, thickness=2):
        cx, cy = center
        cv2.line(img, (cx - size, cy), (cx + size, cy), color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx, cy - size), (cx, cy + size), color, thickness, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)

    def draw_tag_label(self, img, text, x, y, color):
        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        pad = 4

        bx1, by1 = max(0, x), max(0, y - th - pad * 2)
        bx2, by2 = min(img.shape[1], x + tw + pad * 2), y

        cv2.rectangle(img, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 1)
        cv2.putText(img, text, (bx1 + pad, by2 - pad), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def process(self, original_img):
        """Pipeline pemrosesan komprehensif"""
        result_img = original_img.copy()
        detections, masks = self.detect_all_objects(original_img)

        # 1. Overlay Danger Perimeter Buffer Ring di sekeliling Sumber Bahaya
        if "Sumber Bahaya" in detections:
            cb = detections["Sumber Bahaya"]["center"]
            overlay = result_img.copy()
            # Gambar zona lingkaran transparan
            radius_int = int(self.safe_threshold)
            zone_color = (0, 0, 240) # Merah bahaya
            cv2.circle(overlay, cb, radius_int, zone_color, -1)
            cv2.addWeighted(overlay, 0.18, result_img, 0.82, 0, result_img)
            # Garis tepi lingkaran perimeter
            cv2.circle(result_img, cb, radius_int, (0, 0, 255), 2, cv2.LINE_AA)

        # 2. Gambar Bounding Box, Label, dan Crosshair
        for name, data in detections.items():
            x, y, w, h = data["bbox"]
            cx, cy = data["center"]
            color = data["color"]

            cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
            self.draw_tag_label(result_img, name, x, y, color)
            self.draw_cross_marker(result_img, (cx, cy), color, size=14, thickness=2)

        # 3. Hitung Jarak Korban <-> Bahaya
        distance_px = None
        distance_m = None
        status = "UNKNOWN"

        if "Korban" in detections and "Sumber Bahaya" in detections:
            c_k = detections["Korban"]["center"]
            c_b = detections["Sumber Bahaya"]["center"]

            dx = c_k[0] - c_b[0]
            dy = c_k[1] - c_b[1]
            distance_px = math.sqrt(dx * dx + dy * dy)
            distance_m = distance_px * self.gsd

            if distance_px < self.safe_threshold:
                status = "CRITICAL"
                line_col = (0, 0, 255) # Merah
            else:
                status = "SAFE"
                line_col = (0, 240, 240) # Cyan

            # Garis penghubung
            cv2.line(result_img, c_k, c_b, line_col, 2, cv2.LINE_AA)

            # Label jarak (piksel & meter)
            mid_x = (c_k[0] + c_b[0]) // 2
            mid_y = (c_k[1] + c_b[1]) // 2
            dist_label = f"Jarak: {distance_px:.1f} px ({distance_m:.1f} m)"
            self.draw_tag_label(result_img, dist_label, mid_x - 55, mid_y - 8, line_col)

        # 4. Rute Penerbangan Helipad -> Korban (SAR Delivery Route)
        flight_route_m = None
        eta_seconds = None
        if "Helipad" in detections and "Korban" in detections:
            ch = detections["Helipad"]["center"]
            ck = detections["Korban"]["center"]
            flight_px = math.sqrt((ch[0] - ck[0])**2 + (ch[1] - ck[1])**2)
            flight_route_m = flight_px * self.gsd
            eta_seconds = flight_route_m / self.drone_speed if self.drone_speed > 0 else 0

            # Garis rute penerbangan putus-putus / hijau neon
            cv2.line(result_img, ch, ck, (0, 255, 120), 1, cv2.LINE_AA)
            route_mid = ((ch[0] + ck[0]) // 2, (ch[1] + ck[1]) // 2)
            self.draw_tag_label(result_img, f"Rute SAR: {flight_route_m:.1f} m (ETA {eta_seconds:.1f}s)",
                                route_mid[0] - 65, route_mid[1] + 18, (0, 255, 120))

        # 5. Header Telemetri & Badge
        h, w = result_img.shape[:2]
        cv2.rectangle(result_img, (0, 0), (w, 42), (18, 18, 18), -1)
        cv2.line(result_img, (0, 42), (w, 42), (50, 50, 50), 2)
        cv2.putText(result_img, "AeroSAR UAV TACTICAL MISSION SYSTEM", (15, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)

        badge_bg = (0, 0, 220) if status == "CRITICAL" else (0, 180, 0)
        badge_txt = f"STATUS: CRITICAL (< {self.safe_threshold:.0f} px)" if status == "CRITICAL" else f"STATUS: SAFE ({distance_px:.1f} px)"
        (bw, _), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        cv2.rectangle(result_img, (w - bw - 24, 6), (w - 10, 36), badge_bg, -1)
        cv2.putText(result_img, badge_txt, (w - bw - 16, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

        return result_img, detections, distance_px, status, masks, flight_route_m, eta_seconds

    def make_side_by_side(self, original_img, result_img):
        h, w = original_img.shape[:2]
        header_h = 38

        def build_panel(img, title, title_color):
            panel = np.zeros((h + header_h, w, 3), dtype=np.uint8)
            cv2.rectangle(panel, (0, 0), (w, header_h), (25, 25, 25), -1)
            cv2.line(panel, (0, header_h), (w, header_h), (50, 50, 50), 2)
            (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.putText(panel, title, ((w - tw) // 2, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, title_color, 2, cv2.LINE_AA)
            panel[header_h:, :] = img
            return panel

        panel_orig = build_panel(original_img, "ORIGINAL IMAGE", (220, 220, 220))
        panel_res = build_panel(result_img, "RESULT (DETECTION & DISTANCE ANALYSIS)", (0, 255, 200))
        return np.hstack([panel_orig, panel_res])

    def create_pipeline_inspection_image(self, orig_img, masks):
        """Membuat visualisasi 4-Grid tahap segmentasi citra untuk dewan juri/penilai"""
        h, w = orig_img.shape[:2]
        grid_w, grid_h = w // 2, h // 2

        # 1. Asli resize
        orig_small = cv2.resize(orig_img, (grid_w, grid_h))
        cv2.putText(orig_small, "1. Input Asli (Aerial BGR)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # 2. Mask Helipad
        mask_h_rgb = cv2.cvtColor(cv2.resize(masks.get("Helipad", np.zeros((h, w), dtype=np.uint8)), (grid_w, grid_h)), cv2.COLOR_GRAY2BGR)
        mask_h_rgb[:, :, 0] = 0 # Biru 0
        cv2.putText(mask_h_rgb, "2. Mask Helipad (Hijau HSV)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # 3. Mask Korban
        mask_v_rgb = cv2.cvtColor(cv2.resize(masks.get("Korban", np.zeros((h, w), dtype=np.uint8)), (grid_w, grid_h)), cv2.COLOR_GRAY2BGR)
        mask_v_rgb[:, :, 0] = 0 # Oranye tint
        mask_v_rgb[:, :, 1] = mask_v_rgb[:, :, 1] // 2
        cv2.putText(mask_v_rgb, "3. Mask Korban (Oranye Buoy)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

        # 4. Mask Bahaya
        mask_d_rgb = cv2.cvtColor(cv2.resize(masks.get("Sumber Bahaya", np.zeros((h, w), dtype=np.uint8)), (grid_w, grid_h)), cv2.COLOR_GRAY2BGR)
        mask_d_rgb[:, :, 0] = 0
        mask_d_rgb[:, :, 1] = 0 # Merah tint
        cv2.putText(mask_d_rgb, "4. Mask Bahaya (Merah Dual-Range)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        row1 = np.hstack([orig_small, mask_h_rgb])
        row2 = np.hstack([mask_v_rgb, mask_d_rgb])
        pipeline_grid = np.vstack([row1, row2])
        return pipeline_grid
