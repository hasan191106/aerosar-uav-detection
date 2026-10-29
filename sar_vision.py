import cv2
import numpy as np
import math

class SARDroneVision:
    def __init__(self, safe_distance_threshold=200):
        """
        Inisialisasi sistem deteksi drone SAR:
        - safe_distance_threshold: batas aman minimum (dalam piksel) antara Korban dan Sumber Bahaya
        """
        self.safe_threshold = safe_distance_threshold
        
        # Konfigurasi rentang warna HSV dan warna display visual (BGR)
        # 1. Helipad (Pangkalan di Dermaga): Hijau
        self.helipad_hsv_lower = np.array([35, 70, 70])
        self.helipad_hsv_upper = np.array([85, 255, 255])
        self.color_helipad = (0, 200, 0) # BGR: Hijau
        
        # 2. Korban (Victim / Lifebuoy): Oranye
        self.victim_hsv_lower = np.array([10, 100, 100])
        self.victim_hsv_upper = np.array([25, 255, 255])
        self.color_victim = (0, 140, 255) # BGR: Oranye
        
        # 3. Sumber Bahaya (Hazard / Fire / Shark Zone): Merah (wrap-around di HSV)
        self.hazard_hsv_lower1 = np.array([0, 120, 100])
        self.hazard_hsv_upper1 = np.array([9, 255, 255])
        self.hazard_hsv_lower2 = np.array([170, 120, 100])
        self.hazard_hsv_upper2 = np.array([180, 255, 255])
        self.color_hazard = (0, 0, 230) # BGR: Merah

    def _get_mask_hazard(self, hsv):
        mask1 = cv2.inRange(hsv, self.hazard_hsv_lower1, self.hazard_hsv_upper1)
        mask2 = cv2.inRange(hsv, self.hazard_hsv_lower2, self.hazard_hsv_upper2)
        return cv2.bitwise_or(mask1, mask2)

    def _find_best_contour(self, mask, min_area=150):
        """Mencari kontur terbesar dengan luas minimal tertentu"""
        # Operasi morfologi untuk menghilangkan noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_DILATE, kernel, iterations=1)
        
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not valid_contours:
            return None
        return max(valid_contours, key=cv2.contourArea)

    def _extract_object_info(self, contour, label, color):
        """Mengekstrak bounding box, titik pusat (centroid), dan menyiapkan metadata objek"""
        x, y, w, h = cv2.boundingRect(contour)
        
        # Hitung centroid menggunakan image moments
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = x + w // 2
            cy = y + h // 2
            
        return {
            "label": label,
            "bbox": (x, y, w, h),
            "center": (cx, cy),
            "color": color,
            "area": cv2.contourArea(contour)
        }

    def detect_objects(self, image):
        """
        Mendeteksi ketiga objek: Helipad, Korban, dan Sumber Bahaya.
        """
        # Konversi BGR ke HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Gaussian blur untuk meredam noise kamera aerial UAV
        blurred_hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        
        detections = {}
        
        # 1. Deteksi Helipad
        mask_helipad = cv2.inRange(blurred_hsv, self.helipad_hsv_lower, self.helipad_hsv_upper)
        cnt_helipad = self._find_best_contour(mask_helipad, min_area=200)
        if cnt_helipad is not None:
            detections["Helipad"] = self._extract_object_info(cnt_helipad, "Helipad", self.color_helipad)
            
        # 2. Deteksi Korban
        mask_victim = cv2.inRange(blurred_hsv, self.victim_hsv_lower, self.victim_hsv_upper)
        cnt_victim = self._find_best_contour(mask_victim, min_area=80)
        if cnt_victim is not None:
            detections["Korban"] = self._extract_object_info(cnt_victim, "Korban", self.color_victim)
            
        # 3. Deteksi Sumber Bahaya
        mask_hazard = self._get_mask_hazard(blurred_hsv)
        cnt_hazard = self._find_best_contour(mask_hazard, min_area=100)
        if cnt_hazard is not None:
            detections["Sumber Bahaya"] = self._extract_object_info(cnt_hazard, "Sumber Bahaya", self.color_hazard)
            
        return detections

    def draw_crosshair(self, img, center, color, size=12, thickness=2):
        """Menggambar tanda '+' pada titik tengah/centroid objek"""
        cx, cy = center
        # Garis horizontal
        cv2.line(img, (cx - size, cy), (cx + size, cy), color, thickness, cv2.LINE_AA)
        # Garis vertikal
        cv2.line(img, (cx, cy - size), (cx, cy + size), color, thickness, cv2.LINE_AA)
        # Titik pusat kecil di tengah
        cv2.circle(img, (cx, cy), 2, (255, 255, 255), -1, cv2.LINE_AA)

    def draw_styled_label(self, img, text, x, y, color):
        """Menggambar label teks berlatar belakang kontras agar terbaca jelas"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        pad = 4
        # Posisi background box
        box_y1 = max(0, y - h - pad * 2)
        box_y2 = y
        box_x1 = max(0, x)
        box_x2 = min(img.shape[1], x + w + pad * 2)
        
        # Kotak label berlatar belakang gelap
        cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), (25, 25, 25), -1)
        # Border kecil sesuai warna objek
        cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), color, 1)
        # Teks label
        cv2.putText(img, text, (box_x1 + pad, box_y2 - pad), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def process_frame(self, original_img):
        """
        Memproses citra SAR:
        - Deteksi objek
        - Bounding box, label, dan tanda '+'
        - Jarak Korban ke Sumber Bahaya
        - Garis penghubung & nilai jarak
        - Peringatan status CRITICAL
        """
        result_img = original_img.copy()
        detections = self.detect_objects(original_img)
        
        # 1. Gambar Bounding Box, Label, dan Tanda '+' untuk setiap objek yang terdeteksi
        for name, info in detections.items():
            x, y, w, h = info["bbox"]
            cx, cy = info["center"]
            color = info["color"]
            
            # Bounding box
            cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
            
            # Label
            self.draw_styled_label(result_img, name, x, y, color)
            
            # Tanda "+" di centroid objek
            self.draw_crosshair(result_img, (cx, cy), color, size=12, thickness=2)

        # 2. Analisis Jarak antara "Korban" dan "Sumber Bahaya"
        distance = None
        status = "UNKNOWN"
        is_critical = False
        
        if "Korban" in detections and "Sumber Bahaya" in detections:
            c_victim = detections["Korban"]["center"]
            c_hazard = detections["Sumber Bahaya"]["center"]
            
            # Hitung jarak Euclidean dalam piksel
            dx = c_victim[0] - c_hazard[0]
            dy = c_victim[1] - c_hazard[1]
            distance = math.sqrt(dx * dx + dy * dy)
            
            # Evaluasi batas aman
            if distance < self.safe_threshold:
                status = "CRITICAL"
                is_critical = True
                line_color = (0, 0, 255) # Merah mencolok
            else:
                status = "SAFE"
                is_critical = False
                line_color = (0, 255, 255) # Kuning / Cyan
                
            # Gambar garis penghubung antara Korban dan Sumber Bahaya
            cv2.line(result_img, c_victim, c_hazard, line_color, 2, cv2.LINE_AA)
            
            # Tampilkan nilai jarak di titik tengah (midpoint) garis
            mid_x = (c_victim[0] + c_hazard[0]) // 2
            mid_y = (c_victim[1] + c_hazard[1]) // 2
            
            dist_text = f"Jarak: {distance:.1f} px"
            self.draw_styled_label(result_img, dist_text, mid_x - 45, mid_y - 10, line_color)
            
        elif "Korban" not in detections or "Sumber Bahaya" not in detections:
            status = "OBJEK TIDAK LENGKAP"

        # 3. Tampilkan Banner Status Telemetri SAR di Result Image
        # Header background bar
        h, w = result_img.shape[:2]
        cv2.rectangle(result_img, (0, 0), (w, 45), (20, 20, 20), -1)
        cv2.line(result_img, (0, 45), (w, 45), (70, 70, 70), 2)
        
        # Teks header UAV
        header_text = "SAR UAV RECONNAISSANCE | TELEMETRY ACTIVE"
        cv2.putText(result_img, header_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
        
        # Badge Status
        if status == "CRITICAL":
            badge_color = (0, 0, 220) # Merah peringatan
            status_text = f"[!] STATUS: CRITICAL (< {self.safe_threshold} px)"
        elif status == "SAFE":
            badge_color = (0, 180, 0) # Hijau aman
            status_text = f"[OK] STATUS: SAFE ({distance:.1f} px)"
        else:
            badge_color = (100, 100, 100)
            status_text = f"STATUS: {status}"
            
        badge_size, _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        badge_w = badge_size[0] + 20
        cv2.rectangle(result_img, (w - badge_w - 15, 7), (w - 15, 38), badge_color, -1)
        cv2.putText(result_img, status_text, (w - badge_w - 5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        return result_img, detections, distance, status

    def create_side_by_side_comparison(self, original_img, result_img):
        """
        Menggabungkan citra Original dan Result secara berdampingan (side-by-side)
        lengkap dengan title header masing-masing.
        """
        h, w = original_img.shape[:2]
        header_h = 40
        
        def add_panel_title(img, title, title_color):
            panel = np.zeros((h + header_h, w, 3), dtype=np.uint8)
            # Background header
            cv2.rectangle(panel, (0, 0), (w, header_h), (35, 35, 35), -1)
            cv2.line(panel, (0, header_h), (w, header_h), (60, 60, 60), 2)
            # Text title
            (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            tx = (w - tw) // 2
            cv2.putText(panel, title, (tx, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, title_color, 2, cv2.LINE_AA)
            panel[header_h:, :] = img
            return panel

        panel_orig = add_panel_title(original_img, "ORIGINAL IMAGE", (220, 220, 220))
        panel_res = add_panel_title(result_img, "RESULT (OBJECT DETECTION & TELEMETRY)", (0, 255, 200))
        
        # Gabungkan secara horizontal
        side_by_side = np.hstack([panel_orig, panel_res])
        return side_by_side
