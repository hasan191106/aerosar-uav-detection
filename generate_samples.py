import cv2
import numpy as np
import math
import os

def create_synthetic_sar_image(width=800, height=600, danger_near_victim=True):
    """
    Membuat citra simulasi aerial camera drone SAR:
    - Laut bertekstur
    - Dermaga pelabuhan dengan Helipad (lingkaran hijau dengan huruf 'H')
    - Korban di perairan (pelampung oranye / life buoy)
    - Sumber Bahaya (tumpahan minyak terbakar / bahaya merah)
    """
    # 1. Background Laut (gradasi biru laut dalam)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        # Gradasi biru dari atas ke bawah
        b = int(140 + 40 * (y / height))
        g = int(80 + 20 * (y / height))
        r = int(20 + 10 * (y / height))
        image[y, :] = [b, g, r]

    # Beri sedikit noise / tekstur ombak laut
    noise = np.random.normal(0, 8, (height, width, 3)).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 2. Dermaga Pelabuhan (Dock / Pier) di sisi kiri atas
    dock_color = (130, 130, 130) # Abu-abu beton
    cv2.rectangle(image, (0, 0), (220, 240), dock_color, -1)
    # Garis tepi dermaga
    cv2.rectangle(image, (0, 0), (220, 240), (80, 80, 80), 3)

    # --- Objek 1: Helipad di dermaga ---
    # Lingkaran hijau menyala (Green Pad) dengan huruf 'H'
    helipad_center = (110, 120)
    helipad_radius = 45
    cv2.circle(image, helipad_center, helipad_radius, (30, 200, 30), -1)      # Lingkaran hijau (BGR)
    cv2.circle(image, helipad_center, helipad_radius, (255, 255, 255), 3)    # Border putih
    cv2.circle(image, helipad_center, helipad_radius - 10, (20, 160, 20), 2)
    # Tulis simbol 'H'
    cv2.putText(image, "H", (helipad_center[0] - 18, helipad_center[1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4, cv2.LINE_AA)

    # --- Objek 2: Korban di perairan ---
    # Titik korban: di area tengah laut
    victim_center = (520, 320)
    # Buat efek pelampung oranye terang (Orange Lifebuoy)
    cv2.circle(image, victim_center, 26, (0, 120, 255), -1)   # Oranye (BGR)
    cv2.circle(image, victim_center, 12, (255, 255, 255), -1) # Tengah putih
    cv2.circle(image, victim_center, 7, (0, 120, 255), -1)    # Kepala orang di dalam
    cv2.circle(image, victim_center, 26, (0, 80, 200), 2)

    # --- Objek 3: Sumber Bahaya ---
    if danger_near_victim:
        # Dekat korban (< 200 piksel) -> Trigger CRITICAL
        hazard_center = (600, 390) # Jarak sekitar ~106 piksel
    else:
        # Jauh dari korban (> 200 piksel) -> Status SAFE
        hazard_center = (680, 140) # Jarak sekitar ~240 piksel

    # Gambar bentuk bahaya (merah menyala dengan gerigi/segitiga/lingkaran)
    cv2.circle(image, hazard_center, 32, (20, 20, 240), -1)  # Merah (BGR)
    cv2.circle(image, hazard_center, 32, (0, 0, 180), 3)
    # Tanda seru bahaya di tengah
    cv2.putText(image, "!", (hazard_center[0] - 7, hazard_center[1] + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

    return image

if __name__ == "__main__":
    img_critical = create_synthetic_sar_image(danger_near_victim=True)
    cv2.imwrite("/home/hasan/sar_drone_detection/sample_critical.png", img_critical)
    print("Saved sample_critical.png")
    
    img_safe = create_synthetic_sar_image(danger_near_victim=False)
    cv2.imwrite("/home/hasan/sar_drone_detection/sample_safe.png", img_safe)
    print("Saved sample_safe.png")
