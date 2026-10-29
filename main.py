import cv2
import argparse
import os
import matplotlib.pyplot as plt
from generate_samples import create_synthetic_sar_image
from sar_vision import SARDroneVision

def main():
    parser = argparse.ArgumentParser(description="SAR Drone Vision - Object Detection & Safety Analysis")
    parser.add_argument("--image", type=str, default=None, help="Path ke file citra input. Jika kosong, simulasi bawaan akan dibuat.")
    parser.add_argument("--threshold", type=float, default=200.0, help="Batas jarak aman (piksel) antara Korban dan Sumber Bahaya")
    parser.add_argument("--mode", choices=["critical", "safe", "both"], default="both", help="Mode sampel simulasi yang akan dijalankan jika tanpa input file")
    args = parser.parse_args()

    detector = SARDroneVision(safe_distance_threshold=args.threshold)
    output_dir = "/home/hasan/sar_drone_detection/output"
    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: File '{args.image}' tidak ditemukan!")
            return
        img = cv2.imread(args.image)
        tasks.append(("custom_input", img))
    else:
        if args.mode in ["critical", "both"]:
            img_crit = create_synthetic_sar_image(danger_near_victim=True)
            tasks.append(("skenario_critical", img_crit))
        if args.mode in ["safe", "both"]:
            img_safe = create_synthetic_sar_image(danger_near_victim=False)
            tasks.append(("skenario_safe", img_safe))

    for name, orig_img in tasks:
        print("=" * 60)
        print(f"Memproses Skenario: {name}")
        print("=" * 60)
        
        result_img, detections, dist, status = detector.process_frame(orig_img)
        side_by_side = detector.create_side_by_side_comparison(orig_img, result_img)
        
        # Cetak hasil deteksi ke terminal
        print(f"Objek terdeteksi ({len(detections)}):")
        for obj_name, data in detections.items():
            bbox = data["bbox"]
            center = data["center"]
            print(f"  - [{obj_name}] Bounding Box: {bbox}, Centroid (+): {center}")
            
        if dist is not None:
            print(f"Jarak Korban -> Sumber Bahaya: {dist:.2f} px")
            print(f"Batas Aman Threshold: {args.threshold} px")
            print(f"Status Keamanan: {status}")
        else:
            print("Tidak dapat menghitung jarak: Objek 'Korban' atau 'Sumber Bahaya' tidak terdeteksi lengkap.")

        # Simpan hasil citra
        out_path = os.path.join(output_dir, f"{name}_comparison.png")
        cv2.imwrite(out_path, side_by_side)
        print(f"Tersimpan: {out_path}\n")

if __name__ == "__main__":
    main()
