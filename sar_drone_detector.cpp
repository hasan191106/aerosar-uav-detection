/**
 * ============================================================================
 * SAR (Search and Rescue) Drone Vision System (C++ OpenCV Implementation)
 * Deteksi Helipad, Korban, dan Sumber Bahaya Menggunakan C++ OpenCV
 * ============================================================================
 * Kompilasi:
 *   g++ -O3 sar_drone_detector.cpp -o sar_drone_detector `pkg-config --cflags --libs opencv4`
 * Eksekusi:
 *   ./sar_drone_detector [path_citra] [threshold_jarak]
 * ============================================================================
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <map>

struct DetectedObject {
    std::string label;
    cv::Rect bbox;
    cv::Point center;
    cv::Scalar color;
    double area;
};

class SARDroneVision {
private:
    double safe_threshold;
    cv::Scalar color_helipad = cv::Scalar(0, 200, 0);   // Hijau (BGR)
    cv::Scalar color_victim  = cv::Scalar(0, 140, 255); // Oranye (BGR)
    cv::Scalar color_hazard  = cv::Scalar(0, 0, 230);   // Merah (BGR)

public:
    SARDroneVision(double threshold = 200.0) : safe_threshold(threshold) {}

    cv::Rect findBestContour(const cv::Mat& mask, cv::Point& centroid, double min_area = 80.0) {
        cv::Mat cleaned;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(mask, cleaned, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(cleaned, cleaned, cv::MORPH_DILATE, kernel);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(cleaned, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        double max_area = 0;
        int best_idx = -1;
        for (size_t i = 0; i < contours.size(); ++i) {
            double area = cv::contourArea(contours[i]);
            if (area >= min_area && area > max_area) {
                max_area = area;
                best_idx = static_cast<int>(i);
            }
        }

        if (best_idx == -1) return cv::Rect();

        cv::Moments m = cv::moments(contours[best_idx]);
        if (m.m00 != 0) {
            centroid = cv::Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
        } else {
            cv::Rect r = cv::boundingRect(contours[best_idx]);
            centroid = cv::Point(r.x + r.width / 2, r.y + r.height / 2);
        }
        return cv::boundingRect(contours[best_idx]);
    }

    std::map<std::string, DetectedObject> detectObjects(const cv::Mat& bgr_img) {
        cv::Mat hsv, blurred;
        cv::cvtColor(bgr_img, hsv, cv::COLOR_BGR2HSV);
        cv::GaussianBlur(hsv, blurred, cv::Size(5, 5), 0);

        std::map<std::string, DetectedObject> results;

        // 1. Helipad (Hijau pada HSV)
        cv::Mat mask_helipad;
        cv::inRange(blurred, cv::Scalar(35, 70, 70), cv::Scalar(85, 255, 255), mask_helipad);
        cv::Point c_h;
        cv::Rect r_h = findBestContour(mask_helipad, c_h, 200.0);
        if (r_h.area() > 0) {
            results["Helipad"] = {"Helipad", r_h, c_h, color_helipad, static_cast<double>(r_h.area())};
        }

        // 2. Korban (Oranye pada HSV)
        cv::Mat mask_victim;
        cv::inRange(blurred, cv::Scalar(10, 100, 100), cv::Scalar(25, 255, 255), mask_victim);
        cv::Point c_v;
        cv::Rect r_v = findBestContour(mask_victim, c_v, 80.0);
        if (r_v.area() > 0) {
            results["Korban"] = {"Korban", r_v, c_v, color_victim, static_cast<double>(r_v.area())};
        }

        // 3. Sumber Bahaya (Merah: Range 0-9 dan 170-180 pada HSV)
        cv::Mat m1, m2, mask_hazard;
        cv::inRange(blurred, cv::Scalar(0, 120, 100), cv::Scalar(9, 255, 255), m1);
        cv::inRange(blurred, cv::Scalar(170, 120, 100), cv::Scalar(180, 255, 255), m2);
        cv::bitwise_or(m1, m2, mask_hazard);
        cv::Point c_d;
        cv::Rect r_d = findBestContour(mask_hazard, c_d, 100.0);
        if (r_d.area() > 0) {
            results["Sumber Bahaya"] = {"Sumber Bahaya", r_d, c_d, color_hazard, static_cast<double>(r_d.area())};
        }

        return results;
    }

    void drawTag(cv::Mat& img, const std::string& text, int x, int y, const cv::Scalar& color) {
        int baseline = 0;
        double scale = 0.55;
        int thickness = 2;
        cv::Size sz = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
        int pad = 4;

        int bx1 = std::max(0, x);
        int by1 = std::max(0, y - sz.height - pad * 2);
        int bx2 = std::min(img.cols, x + sz.width + pad * 2);
        int by2 = y;

        cv::rectangle(img, cv::Point(bx1, by1), cv::Point(bx2, by2), cv::Scalar(25, 25, 25), -1);
        cv::rectangle(img, cv::Point(bx1, by1), cv::Point(bx2, by2), color, 1);
        cv::putText(img, text, cv::Point(bx1 + pad, by2 - pad), cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
    }

    cv::Mat process(const cv::Mat& orig_img, double& distance, std::string& status) {
        cv::Mat result = orig_img.clone();
        auto detections = detectObjects(orig_img);

        // Gambar Bounding Box, Label, dan Tanda '+'
        for (const auto& kv : detections) {
            const auto& obj = kv.second;
            cv::rectangle(result, obj.bbox, obj.color, 2, cv::LINE_AA);
            drawTag(result, obj.label, obj.bbox.x, obj.bbox.y, obj.color);
            // Tanda '+' pada Centroid
            cv::drawMarker(result, obj.center, obj.color, cv::MARKER_CROSS, 24, 2, cv::LINE_AA);
        }

        distance = -1.0;
        status = "UNKNOWN";

        // Hitung Jarak Korban <-> Sumber Bahaya
        if (detections.count("Korban") && detections.count("Sumber Bahaya")) {
            cv::Point p1 = detections["Korban"].center;
            cv::Point p2 = detections["Sumber Bahaya"].center;
            double dx = p1.x - p2.x;
            double dy = p1.y - p2.y;
            distance = std::sqrt(dx * dx + dy * dy);

            cv::Scalar line_color;
            if (distance < safe_threshold) {
                status = "CRITICAL";
                line_color = cv::Scalar(0, 0, 255); // Merah
            } else {
                status = "SAFE";
                line_color = cv::Scalar(0, 255, 255); // Kuning
            }

            cv::line(result, p1, p2, line_color, 2, cv::LINE_AA);
            cv::Point mid((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);

            char dist_str[64];
            std::snprintf(dist_str, sizeof(dist_str), "Jarak: %.1f px", distance);
            drawTag(result, dist_str, mid.x - 45, mid.y - 8, line_color);
        }

        // Status Header
        cv::rectangle(result, cv::Point(0, 0), cv::Point(result.cols, 42), cv::Scalar(20, 20, 20), -1);
        cv::putText(result, "SAR UAV DRONE RECONNAISSANCE SYSTEM", cv::Point(15, 27),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 2, cv::LINE_AA);

        cv::Scalar badge_color = (status == "CRITICAL") ? cv::Scalar(0, 0, 220) : cv::Scalar(0, 180, 0);
        std::string badge_text = (status == "CRITICAL") ? "STATUS: CRITICAL" : "STATUS: SAFE";
        drawTag(result, badge_text, result.cols - 180, 36, badge_color);

        return result;
    }

    cv::Mat makeSideBySide(const cv::Mat& orig, const cv::Mat& res) {
        cv::Mat side_by_side;
        cv::hconcat(orig, res, side_by_side);
        return side_by_side;
    }
};

int main(int argc, char** argv) {
    std::cout << "SAR Drone Vision C++ System Ready." << std::endl;
    return 0;
}
