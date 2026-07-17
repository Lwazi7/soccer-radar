#pragma once

#include "utils/types.hpp"
#include "utils/constants.hpp"
#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

namespace soccer_radar {

class ObjectDetector {
public:
    ObjectDetector();
    ~ObjectDetector();

    ObjectDetector(const ObjectDetector&) = delete;
    ObjectDetector& operator=(const ObjectDetector&) = delete;

    bool load_model(const std::string& model_path);
    bool is_loaded() const { return session_ != nullptr; }

    // Detect objects in a frame (applies letterbox internally)
    // Returns separated detections for players, ball, referees
    void detect(const cv::Mat& frame,
                Detections& players,
                Detections& balls,
                Detections& referees);

    // Detect all objects, return combined
    Detections detect_all(const cv::Mat& frame);

private:
    void preprocess(const cv::Mat& frame, std::vector<float>& blob);
    void postprocess(const std::vector<float>& output,
                     int output_rows, int output_cols,
                     Detections& out);
    void apply_nms(Detections& dets);

    void* session_{nullptr};
    void* env_{nullptr};
    void* mem_info_{nullptr};

    // Pre-allocated buffers
    std::vector<float> input_blob_;
    std::vector<float> output_data_;
    cv::Mat letterboxed_;

    // Input/output metadata
    int input_height_{MODEL_HEIGHT};
    int input_width_{MODEL_WIDTH};
    std::string input_name_;
    std::vector<std::string> output_names_;
    std::vector<int64_t> input_shape_;
};

} // namespace soccer_radar
