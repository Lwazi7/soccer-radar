#pragma once

#include "utils/types.hpp"
#include "utils/constants.hpp"
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

namespace soccer_radar {

class ObjectDetector {
public:
    ObjectDetector();
    ~ObjectDetector() = default;

    ObjectDetector(const ObjectDetector&) = delete;
    ObjectDetector& operator=(const ObjectDetector&) = delete;
    ObjectDetector(ObjectDetector&&) noexcept = default;
    ObjectDetector& operator=(ObjectDetector&&) noexcept = default;

    bool load_model(const std::string& model_path);
    bool is_loaded() const { return session_ != nullptr; }

    void detect(const cv::Mat& frame,
                Detections& players,
                Detections& balls,
                Detections& referees,
                DetectionTiming* timing = nullptr);

    Detections detect_all(const cv::Mat& frame, DetectionTiming* timing = nullptr);

    int input_width()  const { return input_width_; }
    int input_height() const { return input_height_; }

private:
    void preprocess(const cv::Mat& frame, std::vector<float>& blob);
    void postprocess(const std::vector<float>& output,
                     int output_rows, int output_cols,
                     Detections& out);
    static void apply_nms(Detections& dets);

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::MemoryInfo> mem_info_;
    std::unique_ptr<Ort::Session> session_;

    int input_height_{736};
    int input_width_{1280};
    std::vector<int64_t> input_shape_;
    ONNXTensorElementDataType input_elem_type_{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};
    ONNXTensorElementDataType output_elem_type_{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};

    std::vector<float> input_blob_;
    std::vector<Ort::Float16_t> input_blob_fp16_;
    std::vector<float> output_data_;
    cv::Mat letterboxed_;

    std::string input_name_;
    std::vector<std::string> output_names_;
};

} // namespace soccer_radar
