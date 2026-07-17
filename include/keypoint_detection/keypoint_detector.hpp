#pragma once

#include "utils/types.hpp"
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

namespace soccer_radar {

class KeypointDetector {
public:
    KeypointDetector();
    ~KeypointDetector();

    KeypointDetector(const KeypointDetector&) = delete;
    KeypointDetector& operator=(const KeypointDetector&) = delete;

    bool load_model(const std::string& model_path);
    bool is_loaded() const { return session_ != nullptr; }

    KeypointData detect(const cv::Mat& frame);

    int input_width()  const { return input_width_; }
    int input_height() const { return input_height_; }

private:
    void preprocess(const cv::Mat& frame, std::vector<float>& blob);
    void postprocess(const std::vector<float>& output, int output_rows, int output_cols, KeypointData& out);
    static void apply_nms(std::vector<FieldCorner>& corners);

    void* session_{nullptr};
    void* env_{nullptr};
    void* mem_info_{nullptr};

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
