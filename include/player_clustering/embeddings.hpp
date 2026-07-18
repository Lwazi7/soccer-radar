#pragma once

#include "utils/types.hpp"
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

namespace soccer_radar {

class EmbeddingExtractor {
public:
    EmbeddingExtractor();
    ~EmbeddingExtractor() = default;

    EmbeddingExtractor(const EmbeddingExtractor&) = delete;
    EmbeddingExtractor& operator=(const EmbeddingExtractor&) = delete;
    EmbeddingExtractor(EmbeddingExtractor&&) noexcept = default;
    EmbeddingExtractor& operator=(EmbeddingExtractor&&) noexcept = default;

    bool load_model(const std::string& model_path);
    bool is_loaded() const { return session_ != nullptr; }

    std::vector<float> extract(const cv::Mat& crop);
    std::vector<std::vector<float>> extract_batch(const std::vector<cv::Mat>& crops);

    static std::vector<cv::Mat> get_player_crops(const cv::Mat& frame,
                                                  const std::vector<BBox>& boxes);

    int embedding_dim() const { return embedding_dim_; }

private:
    void preprocess(const cv::Mat& image, std::vector<float>& blob);

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::MemoryInfo> mem_info_;
    std::unique_ptr<Ort::Session> session_;

    int input_height_{224};
    int input_width_{224};
    int embedding_dim_{1280};
    ONNXTensorElementDataType input_elem_type_{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};
    ONNXTensorElementDataType output_elem_type_{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};

    std::vector<float> input_blob_;
    std::vector<Ort::Float16_t> input_blob_fp16_;
    std::vector<float> output_data_;
    std::vector<float> batch_input_blob_;
    std::vector<Ort::Float16_t> batch_input_fp16_;
    std::vector<float> batch_output_data_;
    cv::Mat resize_scratch_;
    std::string input_name_;
    std::vector<std::string> output_names_;
};

} // namespace soccer_radar
