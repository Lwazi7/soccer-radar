#pragma once

#include "utils/types.hpp"
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace soccer_radar {

// MobileNetV4 Conv Small embedding extractor.
// Extracts feature embeddings from player crop images using ONNX Runtime.
// Replaces the original SigLIP embeddings with a much lighter model
// suitable for constrained devices.
class EmbeddingExtractor {
public:
    EmbeddingExtractor();
    ~EmbeddingExtractor();

    EmbeddingExtractor(const EmbeddingExtractor&) = delete;
    EmbeddingExtractor& operator=(const EmbeddingExtractor&) = delete;

    bool load_model(const std::string& model_path);
    bool is_loaded() const { return session_ != nullptr; }

    // Extract embedding from a single crop image
    // Returns embedding vector of dimension EMBEDDING_DIM
    std::vector<float> extract(const cv::Mat& crop);

    // Extract embeddings from multiple crops (batch processing)
    // More efficient than calling extract() multiple times
    std::vector<std::vector<float>> extract_batch(const std::vector<cv::Mat>& crops);

    // Get player crops from frame using detection bounding boxes
    static std::vector<cv::Mat> get_player_crops(const cv::Mat& frame,
                                                  const std::vector<BBox>& boxes);

    int embedding_dim() const { return embedding_dim_; }

private:
    void preprocess(const cv::Mat& image, std::vector<float>& blob);

    void* session_{nullptr};
    void* env_{nullptr};
    void* mem_info_{nullptr};

    int input_height_{224};
    int input_width_{224};
    int embedding_dim_{1280}; // MobileNetV4 Conv Small output dimension

    std::vector<float> input_blob_;
    std::vector<float> output_data_;
    std::string input_name_;
    std::vector<std::string> output_names_;
};

} // namespace soccer_radar
