#include "player_clustering/embeddings.hpp"
#include "utils/types.hpp"
#include "utils/constants.hpp"
#include "utils/onnx_helper.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace soccer_radar {

EmbeddingExtractor::EmbeddingExtractor() {
    input_blob_.resize(3 * 224 * 224);
}

EmbeddingExtractor::~EmbeddingExtractor() {
    if (session_) {
        delete static_cast<Ort::Session*>(session_);
    }
    if (env_) {
        delete static_cast<Ort::Env*>(env_);
    }
    if (mem_info_) {
        delete static_cast<Ort::MemoryInfo*>(mem_info_);
    }
}

bool EmbeddingExtractor::load_model(const std::string& model_path) {
    try {
        auto* env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "EmbeddingExtractor");
        env_ = env;

        Ort::SessionOptions session_options;
        configure_session_options(session_options, 4);

        auto* session = new Ort::Session(*env, model_path.c_str(), session_options);
        session_ = session;

        auto* mem_info = new Ort::MemoryInfo(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
        mem_info_ = mem_info;

        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = session->GetInputNameAllocated(0, allocator);
        input_name_ = input_name.get();

        auto input_type_info = session->GetInputTypeInfo(0);
        auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();

        if (shape.size() >= 4) {
            if (shape[2] > 0) input_height_ = static_cast<int>(shape[2]);
            if (shape[3] > 0) input_width_ = static_cast<int>(shape[3]);
        }

        size_t num_outputs = session->GetOutputCount();
        output_names_.clear();
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name = session->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(name.get());
        }

        auto output_type_info = session->GetOutputTypeInfo(0);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_shape = output_tensor_info.GetShape();

        embedding_dim_ = 1280;
        for (auto d : output_shape) {
            if (d > 1 && d < 10000) {
                embedding_dim_ = static_cast<int>(d);
                break;
            }
        }

        input_blob_.resize(static_cast<size_t>(3 * input_height_ * input_width_));

        std::cout << "[EmbeddingExtractor] Model loaded: " << model_path
                  << " (input: " << input_width_ << "x" << input_height_
                  << ", embedding_dim: " << embedding_dim_ << ")" << std::endl;
        return true;

    } catch (const Ort::Exception& e) {
        std::cerr << "[EmbeddingExtractor] ONNX error: " << e.what() << std::endl;
        return false;
    }
}

void EmbeddingExtractor::preprocess(const cv::Mat& image, std::vector<float>& blob) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(input_width_, input_height_));

    const int hw = input_height_ * input_width_;
    blob.resize(static_cast<size_t>(3 * hw));

    float* ch_r = blob.data();
    float* ch_g = ch_r + hw;
    float* ch_b = ch_g + hw;

    constexpr float mean_r = 0.485f, mean_g = 0.456f, mean_b = 0.406f;
    constexpr float std_r = 0.229f, std_g = 0.224f, std_b = 0.225f;

    for (int row = 0; row < input_height_; ++row) {
        const uint8_t* src_row = resized.ptr<uint8_t>(row);
        const int row_offset = row * input_width_;
        for (int col = 0; col < input_width_; ++col) {
            const int src_idx = col * 3;
            const int dst_idx = row_offset + col;

            float r = static_cast<float>(src_row[src_idx + 2]) / 255.0f;
            float g = static_cast<float>(src_row[src_idx + 1]) / 255.0f;
            float b = static_cast<float>(src_row[src_idx + 0]) / 255.0f;

            ch_r[dst_idx] = (r - mean_r) / std_r;
            ch_g[dst_idx] = (g - mean_g) / std_g;
            ch_b[dst_idx] = (b - mean_b) / std_b;
        }
    }
}

std::vector<float> EmbeddingExtractor::extract(const cv::Mat& crop) {
    if (!session_) return {};

    auto* sess = static_cast<Ort::Session*>(session_);
    auto* mi = static_cast<Ort::MemoryInfo*>(mem_info_);

    preprocess(crop, input_blob_);

    std::array<int64_t, 4> input_shape = {1, 3, input_height_, input_width_};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        *mi, input_blob_.data(), input_blob_.size(),
        input_shape.data(), input_shape.size());

    const char* input_names[] = { input_name_.c_str() };
    std::vector<const char*> out_names;
    for (auto& n : output_names_) out_names.push_back(n.c_str());

    auto output_tensors = sess->Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        out_names.data(), out_names.size());

    auto& output_tensor = output_tensors[0];
    float* output_ptr = output_tensor.GetTensorMutableData<float>();

    auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto output_shape = output_info.GetShape();

    int total_elements = 1;
    for (auto d : output_shape) total_elements *= static_cast<int>(d);

    std::vector<float> embedding(embedding_dim_, 0);

    if (total_elements == embedding_dim_) {
        std::copy(output_ptr, output_ptr + embedding_dim_, embedding.begin());
    } else if (total_elements > embedding_dim_) {
        int spatial_size = total_elements / embedding_dim_;
        for (int c = 0; c < embedding_dim_; ++c) {
            float sum = 0;
            for (int s = 0; s < spatial_size; ++s) {
                sum += output_ptr[c * spatial_size + s];
            }
            embedding[c] = sum / static_cast<float>(spatial_size);
        }
    }

    float norm = 0;
    for (float v : embedding) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-6f) {
        for (float& v : embedding) v /= norm;
    }

    return embedding;
}

std::vector<std::vector<float>> EmbeddingExtractor::extract_batch(
    const std::vector<cv::Mat>& crops) {
    if (crops.empty() || !session_) return {};
    if (crops.size() == 1) return { extract(crops[0]) };

    const size_t batch_size = crops.size();
    const size_t hw = static_cast<size_t>(input_height_ * input_width_);
    const size_t chw = 3 * hw;

    std::vector<float> batched_blob(batch_size * chw);

    constexpr float mean_r = 0.485f, mean_g = 0.456f, mean_b = 0.406f;
    constexpr float std_r = 0.229f, std_g = 0.224f, std_b = 0.225f;

    for (size_t b = 0; b < batch_size; ++b) {
        cv::Mat resized;
        cv::resize(crops[b], resized, cv::Size(input_width_, input_height_));
        
        float* ch_r = batched_blob.data() + b * chw;
        float* ch_g = ch_r + hw;
        float* ch_b = ch_g + hw;

        for (int r = 0; r < input_height_; ++r) {
            const uint8_t* row_ptr = resized.ptr<uint8_t>(r);
            const int row_offset = r * input_width_;
            for (int c = 0; c < input_width_; ++c) {
                const int src_idx = c * 3;
                const int dst_idx = row_offset + c;
                ch_r[dst_idx] = (static_cast<float>(row_ptr[src_idx + 2]) / 255.0f - mean_r) / std_r;
                ch_g[dst_idx] = (static_cast<float>(row_ptr[src_idx + 1]) / 255.0f - mean_g) / std_g;
                ch_b[dst_idx] = (static_cast<float>(row_ptr[src_idx + 0]) / 255.0f - mean_b) / std_b;
            }
        }
    }

    auto* sess = static_cast<Ort::Session*>(session_);
    auto* mi = static_cast<Ort::MemoryInfo*>(mem_info_);

    std::array<int64_t, 4> input_shape = { static_cast<int64_t>(batch_size), 3, input_height_, input_width_ };
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        *mi, batched_blob.data(), batched_blob.size(),
        input_shape.data(), input_shape.size());

    const char* input_names[] = { input_name_.c_str() };
    std::vector<const char*> out_names;
    for (auto& n : output_names_) out_names.push_back(n.c_str());

    auto output_tensors = sess->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
                                    out_names.data(), out_names.size());

    float* out_ptr = output_tensors[0].GetTensorMutableData<float>();
    auto shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    
    size_t elements_per_sample = 1;
    for (size_t i = 1; i < shape.size(); ++i) elements_per_sample *= static_cast<size_t>(shape[i]);

    std::vector<std::vector<float>> results(batch_size, std::vector<float>(embedding_dim_, 0.0f));

    for (size_t b = 0; b < batch_size; ++b) {
        float* sample_ptr = out_ptr + b * elements_per_sample;
        auto& emb = results[b];

        if (elements_per_sample == static_cast<size_t>(embedding_dim_)) {
            std::copy(sample_ptr, sample_ptr + embedding_dim_, emb.begin());
        } else {
            size_t spatial_size = elements_per_sample / embedding_dim_;
            for (int c = 0; c < embedding_dim_; ++c) {
                float sum = 0.0f;
                for (size_t s = 0; s < spatial_size; ++s) sum += sample_ptr[c * spatial_size + s];
                emb[c] = sum / static_cast<float>(spatial_size);
            }
        }

        float norm = 0.0f;
        for (float v : emb) norm += v * v;
        norm = std::sqrt(norm);
        if (norm > 1e-6f) {
            for (float& v : emb) v /= norm;
        }
    }

    return results;
}

std::vector<cv::Mat> EmbeddingExtractor::get_player_crops(const cv::Mat& frame,
                                                            const std::vector<BBox>& boxes) {
    std::vector<cv::Mat> crops;
    crops.reserve(boxes.size());

    for (const auto& box : boxes) {
        int x1 = std::max(0, static_cast<int>(box.x1));
        int y1 = std::max(0, static_cast<int>(box.y1));
        int x2 = std::min(frame.cols, static_cast<int>(box.x2));
        int y2 = std::min(frame.rows, static_cast<int>(box.y2));

        if (x2 > x1 && y2 > y1) {
            cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
            crops.push_back(frame(roi).clone());
        }
    }

    return crops;
}

} // namespace soccer_radar
