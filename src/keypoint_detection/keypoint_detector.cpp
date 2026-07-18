#include "keypoint_detection/keypoint_detector.hpp"
#include "utils/constants.hpp"
#include "utils/letterbox.hpp"
#include "utils/onnx_helper.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <cmath>

namespace soccer_radar {

KeypointDetector::KeypointDetector() {
    input_blob_.resize(3 * MODEL_HEIGHT * MODEL_WIDTH);
}

bool KeypointDetector::load_model(const std::string& model_path) {
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "KeypointDetector");

        Ort::SessionOptions session_options;
        configure_session_options(session_options, 4);

        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);

        mem_info_ = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = session_->GetInputNameAllocated(0, allocator);
        input_name_ = input_name.get();

        auto input_type_info = session_->GetInputTypeInfo(0);
        auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        input_shape_ = tensor_info.GetShape();
        input_elem_type_ = tensor_info.GetElementType();

        for (auto& dim : input_shape_) {
            if (dim <= 0) dim = 1;
        }
        if (input_shape_.size() >= 4) {
            input_height_ = static_cast<int>(input_shape_[2]);
            input_width_ = static_cast<int>(input_shape_[3]);
        }

        size_t num_outputs = session_->GetOutputCount();
        output_names_.clear();
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(name.get());
        }

        if (num_outputs > 0) {
            auto out_info = session_->GetOutputTypeInfo(0);
            output_elem_type_ = out_info.GetTensorTypeAndShapeInfo().GetElementType();
        }

        input_blob_.resize(static_cast<size_t>(3 * input_height_ * input_width_));
        if (input_elem_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            input_blob_fp16_.resize(input_blob_.size());
        }

        std::cout << "[KeypointDetector] Football-TV2Radar Corner Model loaded: " << model_path
                  << " (input: " << input_width_ << "x" << input_height_
                  << ", format: " << (input_elem_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ? "FP16" : "FP32") << ")" << std::endl;
        return true;

    } catch (const Ort::Exception& e) {
        std::cerr << "[KeypointDetector] ONNX error: " << e.what() << std::endl;
        return false;
    }
}

void KeypointDetector::preprocess(const cv::Mat& frame, std::vector<float>& blob) {
    letterbox_frame(frame, letterboxed_,
                    input_width_, input_height_,
                    LETTERBOX_PAD_TOP, LETTERBOX_PAD_BOTTOM,
                    LETTERBOX_PAD_LEFT, LETTERBOX_PAD_RIGHT);

    const int hw = input_height_ * input_width_;
    blob.resize(static_cast<size_t>(3 * hw));

    float* ch_r = blob.data();
    float* ch_g = ch_r + hw;
    float* ch_b = ch_g + hw;

    for (int row = 0; row < input_height_; ++row) {
        const uint8_t* src_row = letterboxed_.ptr<uint8_t>(row);
        const int row_offset = row * input_width_;
        for (int col = 0; col < input_width_; ++col) {
            const int src_idx = col * 3;
            const int dst_idx = row_offset + col;
            ch_r[dst_idx] = static_cast<float>(src_row[src_idx + 2]) * (1.0f / 255.0f);
            ch_g[dst_idx] = static_cast<float>(src_row[src_idx + 1]) * (1.0f / 255.0f);
            ch_b[dst_idx] = static_cast<float>(src_row[src_idx + 0]) * (1.0f / 255.0f);
        }
    }
}

void KeypointDetector::postprocess(const std::vector<float>& output,
                                    int output_rows, int output_cols,
                                    KeypointData& out) {
    out.clear();
    if (output_rows <= 0 || output_cols <= 0) return;

    bool needs_transpose = (output_cols > output_rows) && (output_cols > 100);
    int num_detections = needs_transpose ? output_cols : output_rows;
    int num_features   = needs_transpose ? output_rows : output_cols;

    std::vector<FieldCorner> candidate_corners;

    for (int i = 0; i < num_detections; ++i) {
        float max_score = 0.0f;
        int max_class = 0;

        if (needs_transpose) {
            for (int c = 0; c < NUM_CORNER_CLASSES && (4 + c) < num_features; ++c) {
                float score = output[(4 + c) * num_detections + i];
                if (score > max_score) {
                    max_score = score;
                    max_class = c;
                }
            }
            if (max_score < KEYPOINT_CONFIDENCE) continue;

            float cx = output[0 * num_detections + i];
            float cy = output[1 * num_detections + i];

            cx -= static_cast<float>(LETTERBOX_PAD_LEFT);
            cy -= static_cast<float>(LETTERBOX_PAD_TOP);

            if (cx >= 0.0f && cx <= static_cast<float>(INPUT_WIDTH) &&
                cy >= 0.0f && cy <= static_cast<float>(INPUT_HEIGHT)) {
                candidate_corners.push_back({cx, cy, max_score, max_class});
            }
        } else {
            const int base = i * num_features;
            for (int c = 0; c < NUM_CORNER_CLASSES && (4 + c) < num_features; ++c) {
                float score = output[base + 4 + c];
                if (score > max_score) {
                    max_score = score;
                    max_class = c;
                }
            }
            if (max_score < KEYPOINT_CONFIDENCE) continue;

            float cx = output[base + 0];
            float cy = output[base + 1];

            cx -= static_cast<float>(LETTERBOX_PAD_LEFT);
            cy -= static_cast<float>(LETTERBOX_PAD_TOP);

            if (cx >= 0.0f && cx <= static_cast<float>(INPUT_WIDTH) &&
                cy >= 0.0f && cy <= static_cast<float>(INPUT_HEIGHT)) {
                candidate_corners.push_back({cx, cy, max_score, max_class});
            }
        }
    }

    apply_nms(candidate_corners);
    out.corners = std::move(candidate_corners);
}

void KeypointDetector::apply_nms(std::vector<FieldCorner>& corners) {
    if (corners.empty()) return;

    std::sort(corners.begin(), corners.end(),
              [](const FieldCorner& a, const FieldCorner& b) { return a.confidence > b.confidence; });

    std::vector<bool> suppressed(corners.size(), false);
    std::vector<FieldCorner> kept;

    constexpr float NMS_RADIUS_SQ = 15.0f * 15.0f;

    for (size_t i = 0; i < corners.size(); ++i) {
        if (suppressed[i]) continue;
        kept.push_back(corners[i]);

        for (size_t j = i + 1; j < corners.size(); ++j) {
            if (suppressed[j]) continue;
            if (corners[i].class_id != corners[j].class_id) continue;

            float dx = corners[i].x - corners[j].x;
            float dy = corners[i].y - corners[j].y;
            if ((dx * dx + dy * dy) < NMS_RADIUS_SQ) {
                suppressed[j] = true;
            }
        }
    }

    corners = std::move(kept);
}

KeypointData KeypointDetector::detect(const cv::Mat& frame) {
    KeypointData result;
    if (!session_) return result;

    preprocess(frame, input_blob_);

    std::array<int64_t, 4> input_shape = {1, 3, input_height_, input_width_};
    Ort::Value input_tensor{nullptr};

    if (input_elem_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        if (input_blob_fp16_.size() != input_blob_.size()) {
            input_blob_fp16_.resize(input_blob_.size());
        }
        for (size_t i = 0; i < input_blob_.size(); ++i) {
            input_blob_fp16_[i] = float_to_half(input_blob_[i]);
        }
        input_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
            *mem_info_, input_blob_fp16_.data(), input_blob_fp16_.size(),
            input_shape.data(), input_shape.size());
    } else {
        input_tensor = Ort::Value::CreateTensor<float>(
            *mem_info_, input_blob_.data(), input_blob_.size(),
            input_shape.data(), input_shape.size());
    }

    const char* input_names[] = { input_name_.c_str() };
    std::vector<const char*> out_names;
    for (auto& n : output_names_) out_names.push_back(n.c_str());

    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        out_names.data(), out_names.size());

    auto& output_tensor = output_tensors[0];
    auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto output_shape = output_info.GetShape();

    int total_elements = 1;
    for (auto d : output_shape) total_elements *= static_cast<int>(d);

    output_data_.resize(total_elements);
    if (output_elem_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        Ort::Float16_t* out_fp16 = output_tensor.GetTensorMutableData<Ort::Float16_t>();
        for (int i = 0; i < total_elements; ++i) {
            output_data_[i] = half_to_float(out_fp16[i]);
        }
    } else {
        float* output_ptr = output_tensor.GetTensorMutableData<float>();
        std::copy(output_ptr, output_ptr + total_elements, output_data_.begin());
    }

    int rows, cols;
    if (output_shape.size() == 3) {
        rows = static_cast<int>(output_shape[1]);
        cols = static_cast<int>(output_shape[2]);
    } else if (output_shape.size() == 2) {
        rows = static_cast<int>(output_shape[0]);
        cols = static_cast<int>(output_shape[1]);
    } else {
        return result;
    }

    postprocess(output_data_, rows, cols, result);
    return result;
}

} // namespace soccer_radar
