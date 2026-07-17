#include "keypoint_detection/keypoint_detector.hpp"
#include "utils/letterbox.hpp"
#include "utils/onnx_helper.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>
#include <cstring>
#include <cmath>
#include <iostream>

namespace soccer_radar {

KeypointDetector::KeypointDetector() {
    input_blob_.resize(3 * MODEL_HEIGHT * MODEL_WIDTH);
}

KeypointDetector::~KeypointDetector() {
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

bool KeypointDetector::load_model(const std::string& model_path) {
    try {
        auto* env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "KeypointDetector");
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

        input_blob_.resize(static_cast<size_t>(3 * input_height_ * input_width_));

        std::cout << "[KeypointDetector] Model loaded: " << model_path
                  << " (input: " << input_width_ << "x" << input_height_ << ")" << std::endl;
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

void KeypointDetector::postprocess(const std::vector<float>& output, KeypointData& out) {
    out.clear();

    int total = static_cast<int>(output.size());
    const int features = 92;
    const int kpt_offset = 5;
    const int kpt_stride = 3;

    int num_anchors = total / features;
    if (num_anchors <= 0 || total != features * num_anchors) return;

    float best_score = -1.0f;
    int best_anchor = -1;

    for (int i = 0; i < num_anchors; ++i) {
        float score = output[4 * num_anchors + i];
        if (score > best_score) {
            best_score = score;
            best_anchor = i;
        }
    }

    if (best_anchor < 0 || best_score < CONFIDENCE_THRESHOLD) {
        return;
    }

    for (int k = 0; k < NUM_KEYPOINTS; ++k) {
        int row_x   = kpt_offset + k * kpt_stride + 0;
        int row_y   = kpt_offset + k * kpt_stride + 1;
        int row_conf = kpt_offset + k * kpt_stride + 2;

        float x    = output[row_x * num_anchors + best_anchor];
        float y    = output[row_y * num_anchors + best_anchor];
        float raw_conf = output[row_conf * num_anchors + best_anchor];

        float conf = 1.0f / (1.0f + std::exp(-raw_conf));

        x -= static_cast<float>(LETTERBOX_PAD_LEFT);
        y -= static_cast<float>(LETTERBOX_PAD_TOP);

        out.points.push_back({x, y, conf});
    }
}

KeypointData KeypointDetector::detect(const cv::Mat& frame) {
    KeypointData result;
    if (!session_) return result;

    auto* sess = static_cast<Ort::Session*>(session_);
    auto* mi = static_cast<Ort::MemoryInfo*>(mem_info_);

    preprocess(frame, input_blob_);

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
    auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto output_shape = output_info.GetShape();
    float* output_ptr = output_tensor.GetTensorMutableData<float>();

    int total_elements = 1;
    for (auto d : output_shape) total_elements *= static_cast<int>(d);

    output_data_.assign(output_ptr, output_ptr + total_elements);
    postprocess(output_data_, result);

    return result;
}

} // namespace soccer_radar
