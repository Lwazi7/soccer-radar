#include "object_detection/object_detector.hpp"
#include "utils/letterbox.hpp"
#include "utils/onnx_helper.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <numeric>
#include <cmath>

namespace soccer_radar {

ObjectDetector::ObjectDetector() {
    input_blob_.resize(3 * MODEL_HEIGHT * MODEL_WIDTH);
}

ObjectDetector::~ObjectDetector() {
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

bool ObjectDetector::load_model(const std::string& model_path) {
    try {
        auto* env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ObjectDetector");
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
        input_shape_ = tensor_info.GetShape();

        for (auto& dim : input_shape_) {
            if (dim <= 0) dim = 1;
        }
        if (input_shape_.size() >= 4) {
            input_height_ = static_cast<int>(input_shape_[2]);
            input_width_ = static_cast<int>(input_shape_[3]);
        }

        size_t num_outputs = session->GetOutputCount();
        output_names_.clear();
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name = session->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(name.get());
        }

        input_blob_.resize(static_cast<size_t>(3 * input_height_ * input_width_));

        std::cout << "[ObjectDetector] Model loaded: " << model_path
                  << " (input: " << input_width_ << "x" << input_height_ << ")" << std::endl;
        return true;

    } catch (const Ort::Exception& e) {
        std::cerr << "[ObjectDetector] ONNX error: " << e.what() << std::endl;
        return false;
    }
}

void ObjectDetector::preprocess(const cv::Mat& frame, std::vector<float>& blob) {
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

void ObjectDetector::postprocess(const std::vector<float>& output,
                                  int output_rows, int output_cols,
                                  Detections& out) {
    out.clear();
    if (output_rows <= 0 || output_cols <= 0) return;

    bool needs_transpose = (output_cols > output_rows) && (output_cols > 100);
    int num_detections = needs_transpose ? output_cols : output_rows;
    int num_features   = needs_transpose ? output_rows : output_cols;

    for (int i = 0; i < num_detections; ++i) {
        float max_score = 0.0f;
        int max_class = 0;

        if (needs_transpose) {
            for (int c = 0; c < NUM_CLASSES && (4 + c) < num_features; ++c) {
                float score = output[(4 + c) * num_detections + i];
                if (score > max_score) {
                    max_score = score;
                    max_class = c;
                }
            }
            if (max_score < CONFIDENCE_THRESHOLD) continue;

            float cx = output[0 * num_detections + i];
            float cy = output[1 * num_detections + i];
            float w  = output[2 * num_detections + i];
            float h  = output[3 * num_detections + i];

            BBox bbox;
            bbox.x1 = cx - w * 0.5f;
            bbox.y1 = cy - h * 0.5f;
            bbox.x2 = cx + w * 0.5f;
            bbox.y2 = cy + h * 0.5f;
            bbox.confidence = max_score;
            bbox.class_id = max_class;

            unletterbox_bbox(bbox.x1, bbox.y1, bbox.x2, bbox.y2,
                             LETTERBOX_PAD_LEFT, LETTERBOX_PAD_TOP);

            bbox.x1 = std::max(0.0f, std::min(bbox.x1, static_cast<float>(INPUT_WIDTH)));
            bbox.y1 = std::max(0.0f, std::min(bbox.y1, static_cast<float>(INPUT_HEIGHT)));
            bbox.x2 = std::max(0.0f, std::min(bbox.x2, static_cast<float>(INPUT_WIDTH)));
            bbox.y2 = std::max(0.0f, std::min(bbox.y2, static_cast<float>(INPUT_HEIGHT)));

            if (bbox.valid()) {
                out.add(bbox);
            }
        } else {
            const int base = i * num_features;
            for (int c = 0; c < NUM_CLASSES && (4 + c) < num_features; ++c) {
                float score = output[base + 4 + c];
                if (score > max_score) {
                    max_score = score;
                    max_class = c;
                }
            }
            if (max_score < CONFIDENCE_THRESHOLD) continue;

            float cx = output[base + 0];
            float cy = output[base + 1];
            float w  = output[base + 2];
            float h  = output[base + 3];

            BBox bbox;
            bbox.x1 = cx - w * 0.5f;
            bbox.y1 = cy - h * 0.5f;
            bbox.x2 = cx + w * 0.5f;
            bbox.y2 = cy + h * 0.5f;
            bbox.confidence = max_score;
            bbox.class_id = max_class;

            unletterbox_bbox(bbox.x1, bbox.y1, bbox.x2, bbox.y2,
                             LETTERBOX_PAD_LEFT, LETTERBOX_PAD_TOP);

            bbox.x1 = std::max(0.0f, std::min(bbox.x1, static_cast<float>(INPUT_WIDTH)));
            bbox.y1 = std::max(0.0f, std::min(bbox.y1, static_cast<float>(INPUT_HEIGHT)));
            bbox.x2 = std::max(0.0f, std::min(bbox.x2, static_cast<float>(INPUT_WIDTH)));
            bbox.y2 = std::max(0.0f, std::min(bbox.y2, static_cast<float>(INPUT_HEIGHT)));

            if (bbox.valid()) {
                out.add(bbox);
            }
        }
    }

    apply_nms(out);
}

void ObjectDetector::apply_nms(Detections& dets) {
    if (dets.empty()) return;

    std::sort(dets.boxes.begin(), dets.boxes.end(),
              [](const BBox& a, const BBox& b) { return a.confidence > b.confidence; });

    std::vector<bool> suppressed(dets.size(), false);
    Detections kept;

    for (int i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        kept.add(dets.boxes[i]);

        for (int j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j]) continue;
            if (dets.boxes[i].class_id != dets.boxes[j].class_id) continue;

            float x1 = std::max(dets.boxes[i].x1, dets.boxes[j].x1);
            float y1 = std::max(dets.boxes[i].y1, dets.boxes[j].y1);
            float x2 = std::min(dets.boxes[i].x2, dets.boxes[j].x2);
            float y2 = std::min(dets.boxes[i].y2, dets.boxes[j].y2);

            float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
            float uni = dets.boxes[i].area() + dets.boxes[j].area() - inter;
            float iou = (uni > 0.0f) ? inter / uni : 0.0f;

            if (iou > NMS_THRESHOLD) {
                suppressed[j] = true;
            }
        }
    }

    dets = std::move(kept);
}

void ObjectDetector::detect(const cv::Mat& frame,
                             Detections& players,
                             Detections& balls,
                             Detections& referees) {
    if (!session_) return;

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

    int rows, cols;
    if (output_shape.size() == 3) {
        rows = static_cast<int>(output_shape[1]);
        cols = static_cast<int>(output_shape[2]);
    } else if (output_shape.size() == 2) {
        rows = static_cast<int>(output_shape[0]);
        cols = static_cast<int>(output_shape[1]);
    } else {
        return;
    }

    Detections all_dets;
    postprocess(output_data_, rows, cols, all_dets);

    players = all_dets.filter(static_cast<int>(ObjectClass::Player));
    balls = all_dets.filter(static_cast<int>(ObjectClass::Ball));
    referees = all_dets.filter(static_cast<int>(ObjectClass::Referee));
}

Detections ObjectDetector::detect_all(const cv::Mat& frame) {
    Detections players, balls, referees;
    detect(frame, players, balls, referees);

    Detections all;
    for (auto& b : players.boxes) all.add(b);
    for (auto& b : balls.boxes) all.add(b);
    for (auto& b : referees.boxes) all.add(b);
    return all;
}

} // namespace soccer_radar
