#include "utils/video_io.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <iostream>

namespace soccer_radar {

// ============================================================================
// VideoReader
// ============================================================================

VideoReader::VideoReader(const std::string& path) {
    open(path);
}

bool VideoReader::open(const std::string& path) {
    close();
    cap_ = std::make_unique<cv::VideoCapture>(path);
    if (!cap_->isOpened()) {
        cap_.reset();
        return false;
    }
    width_ = static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_WIDTH));
    height_ = static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_HEIGHT));
    fps_ = cap_->get(cv::CAP_PROP_FPS);
    total_frames_ = static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_COUNT));
    return true;
}

bool VideoReader::read(cv::Mat& frame) {
    if (!is_open()) return false;
    if (!cap_->read(frame) || frame.empty()) return false;
    return true;
}

void VideoReader::close() {
    if (cap_) {
        cap_->release();
        cap_.reset();
    }
}

// ============================================================================
// VideoWriter
// ============================================================================

bool VideoWriter::open(const std::string& path, int width, int height, double fps) {
    close();

    // Multi-codec probe: avc1 -> h264 -> mp4v -> XVID
    int codecs[] = {
        cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
        cv::VideoWriter::fourcc('h', '2', '6', '4'),
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D')
    };

    for (int fourcc : codecs) {
        writer_ = std::make_unique<cv::VideoWriter>(path, fourcc, fps, cv::Size(width, height));
        if (writer_->isOpened()) {
            return true;
        }
        writer_.reset();
    }
    return false;
}

bool VideoWriter::write(const cv::Mat& frame) {
    if (!is_open() || frame.empty()) return false;
    writer_->write(frame);
    return true;
}

void VideoWriter::close() {
    if (writer_) {
        writer_->release();
        writer_.reset();
    }
}

// ============================================================================
// Utilities
// ============================================================================

std::string generate_output_path(const std::string& input_path,
                                 const std::string& suffix) {
    std::filesystem::path p(input_path);
    std::string stem = p.stem().string();
    std::string ext = p.extension().string();
    if (ext.empty()) ext = ".mp4";
    return (p.parent_path() / (stem + suffix + ext)).string();
}

} // namespace soccer_radar
