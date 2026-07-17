#include "utils/video_io.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>

namespace soccer_radar {

// ============================================================================
// VideoReader
// ============================================================================

VideoReader::VideoReader(const std::string& path) {
    open(path);
}

VideoReader::~VideoReader() {
    close();
}

VideoReader::VideoReader(VideoReader&& other) noexcept
    : cap_(other.cap_), width_(other.width_), height_(other.height_),
      fps_(other.fps_), total_frames_(other.total_frames_) {
    other.cap_ = nullptr;
}

VideoReader& VideoReader::operator=(VideoReader&& other) noexcept {
    if (this != &other) {
        close();
        cap_ = other.cap_;
        width_ = other.width_;
        height_ = other.height_;
        fps_ = other.fps_;
        total_frames_ = other.total_frames_;
        other.cap_ = nullptr;
    }
    return *this;
}

bool VideoReader::open(const std::string& path) {
    close();
    auto* cap = new cv::VideoCapture(path);
    if (!cap->isOpened()) {
        delete cap;
        return false;
    }
    cap_ = cap;
    width_ = static_cast<int>(cap->get(cv::CAP_PROP_FRAME_WIDTH));
    height_ = static_cast<int>(cap->get(cv::CAP_PROP_FRAME_HEIGHT));
    fps_ = cap->get(cv::CAP_PROP_FPS);
    total_frames_ = static_cast<int>(cap->get(cv::CAP_PROP_FRAME_COUNT));
    return true;
}

bool VideoReader::read(cv::Mat& frame) {
    if (!cap_) return false;
    auto* cap = static_cast<cv::VideoCapture*>(cap_);
    return cap->read(frame);
}

void VideoReader::close() {
    if (cap_) {
        auto* cap = static_cast<cv::VideoCapture*>(cap_);
        cap->release();
        delete cap;
        cap_ = nullptr;
    }
}

// ============================================================================
// VideoWriter
// ============================================================================

VideoWriter::~VideoWriter() {
    close();
}

bool VideoWriter::open(const std::string& path, int width, int height, double fps) {
    close();

    // Use H.264 (mp4v) codec which is widely supported
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    auto* writer = new cv::VideoWriter(path, fourcc, fps, cv::Size(width, height));
    if (!writer->isOpened()) {
        delete writer;
        return false;
    }
    writer_ = writer;
    return true;
}

bool VideoWriter::write(const cv::Mat& frame) {
    if (!writer_) return false;
    auto* w = static_cast<cv::VideoWriter*>(writer_);
    w->write(frame);
    return true;
}

void VideoWriter::close() {
    if (writer_) {
        auto* w = static_cast<cv::VideoWriter*>(writer_);
        w->release();
        delete w;
        writer_ = nullptr;
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
