#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <memory>

namespace soccer_radar {

class VideoReader {
public:
    VideoReader() = default;
    explicit VideoReader(const std::string& path);
    ~VideoReader() = default;

    VideoReader(const VideoReader&) = delete;
    VideoReader& operator=(const VideoReader&) = delete;
    VideoReader(VideoReader&&) noexcept = default;
    VideoReader& operator=(VideoReader&&) noexcept = default;

    bool open(const std::string& path);
    bool read(cv::Mat& frame);
    void close();

    bool is_open() const { return cap_ != nullptr && cap_->isOpened(); }
    int width()    const { return width_; }
    int height()   const { return height_; }
    double fps()   const { return fps_; }
    int total_frames() const { return total_frames_; }

private:
    std::unique_ptr<cv::VideoCapture> cap_;
    int width_{0};
    int height_{0};
    double fps_{0.0};
    int total_frames_{0};
};

class VideoWriter {
public:
    VideoWriter() = default;
    ~VideoWriter() = default;

    VideoWriter(const VideoWriter&) = delete;
    VideoWriter& operator=(const VideoWriter&) = delete;
    VideoWriter(VideoWriter&&) noexcept = default;
    VideoWriter& operator=(VideoWriter&&) noexcept = default;

    bool open(const std::string& path, int width, int height, double fps);
    bool write(const cv::Mat& frame);
    void close();

    bool is_open() const { return writer_ != nullptr && writer_->isOpened(); }

private:
    std::unique_ptr<cv::VideoWriter> writer_;
};

// Generate output path from input path with suffix
std::string generate_output_path(const std::string& input_path,
                                 const std::string& suffix);

} // namespace soccer_radar
