#pragma once

#include <opencv2/core.hpp>
#include <string>

namespace soccer_radar {

class VideoReader {
public:
    VideoReader() = default;
    explicit VideoReader(const std::string& path);
    ~VideoReader();

    VideoReader(const VideoReader&) = delete;
    VideoReader& operator=(const VideoReader&) = delete;
    VideoReader(VideoReader&&) noexcept;
    VideoReader& operator=(VideoReader&&) noexcept;

    bool open(const std::string& path);
    bool read(cv::Mat& frame);
    void close();

    bool is_open() const { return cap_ != nullptr; }
    int width()    const { return width_; }
    int height()   const { return height_; }
    double fps()   const { return fps_; }
    int total_frames() const { return total_frames_; }

private:
    void* cap_{nullptr};
    int width_{0};
    int height_{0};
    double fps_{0};
    int total_frames_{0};
};

class VideoWriter {
public:
    VideoWriter() = default;
    ~VideoWriter();

    VideoWriter(const VideoWriter&) = delete;
    VideoWriter& operator=(const VideoWriter&) = delete;

    bool open(const std::string& path, int width, int height, double fps);
    bool write(const cv::Mat& frame);
    void close();

    bool is_open() const { return writer_ != nullptr; }

private:
    void* writer_{nullptr};
};

// Generate output path from input path with suffix
std::string generate_output_path(const std::string& input_path,
                                 const std::string& suffix);

} // namespace soccer_radar
