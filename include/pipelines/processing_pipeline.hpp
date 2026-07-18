#pragma once

#include "utils/types.hpp"
#include "utils/video_io.hpp"
#include <string>
#include <vector>
#include <unordered_map>

namespace soccer_radar {

// Video I/O and ball track interpolation.
// Streaming design: processes frames one at a time to minimize memory usage.
class ProcessingPipeline {
public:
    ProcessingPipeline() = default;
    ~ProcessingPipeline() = default;

    // Interpolate missing ball detections with a constant-acceleration motion model
    // tracks: frame_index -> ball BBox
    void interpolate_ball_tracks(
        std::unordered_map<int, BBox>& ball_tracks,
        int total_frames);

    // Generate output path
    static std::string generate_output_path(const std::string& input,
                                            const std::string& suffix);
};

} // namespace soccer_radar
