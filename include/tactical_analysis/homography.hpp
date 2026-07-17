#pragma once

#include "utils/types.hpp"
#include "utils/constants.hpp"
#include <opencv2/core.hpp>
#include <vector>

namespace soccer_radar {

// Homography transformation between frame coordinates and pitch coordinates.
// Uses OpenCV's findHomography with RANSAC for robust estimation.
class HomographyTransformer {
public:
    explicit HomographyTransformer(float confidence_threshold = KEYPOINT_CONFIDENCE);
    ~HomographyTransformer() = default;

    // Compute homography from detected field keypoints
    // Returns true if homography was successfully computed
    bool compute(const KeypointData& keypoints);

    // Transform points from frame coordinates to pitch coordinates
    std::vector<std::array<float,2>> transform_to_pitch(
        const std::vector<std::array<float,2>>& frame_points) const;

    // Transform points from pitch coordinates to frame coordinates
    std::vector<std::array<float,2>> transform_to_frame(
        const std::vector<std::array<float,2>>& pitch_points) const;

    bool is_valid() const { return valid_; }

    // Draw the tactical pitch view
    static cv::Mat draw_pitch(int width = PITCH_DRAW_WIDTH, int height = PITCH_DRAW_HEIGHT);

    // Draw player/ball/referee positions on the pitch
    static void draw_positions_on_pitch(cv::Mat& pitch,
                                         const std::vector<std::array<float,2>>& team1_pts,
                                         const std::vector<std::array<float,2>>& team2_pts,
                                         const std::vector<std::array<float,2>>& ball_pts,
                                         const std::vector<std::array<float,2>>& ref_pts);

private:
    float confidence_threshold_;
    cv::Mat homography_matrix_;     // frame -> pitch
    cv::Mat inv_homography_matrix_; // pitch -> frame
    bool valid_{false};

    // All pitch reference points (standard + extra)
    std::vector<std::pair<float,float>> all_pitch_points_;
};

} // namespace soccer_radar
