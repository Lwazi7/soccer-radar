#pragma once

#include "utils/types.hpp"
#include "utils/constants.hpp"
#include <opencv2/core.hpp>
#include <vector>
#include <unordered_map>

namespace soccer_radar {

// Homography transformation using Football-TV2Radar combinatorial RANSAC solver.
// Solves pitch symmetry ambiguity across 13 corner/intersection classes (in meters: [-52.5, 52.5] x [-34, 34]).
class HomographyTransformer {
public:
    explicit HomographyTransformer(float confidence_threshold = KEYPOINT_CONFIDENCE);
    ~HomographyTransformer() = default;

    // Compute homography from detected corners using Football-TV2Radar solver
    bool compute(const KeypointData& keypoints);

    // Transform points from frame coordinates (pixels) to pitch coordinates (meters)
    std::vector<std::array<float,2>> transform_to_pitch(
        const std::vector<std::array<float,2>>& frame_points) const;

    // Transform points from pitch coordinates (meters) to frame coordinates (pixels)
    std::vector<std::array<float,2>> transform_to_frame(
        const std::vector<std::array<float,2>>& pitch_points) const;

    bool is_valid() const { return valid_; }

    // Draw the 2D tactical pitch view (mapped in meters)
    static cv::Mat draw_pitch(int width = PITCH_DRAW_WIDTH, int height = PITCH_DRAW_HEIGHT);

    // Draw player/ball/referee positions on the pitch
    static void draw_positions_on_pitch(cv::Mat& pitch,
                                         const std::vector<std::array<float,2>>& team1_pts,
                                         const std::vector<std::array<float,2>>& team2_pts,
                                         const std::vector<std::array<float,2>>& ball_pts,
                                         const std::vector<std::array<float,2>>& ref_pts);

private:
    static bool are_collinear(const FieldCorner& p1, const FieldCorner& p2, const FieldCorner& p3);
    static std::vector<std::vector<FieldCorner>> choose_subsets(const std::vector<FieldCorner>& corners, int num_subsets = 5);
    static void generate_permutations_recursive(
        const std::vector<std::vector<std::pair<float,float>>>& class_options,
        size_t depth,
        std::vector<std::pair<float,float>>& current_perm,
        std::vector<std::vector<std::pair<float,float>>>& all_perms);

    double solve_subset(const std::vector<FieldCorner>& subset,
                        const std::vector<FieldCorner>& all_corners,
                        cv::Mat& best_H);

    float confidence_threshold_;
    cv::Mat homography_matrix_;
    cv::Mat inv_homography_matrix_;
    bool valid_{false};
};

} // namespace soccer_radar
