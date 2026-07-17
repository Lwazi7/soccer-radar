#include "tactical_analysis/homography.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include <limits>
#include <map>

namespace soccer_radar {

// Football-TV2Radar 13 class reference coordinates in METERS: [-52.5, 52.5] x [-34, 34]
static const std::unordered_map<int, std::vector<std::pair<float, float>>>& get_corner_reference_points() {
    static const std::unordered_map<int, std::vector<std::pair<float, float>>> points = {
        {4,  {{0.0f, 0.0f}}},                                                              // Center point
        {3,  {{-41.5f, 0.0f}, {41.5f, 0.0f}}},                                             // Penalty spots
        {2,  {{-52.5f, -3.66f}, {-52.5f, 3.66f}, {52.5f, -3.66f}, {52.5f, 3.66f}}},          // Goal posts / goal area inner
        {12, {{-32.35f, 0.0f}, {32.35f, 0.0f}}},                                           // Center circle outer points
        {11, {{-52.5f, -9.15f}, {-52.5f, 9.15f}, {52.5f, -9.15f}, {52.5f, 9.15f}}},          // Goal area outer corners
        {10, {{-9.15f, 0.0f}, {9.15f, 0.0f}}},                                             // Center circle side points
        {9,  {{-52.5f, -20.15f}, {-52.5f, 20.15f}, {52.5f, -20.15f}, {52.5f, 20.15f}}},      // Penalty box outer goal line corners
        {8,  {{0.0f, -34.0f}, {0.0f, 34.0f}}},                                             // Center line / touch line intersections
        {7,  {{-52.5f, -34.0f}, {-52.5f, 34.0f}, {52.5f, -34.0f}, {52.5f, 34.0f}}},          // 4 main corner flags of the pitch
        {6,  {{-47.0f, -9.15f}, {-47.0f, 9.15f}, {47.0f, -9.15f}, {47.0f, 9.15f}}},          // Goal area top corners
        {5,  {{-36.0f, -7.32f}, {-36.0f, 7.32f}, {36.0f, -7.32f}, {36.0f, 7.32f}}},          // Penalty arc junctions
        {1,  {{0.0f, -9.15f}, {0.0f, 9.15f}}},                                             // Center circle top/bottom points
        {0,  {{-36.0f, -20.15f}, {-36.0f, 20.15f}, {36.0f, -20.15f}, {36.0f, 20.15f}}}       // Penalty box top corners
    };
    return points;
}

HomographyTransformer::HomographyTransformer(float confidence_threshold)
    : confidence_threshold_(confidence_threshold) {
}

bool HomographyTransformer::are_collinear(const FieldCorner& p1, const FieldCorner& p2, const FieldCorner& p3) {
    float v1x = p2.x - p1.x;
    float v1y = p2.y - p1.y;
    float v2x = p3.x - p1.x;
    float v2y = p3.y - p1.y;
    float cross = v1x * v2y - v1y * v2x;
    return std::abs(cross) < 1e-4f;
}

std::vector<std::vector<FieldCorner>> HomographyTransformer::choose_subsets(
    const std::vector<FieldCorner>& corners, int num_subsets) {

    std::vector<std::vector<FieldCorner>> subsets;
    if (corners.size() < 4) return subsets;

    if (corners.size() == 4) {
        if (!are_collinear(corners[0], corners[1], corners[2]) &&
            !are_collinear(corners[0], corners[1], corners[3]) &&
            !are_collinear(corners[1], corners[2], corners[3])) {
            subsets.push_back(corners);
        }
        return subsets;
    }

    std::mt19937 rng(42);
    int attempt = 0;
    while (static_cast<int>(subsets.size()) < num_subsets && attempt < 2000) {
        attempt++;
        std::vector<int> indices(corners.size());
        for (size_t i = 0; i < corners.size(); ++i) indices[i] = static_cast<int>(i);
        std::shuffle(indices.begin(), indices.end(), rng);

        std::vector<FieldCorner> sel = { corners[indices[0]], corners[indices[1]], corners[indices[2]], corners[indices[3]] };

        bool collinear = false;
        if (are_collinear(sel[0], sel[1], sel[2]) || are_collinear(sel[0], sel[1], sel[3]) ||
            are_collinear(sel[0], sel[2], sel[3]) || are_collinear(sel[1], sel[2], sel[3])) {
            collinear = true;
        }

        if (!collinear) {
            subsets.push_back(std::move(sel));
        }
    }
    return subsets;
}

void HomographyTransformer::generate_permutations_recursive(
    const std::vector<std::vector<std::pair<float,float>>>& class_options,
    size_t depth,
    std::vector<std::pair<float,float>>& current_perm,
    std::vector<std::vector<std::pair<float,float>>>& all_perms) {

    if (depth == class_options.size()) {
        all_perms.push_back(current_perm);
        return;
    }

    for (const auto& pt : class_options[depth]) {
        // Ensure exact same physical reference coordinate isn't reused inside the same 4-point subset
        bool reused = false;
        for (size_t i = 0; i < depth; ++i) {
            if (std::abs(current_perm[i].first - pt.first) < 0.1f &&
                std::abs(current_perm[i].second - pt.second) < 0.1f) {
                reused = true;
                break;
            }
        }
        if (!reused) {
            current_perm[depth] = pt;
            generate_permutations_recursive(class_options, depth + 1, current_perm, all_perms);
        }
    }
}

double HomographyTransformer::solve_subset(const std::vector<FieldCorner>& subset,
                                           const std::vector<FieldCorner>& all_corners,
                                           cv::Mat& best_H) {
    const auto& ref_map = get_corner_reference_points();

    std::vector<cv::Point2f> src_4pts(4);
    std::vector<std::vector<std::pair<float,float>>> class_options(4);

    for (int i = 0; i < 4; ++i) {
        src_4pts[i] = cv::Point2f(subset[i].x, subset[i].y);
        auto it = ref_map.find(subset[i].class_id);
        if (it != ref_map.end()) {
            class_options[i] = it->second;
        } else {
            return -1e18; // Unknown class
        }
    }

    std::vector<std::vector<std::pair<float,float>>> all_perms;
    std::vector<std::pair<float,float>> current_perm(4);
    generate_permutations_recursive(class_options, 0, current_perm, all_perms);

    if (all_perms.empty()) return -1e18;

    double max_score = -1e18;
    bool found_any = false;

    // Prepare all detected points for fast scoring projection
    std::vector<cv::Point2f> whole_src;
    std::vector<int> whole_cls;
    whole_src.reserve(all_corners.size());
    whole_cls.reserve(all_corners.size());
    for (const auto& c : all_corners) {
        if (c.confidence >= confidence_threshold_) {
            whole_src.emplace_back(c.x, c.y);
            whole_cls.push_back(c.class_id);
        }
    }

    for (const auto& perm : all_perms) {
        std::vector<cv::Point2f> dst_4pts(4);
        for (int i = 0; i < 4; ++i) {
            dst_4pts[i] = cv::Point2f(perm[i].first, perm[i].second);
        }

        cv::Mat H = cv::findHomography(src_4pts, dst_4pts, 0);
        if (H.empty()) continue;

        std::vector<cv::Point2f> projected;
        cv::perspectiveTransform(whole_src, projected, H);

        double score = 0.0;
        for (size_t i = 0; i < projected.size(); ++i) {
            auto it = ref_map.find(whole_cls[i]);
            if (it == ref_map.end()) continue;

            double min_dist_sq = 1e18;
            for (const auto& ref_pt : it->second) {
                double dx = static_cast<double>(projected[i].x - ref_pt.first);
                double dy = static_cast<double>(projected[i].y - ref_pt.second);
                double dist_sq = dx * dx + dy * dy;
                if (dist_sq < min_dist_sq) min_dist_sq = dist_sq;
            }
            score -= min_dist_sq; // Lower squared error across all detected corners = higher score
        }

        if (score > max_score) {
            max_score = score;
            best_H = H.clone();
            found_any = true;
        }
    }

    return found_any ? max_score : -1e18;
}

bool HomographyTransformer::compute(const KeypointData& keypoints) {
    valid_ = false;
    if (keypoints.num_corners() < MIN_CORNERS_FOR_HOMOGRAPHY) return false;

    std::vector<FieldCorner> valid_corners;
    for (const auto& c : keypoints.corners) {
        if (c.confidence >= confidence_threshold_) {
            valid_corners.push_back(c);
        }
    }
    if (valid_corners.size() < static_cast<size_t>(MIN_CORNERS_FOR_HOMOGRAPHY)) return false;

    auto subsets = choose_subsets(valid_corners, 5);
    if (subsets.empty()) return false;

    double global_max_score = -1e18;
    cv::Mat best_global_H;

    for (const auto& sub : subsets) {
        cv::Mat H;
        double s = solve_subset(sub, valid_corners, H);
        if (s > global_max_score && !H.empty()) {
            global_max_score = s;
            best_global_H = H;
        }
    }

    if (best_global_H.empty() || global_max_score <= -1e17) return false;

    homography_matrix_ = best_global_H.clone();
    inv_homography_matrix_ = homography_matrix_.inv();
    valid_ = true;
    return true;
}

std::vector<std::array<float,2>> HomographyTransformer::transform_to_pitch(
    const std::vector<std::array<float,2>>& frame_points) const {

    std::vector<std::array<float,2>> result;
    if (!valid_ || frame_points.empty()) return result;

    std::vector<cv::Point2f> src_pts;
    src_pts.reserve(frame_points.size());
    for (const auto& pt : frame_points) {
        src_pts.emplace_back(pt[0], pt[1]);
    }

    std::vector<cv::Point2f> dst_pts;
    cv::perspectiveTransform(src_pts, dst_pts, homography_matrix_);

    result.reserve(dst_pts.size());
    for (const auto& pt : dst_pts) {
        result.push_back({pt.x, pt.y});
    }
    return result;
}

std::vector<std::array<float,2>> HomographyTransformer::transform_to_frame(
    const std::vector<std::array<float,2>>& pitch_points) const {

    std::vector<std::array<float,2>> result;
    if (!valid_ || pitch_points.empty()) return result;

    std::vector<cv::Point2f> src_pts;
    src_pts.reserve(pitch_points.size());
    for (const auto& pt : pitch_points) {
        src_pts.emplace_back(pt[0], pt[1]);
    }

    std::vector<cv::Point2f> dst_pts;
    cv::perspectiveTransform(src_pts, dst_pts, inv_homography_matrix_);

    result.reserve(dst_pts.size());
    for (const auto& pt : dst_pts) {
        result.push_back({pt.x, pt.y});
    }
    return result;
}

cv::Mat HomographyTransformer::draw_pitch(int width, int height) {
    cv::Mat pitch(height, width, CV_8UC3, cv::Scalar(34, 139, 34)); // Forest green

    // Football-TV2Radar coordinate space: X in [-52.5, 52.5] meters (105m), Y in [-34, 34] meters (68m)
    const float sx = static_cast<float>(width) / 105.0f;
    const float sy = static_cast<float>(height) / 68.0f;

    cv::Scalar line_color(255, 255, 255);
    int thickness = 2;

    auto px = [&](float x_meters) -> int { return static_cast<int>((x_meters + 52.5f) * sx); };
    auto py = [&](float y_meters) -> int { return static_cast<int>((y_meters + 34.0f) * sy); };

    // Outer boundary
    cv::rectangle(pitch, cv::Point(px(-52.5f), py(-34.0f)), cv::Point(px(52.5f), py(34.0f)), line_color, thickness);

    // Center line
    cv::line(pitch, cv::Point(px(0.0f), py(-34.0f)), cv::Point(px(0.0f), py(34.0f)), line_color, thickness);

    // Center circle (radius = 9.15m)
    int circle_rx = static_cast<int>(9.15f * sx);
    int circle_ry = static_cast<int>(9.15f * sy);
    cv::ellipse(pitch, cv::Point(px(0.0f), py(0.0f)), cv::Size(circle_rx, circle_ry), 0, 0, 360, line_color, thickness);
    cv::circle(pitch, cv::Point(px(0.0f), py(0.0f)), 3, line_color, -1);

    // Left penalty box: X in [-52.5, -36], Y in [-20.15, 20.15]
    cv::rectangle(pitch, cv::Point(px(-52.5f), py(-20.15f)), cv::Point(px(-36.0f), py(20.15f)), line_color, thickness);
    // Left goal area: X in [-52.5, -47], Y in [-9.15, 9.15]
    cv::rectangle(pitch, cv::Point(px(-52.5f), py(-9.15f)), cv::Point(px(-47.0f), py(9.15f)), line_color, thickness);
    // Left penalty spot (-41.5m, 0)
    cv::circle(pitch, cv::Point(px(-41.5f), py(0.0f)), 3, line_color, -1);

    // Right penalty box: X in [36, 52.5], Y in [-20.15, 20.15]
    cv::rectangle(pitch, cv::Point(px(36.0f), py(-20.15f)), cv::Point(px(52.5f), py(20.15f)), line_color, thickness);
    // Right goal area: X in [47, 52.5], Y in [-9.15, 9.15]
    cv::rectangle(pitch, cv::Point(px(47.0f), py(-9.15f)), cv::Point(px(52.5f), py(9.15f)), line_color, thickness);
    // Right penalty spot (41.5m, 0)
    cv::circle(pitch, cv::Point(px(41.5f), py(0.0f)), 3, line_color, -1);

    return pitch;
}

void HomographyTransformer::draw_positions_on_pitch(
    cv::Mat& pitch,
    const std::vector<std::array<float,2>>& team1_pts,
    const std::vector<std::array<float,2>>& team2_pts,
    const std::vector<std::array<float,2>>& ball_pts,
    const std::vector<std::array<float,2>>& ref_pts) {

    const float sx = static_cast<float>(pitch.cols) / 105.0f;
    const float sy = static_cast<float>(pitch.rows) / 68.0f;

    auto px = [&](float x_meters) -> int { return static_cast<int>((x_meters + 52.5f) * sx); };
    auto py = [&](float y_meters) -> int { return static_cast<int>((y_meters + 34.0f) * sy); };

    for (const auto& pt : team1_pts) {
        if (std::isnan(pt[0]) || std::isnan(pt[1])) continue;
        int x = px(pt[0]);
        int y = py(pt[1]);
        cv::circle(pitch, cv::Point(x, y), 8, cv::Scalar(COLOR_TEAM1[0], COLOR_TEAM1[1], COLOR_TEAM1[2]), -1);
    }

    for (const auto& pt : team2_pts) {
        if (std::isnan(pt[0]) || std::isnan(pt[1])) continue;
        int x = px(pt[0]);
        int y = py(pt[1]);
        cv::circle(pitch, cv::Point(x, y), 8, cv::Scalar(COLOR_TEAM2[0], COLOR_TEAM2[1], COLOR_TEAM2[2]), -1);
    }

    for (const auto& pt : ball_pts) {
        if (std::isnan(pt[0]) || std::isnan(pt[1])) continue;
        int x = px(pt[0]);
        int y = py(pt[1]);
        cv::circle(pitch, cv::Point(x, y), 6, cv::Scalar(255, 255, 255), -1);
    }

    for (const auto& pt : ref_pts) {
        if (std::isnan(pt[0]) || std::isnan(pt[1])) continue;
        int x = px(pt[0]);
        int y = py(pt[1]);
        cv::rectangle(pitch, cv::Point(x - 6, y - 6), cv::Point(x + 6, y + 6), cv::Scalar(0, 0, 0), -1);
    }
}

} // namespace soccer_radar
