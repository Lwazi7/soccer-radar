#include "tactical_analysis/homography.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <iostream>

namespace soccer_radar {

HomographyTransformer::HomographyTransformer(float confidence_threshold)
    : confidence_threshold_(confidence_threshold) {
    all_pitch_points_ = get_all_pitch_points();
}

bool HomographyTransformer::compute(const KeypointData& keypoints) {
    valid_ = false;
    if (keypoints.empty()) return false;

    const auto& mapping = get_keypoint_to_pitch_mapping();

    std::vector<cv::Point2f> frame_points;
    std::vector<cv::Point2f> pitch_points;

    for (int i = 0; i < keypoints.num_keypoints() && i < NUM_KEYPOINTS; ++i) {
        const auto& kpt = keypoints.points[i];

        if (kpt[2] > confidence_threshold_) {
            frame_points.emplace_back(kpt[0], kpt[1]);

            int pitch_idx = mapping[i];
            if (pitch_idx < static_cast<int>(all_pitch_points_.size())) {
                pitch_points.emplace_back(all_pitch_points_[pitch_idx].first,
                                          all_pitch_points_[pitch_idx].second);
            }
        }
    }

    if (static_cast<int>(frame_points.size()) < MIN_KEYPOINTS_FOR_HOMOGRAPHY) {
        return false;
    }

    // Compute homography using RANSAC for robustness against noisy keypoints
    homography_matrix_ = cv::findHomography(frame_points, pitch_points,
                                             cv::RANSAC, 3.0);

    if (homography_matrix_.empty()) {
        return false;
    }

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
    cv::Mat pitch(height, width, CV_8UC3, cv::Scalar(34, 139, 34));

    const float sx = static_cast<float>(width) / 12000.0f;
    const float sy = static_cast<float>(height) / 7000.0f;

    cv::Scalar line_color(255, 255, 255);
    int thickness = 2;

    auto px = [&](float x) -> int { return static_cast<int>(x * sx); };
    auto py = [&](float y) -> int { return static_cast<int>(y * sy); };

    cv::rectangle(pitch, cv::Point(0, 0), cv::Point(width - 1, height - 1),
                  line_color, thickness);

    cv::line(pitch, cv::Point(px(6000), 0), cv::Point(px(6000), height),
             line_color, thickness);

    int circle_rx = static_cast<int>(915 * sx);
    int circle_ry = static_cast<int>(915 * sy);
    cv::ellipse(pitch, cv::Point(px(6000), py(3500)),
                cv::Size(circle_rx, circle_ry), 0, 0, 360, line_color, thickness);

    cv::circle(pitch, cv::Point(px(6000), py(3500)), 3, line_color, -1);

    cv::rectangle(pitch, cv::Point(px(0), py(1450)),
                  cv::Point(px(2015), py(5550)), line_color, thickness);

    cv::rectangle(pitch, cv::Point(px(0), py(2584)),
                  cv::Point(px(550), py(4416)), line_color, thickness);

    cv::rectangle(pitch, cv::Point(px(9985), py(1450)),
                  cv::Point(px(12000), py(5550)), line_color, thickness);

    cv::rectangle(pitch, cv::Point(px(11450), py(2584)),
                  cv::Point(px(12000), py(4416)), line_color, thickness);

    cv::circle(pitch, cv::Point(px(1100), py(3500)), 3, line_color, -1);
    cv::circle(pitch, cv::Point(px(10900), py(3500)), 3, line_color, -1);

    return pitch;
}

void HomographyTransformer::draw_positions_on_pitch(
    cv::Mat& pitch,
    const std::vector<std::array<float,2>>& team1_pts,
    const std::vector<std::array<float,2>>& team2_pts,
    const std::vector<std::array<float,2>>& ball_pts,
    const std::vector<std::array<float,2>>& ref_pts) {

    const float sx = static_cast<float>(pitch.cols) / 12000.0f;
    const float sy = static_cast<float>(pitch.rows) / 7000.0f;

    for (const auto& pt : team1_pts) {
        if (std::isnan(pt[0]) || std::isnan(pt[1])) continue;
        int x = static_cast<int>(pt[0] * sx);
        int y = static_cast<int>(pt[1] * sy);
        cv::circle(pitch, cv::Point(x, y), 8,
                  cv::Scalar(COLOR_TEAM1[0], COLOR_TEAM1[1], COLOR_TEAM1[2]), -1);
    }

    for (const auto& pt : team2_pts) {
        if (std::isnan(pt[0]) || std::isnan(pt[1])) continue;
        int x = static_cast<int>(pt[0] * sx);
        int y = static_cast<int>(pt[1] * sy);
        cv::circle(pitch, cv::Point(x, y), 8,
                  cv::Scalar(COLOR_TEAM2[0], COLOR_TEAM2[1], COLOR_TEAM2[2]), -1);
    }

    for (const auto& pt : ball_pts) {
        if (std::isnan(pt[0]) || std::isnan(pt[1])) continue;
        int x = static_cast<int>(pt[0] * sx);
        int y = static_cast<int>(pt[1] * sy);
        cv::circle(pitch, cv::Point(x, y), 6, cv::Scalar(255, 255, 255), -1);
    }

    for (const auto& pt : ref_pts) {
        if (std::isnan(pt[0]) || std::isnan(pt[1])) continue;
        int x = static_cast<int>(pt[0] * sx);
        int y = static_cast<int>(pt[1] * sy);
        cv::rectangle(pitch, cv::Point(x - 6, y - 6), cv::Point(x + 6, y + 6),
                     cv::Scalar(0, 0, 0), -1);
    }
}

} // namespace soccer_radar
