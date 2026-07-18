#include "pipelines/tactical_pipeline.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>

namespace soccer_radar {

TacticalPipeline::TacticalPipeline() {}

cv::Mat TacticalPipeline::process_frame(const Detections& players,
                                          const Detections& balls,
                                          const Detections& referees,
                                          const KeypointData& keypoints,
                                          TacticalMetadata& metadata) {
    // Compute homography from keypoints using Football-TV2Radar combinatorial RANSAC solver
    bool valid = homography_.compute(keypoints);
    metadata.transform_valid = valid;

    // Separate players by team (class_id after stable team clustering)
    Detections team1_dets = players.filter(0);
    Detections team2_dets = players.filter(1);

    metadata.num_team1 = team1_dets.size();
    metadata.num_team2 = team2_dets.size();
    metadata.num_players = metadata.num_team1 + metadata.num_team2;
    metadata.num_balls = balls.size();
    metadata.num_referees = referees.size();

    // Transform positions to pitch coordinates (meters: [-52.5, 52.5] x [-34, 34]).
    // For players and referees, we project bottom_centers() (cx, y2) representing their feet on the grass plane (Z=0)
    // exactly like Football-TV2Radar (cy + h / 2 = y2). Projecting torso/head centers (cx, cy) off Z=0 causes jumping.
    std::vector<std::array<float,2>> team1_pts, team2_pts, ball_pts, ref_pts;

    if (valid) {
        team1_pts = homography_.transform_to_pitch(team1_dets.bottom_centers());
        team2_pts = homography_.transform_to_pitch(team2_dets.bottom_centers());
        ball_pts = homography_.transform_to_pitch(balls.centers());
        ref_pts = homography_.transform_to_pitch(referees.bottom_centers());
    }

    // Draw tactical view
    cv::Mat tactical = HomographyTransformer::draw_pitch(PITCH_DRAW_WIDTH, PITCH_DRAW_HEIGHT);
    HomographyTransformer::draw_positions_on_pitch(tactical, team1_pts, team2_pts,
                                                     ball_pts, ref_pts);

    return tactical;
}

cv::Mat TacticalPipeline::create_overlay(const cv::Mat& original,
                                           const cv::Mat& tactical,
                                           int overlay_w,
                                           int overlay_h) {
    cv::Mat combined = original.clone();

    cv::Mat resized;
    cv::resize(tactical, resized, cv::Size(overlay_w, overlay_h));

    int h = combined.rows;
    int w = combined.cols;
    int margin = 10;

    int start_y = margin;
    int end_y = margin + overlay_h;
    int start_x = w - overlay_w - margin;
    int end_x = w - margin;

    if (end_y > h) end_y = h;
    if (end_x > w) end_x = w;
    if (start_y < 0) start_y = 0;
    if (start_x < 0) start_x = 0;

    int actual_h = end_y - start_y;
    int actual_w = end_x - start_x;

    if (actual_h > 0 && actual_w > 0) {
        cv::Mat roi = combined(cv::Rect(start_x, start_y, actual_w, actual_h));
        cv::Mat resized_roi = resized(cv::Rect(0, 0,
            std::min(actual_w, resized.cols),
            std::min(actual_h, resized.rows)));
        resized_roi.copyTo(roi(cv::Rect(0, 0, resized_roi.cols, resized_roi.rows)));

        cv::rectangle(combined, cv::Point(start_x - 2, start_y - 2),
                     cv::Point(end_x + 2, end_y + 2),
                     cv::Scalar(255, 255, 255), 2);

        cv::putText(combined, "Tactical View",
                   cv::Point(start_x, start_y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5,
                   cv::Scalar(255, 255, 255), 1);
    }

    return combined;
}

} // namespace soccer_radar
