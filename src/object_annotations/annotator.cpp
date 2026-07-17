#include "object_annotations/annotator.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>

namespace soccer_radar {

void Annotator::draw_player_ellipses(cv::Mat& frame, const Detections& players) {
    for (const auto& box : players.boxes) {
        // Draw ellipse at the bottom center of the bounding box
        float cx = box.cx();
        float cy = box.y2; // Bottom center
        float rx = box.width() * 0.5f;
        float ry = box.height() * 0.15f;

        // Color based on team (class_id after clustering) or default green
        cv::Scalar color;
        if (box.class_id == 0) {
            color = cv::Scalar(COLOR_PLAYER[0], COLOR_PLAYER[1], COLOR_PLAYER[2]);
        } else if (box.class_id == 1) {
            color = cv::Scalar(COLOR_TEAM2[0], COLOR_TEAM2[1], COLOR_TEAM2[2]);
        } else {
            color = cv::Scalar(COLOR_PLAYER[0], COLOR_PLAYER[1], COLOR_PLAYER[2]);
        }

        cv::ellipse(frame, cv::Point(static_cast<int>(cx), static_cast<int>(cy)),
                    cv::Size(static_cast<int>(rx), static_cast<int>(ry)),
                    0, 0, 360, color, 2);
    }
}

void Annotator::draw_ball_triangles(cv::Mat& frame, const Detections& balls) {
    for (const auto& box : balls.boxes) {
        // Draw triangle above the ball
        float cx = box.cx();
        float cy = box.y1 - 10; // Above the box

        cv::Point pts[3] = {
            cv::Point(static_cast<int>(cx), static_cast<int>(cy - 10)),
            cv::Point(static_cast<int>(cx - 8), static_cast<int>(cy + 5)),
            cv::Point(static_cast<int>(cx + 8), static_cast<int>(cy + 5))
        };

        cv::fillConvexPoly(frame, pts, 3,
                          cv::Scalar(COLOR_BALL[0], COLOR_BALL[1], COLOR_BALL[2]));
    }
}

void Annotator::draw_referee_ellipses(cv::Mat& frame, const Detections& referees) {
    for (const auto& box : referees.boxes) {
        float cx = box.cx();
        float cy = box.y2;
        float rx = box.width() * 0.5f;
        float ry = box.height() * 0.15f;

        cv::ellipse(frame, cv::Point(static_cast<int>(cx), static_cast<int>(cy)),
                    cv::Size(static_cast<int>(rx), static_cast<int>(ry)),
                    0, 0, 360,
                    cv::Scalar(COLOR_REFEREE[0], COLOR_REFEREE[1], COLOR_REFEREE[2]), 2);
    }
}

void Annotator::draw_track_labels(cv::Mat& frame, const Detections& detections) {
    for (const auto& box : detections.boxes) {
        if (box.track_id < 0) continue;

        std::string label = "#" + std::to_string(box.track_id);
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                              0.5, 1, &baseline);

        int x = static_cast<int>(box.x1);
        int y = static_cast<int>(box.y1) - 5;

        // Background rectangle
        cv::rectangle(frame,
                     cv::Point(x, y - text_size.height - 2),
                     cv::Point(x + text_size.width, y + 2),
                     cv::Scalar(0, 0, 0), -1);

        // Text
        cv::putText(frame, label, cv::Point(x, y),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5,
                   cv::Scalar(255, 255, 255), 1);
    }
}

void Annotator::draw_all(cv::Mat& frame,
                         const Detections& players,
                         const Detections& balls,
                         const Detections& referees) {
    draw_player_ellipses(frame, players);
    draw_track_labels(frame, players);
    draw_ball_triangles(frame, balls);
    draw_referee_ellipses(frame, referees);
}

void Annotator::draw_keypoints(cv::Mat& frame, const KeypointData& kpts,
                                float confidence_threshold) {
    if (kpts.empty()) return;

    const auto& connections = get_keypoint_connections();

    // Draw connections
    for (const auto& conn : connections) {
        int idx1 = conn.first;
        int idx2 = conn.second;

        if (idx1 >= kpts.num_keypoints() || idx2 >= kpts.num_keypoints()) continue;

        const auto& pt1 = kpts.points[idx1];
        const auto& pt2 = kpts.points[idx2];

        if (pt1[2] > confidence_threshold && pt2[2] > confidence_threshold) {
            cv::line(frame,
                    cv::Point(static_cast<int>(pt1[0]), static_cast<int>(pt1[1])),
                    cv::Point(static_cast<int>(pt2[0]), static_cast<int>(pt2[1])),
                    cv::Scalar(255, 0, 0), 2); // Blue
        }
    }

    // Draw keypoints
    for (int i = 0; i < kpts.num_keypoints(); ++i) {
        const auto& pt = kpts.points[i];
        if (pt[2] > confidence_threshold) {
            int x = static_cast<int>(pt[0]);
            int y = static_cast<int>(pt[1]);

            cv::circle(frame, cv::Point(x, y), 5, cv::Scalar(0, 255, 0), -1);

            // Label
            std::string label = std::to_string(i);
            cv::putText(frame, label, cv::Point(x + 8, y - 8),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4,
                       cv::Scalar(255, 255, 255), 1);
        }
    }
}

Detections Annotator::tracks_to_player_detections(const FrameTracks& ft) {
    Detections dets;
    for (const auto& [track_id, bbox] : ft.players) {
        BBox b = bbox;
        b.track_id = track_id;
        b.class_id = 0;

        // Find team label
        for (const auto& [tid, team] : ft.player_teams) {
            if (tid == track_id) {
                b.class_id = team;
                break;
            }
        }

        dets.add(b);
    }
    return dets;
}

Detections Annotator::tracks_to_ball_detections(const FrameTracks& ft) {
    Detections dets;
    if (ft.ball.valid()) {
        BBox b = ft.ball;
        b.class_id = 1;
        dets.add(b);
    }
    return dets;
}

Detections Annotator::tracks_to_referee_detections(const FrameTracks& ft) {
    Detections dets;
    int id = 0;
    for (const auto& [ref_id, bbox] : ft.referees) {
        BBox b = bbox;
        b.track_id = id++;
        b.class_id = 2;
        dets.add(b);
    }
    return dets;
}

} // namespace soccer_radar
