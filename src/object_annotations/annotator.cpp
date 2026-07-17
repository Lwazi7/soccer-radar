#include "object_annotations/annotator.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>

namespace soccer_radar {

void Annotator::draw_player_ellipses(cv::Mat& frame, const Detections& players) {
    for (const auto& box : players.boxes) {
        float cx = box.cx();
        float cy = box.y2;
        float rx = box.width() * 0.5f;
        float ry = box.height() * 0.15f;

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
        float cx = box.cx();
        float cy = box.y1 - 10;

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
                    0, 0, 360, cv::Scalar(COLOR_REFEREE[0], COLOR_REFEREE[1], COLOR_REFEREE[2]), 2);
    }
}

void Annotator::draw_track_labels(cv::Mat& frame, const Detections& detections) {
    for (const auto& box : detections.boxes) {
        if (box.track_id < 0) continue;

        std::string label = "#" + std::to_string(box.track_id);
        int base_line = 0;
        cv::Size size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base_line);

        int x = static_cast<int>(box.cx() - size.width * 0.5f);
        int y = static_cast<int>(box.y1 - 5);

        cv::rectangle(frame,
                      cv::Point(x - 2, y - size.height - 2),
                      cv::Point(x + size.width + 2, y + 2),
                      cv::Scalar(0, 0, 0), -1);

        cv::putText(frame, label, cv::Point(x, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
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

    for (const auto& c : kpts.corners) {
        if (c.confidence < confidence_threshold) continue;

        int x = static_cast<int>(c.x);
        int y = static_cast<int>(c.y);

        cv::circle(frame, cv::Point(x, y), 5, cv::Scalar(0, 255, 255), -1);

        std::string label = "cls: " + std::to_string(c.class_id);
        cv::putText(frame, label, cv::Point(x + 8, y - 8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 255), 1);
    }
}

Detections Annotator::tracks_to_player_detections(const FrameTracks& ft) {
    Detections dets;
    for (size_t i = 0; i < ft.players.size(); ++i) {
        BBox b = ft.players[i].second;
        b.track_id = ft.players[i].first;
        if (i < ft.player_teams.size()) {
            b.class_id = ft.player_teams[i].second;
        }
        dets.add(b);
    }
    return dets;
}

Detections Annotator::tracks_to_ball_detections(const FrameTracks& ft) {
    Detections dets;
    if (ft.ball.valid()) {
        dets.add(ft.ball);
    }
    return dets;
}

Detections Annotator::tracks_to_referee_detections(const FrameTracks& ft) {
    Detections dets;
    for (const auto& [id, box] : ft.referees) {
        BBox b = box;
        b.track_id = id;
        dets.add(b);
    }
    return dets;
}

} // namespace soccer_radar
