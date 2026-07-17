#pragma once

#include "utils/types.hpp"
#include "utils/constants.hpp"
#include <opencv2/core.hpp>
#include <vector>

namespace soccer_radar {

// Annotation utilities for drawing detections, tracks, teams, and keypoints.
// All drawing is done directly with OpenCV primitives -- no external dependencies.
class Annotator {
public:
    Annotator() = default;
    ~Annotator() = default;

    // Draw ellipses around players (like the original's EllipseAnnotator)
    void draw_player_ellipses(cv::Mat& frame, const Detections& players);

    // Draw triangles above ball detections
    void draw_ball_triangles(cv::Mat& frame, const Detections& balls);

    // Draw referee ellipses
    void draw_referee_ellipses(cv::Mat& frame, const Detections& referees);

    // Draw track ID labels
    void draw_track_labels(cv::Mat& frame, const Detections& detections);

    // Draw all annotations
    void draw_all(cv::Mat& frame,
                  const Detections& players,
                  const Detections& balls,
                  const Detections& referees);

    // Draw keypoints with connections
    void draw_keypoints(cv::Mat& frame, const KeypointData& kpts,
                        float confidence_threshold = KEYPOINT_CONFIDENCE);

    // Convert FrameTracks back to Detections for annotation
    static Detections tracks_to_player_detections(const FrameTracks& ft);
    static Detections tracks_to_ball_detections(const FrameTracks& ft);
    static Detections tracks_to_referee_detections(const FrameTracks& ft);
};

} // namespace soccer_radar
