#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace soccer_radar {

// ============================================================================
// Video / Frame Configuration
// ============================================================================
constexpr int INPUT_WIDTH  = 1280;
constexpr int INPUT_HEIGHT = 720;
constexpr int MODEL_WIDTH  = 1280;
constexpr int MODEL_HEIGHT = 736;

// Letterbox padding: (736 - 720) / 2 = 8 pixels top and bottom
constexpr int LETTERBOX_PAD_TOP    = 8;
constexpr int LETTERBOX_PAD_BOTTOM = 8;
constexpr int LETTERBOX_PAD_LEFT   = 0;
constexpr int LETTERBOX_PAD_RIGHT  = 0;

// ============================================================================
// ONNX Model Paths (relative to project root)
// ============================================================================
constexpr const char* OBJECT_DETECTION_MODEL    = "models/soccana_object_720p.onnx";
constexpr const char* KEYPOINT_DETECTION_MODEL  = "models/soccana_keypoint_720p.onnx";
constexpr const char* EMBEDDING_MODEL           = "models/mobilenetv4_conv_small.onnx";

// ============================================================================
// Detection Classes
// ============================================================================
enum class ObjectClass : int {
    Player   = 0,
    Ball     = 1,
    Referee  = 2
};

constexpr int NUM_CLASSES = 3;

// ============================================================================
// Detection Parameters
// ============================================================================
constexpr float CONFIDENCE_THRESHOLD = 0.25f;
constexpr float NMS_THRESHOLD        = 0.45f;
constexpr float KEYPOINT_CONFIDENCE  = 0.5f;

// ============================================================================
// Tracking Parameters
// ============================================================================
constexpr float TRACKER_MATCH_THRESH = 0.5f;
constexpr int   TRACKER_BUFFER_SIZE  = 120;
constexpr int   MAX_TRACK_ID         = 100000;

// ============================================================================
// Clustering Parameters
// ============================================================================
constexpr int   EMBEDDING_DIM        = 1000; // MobileNetV4 Conv Small output dim
constexpr int   EMBEDDING_BATCH_SIZE = 24;
constexpr int   PCA_COMPONENTS       = 3;
constexpr int   NUM_TEAMS            = 2;
constexpr int   KMEANS_MAX_ITER      = 100;
constexpr float KMEANS_EPS           = 1e-4f;

// ============================================================================
// Training Parameters
// ============================================================================
constexpr int TRAINING_FRAME_STRIDE = 12;
constexpr int TRAINING_FRAME_LIMIT  = 120 * 24; // ~2 minutes at 24fps

// ============================================================================
// Ball Interpolation
// ============================================================================
constexpr int BALL_INTERPOLATION_LIMIT = 30;

// ============================================================================
// Keypoint Configuration (29 keypoints)
// ============================================================================
constexpr int NUM_KEYPOINTS     = 29;
constexpr int KEYPOINT_DIMS     = 3; // x, y, confidence
constexpr int MIN_KEYPOINTS_FOR_HOMOGRAPHY = 4;

// Keypoint names (for debugging / labels)
inline const char* get_keypoint_name(int idx) {
    static const char* names[] = {
        "sideline_top_left",
        "big_rect_left_top_pt1",
        "big_rect_left_top_pt2",
        "big_rect_left_bottom_pt1",
        "big_rect_left_bottom_pt2",
        "small_rect_left_top_pt1",
        "small_rect_left_top_pt2",
        "small_rect_left_bottom_pt1",
        "small_rect_left_bottom_pt2",
        "sideline_bottom_left",
        "left_semicircle_right",
        "center_line_top",
        "center_line_bottom",
        "center_circle_top",
        "center_circle_bottom",
        "field_center",
        "sideline_top_right",
        "big_rect_right_top_pt1",
        "big_rect_right_top_pt2",
        "big_rect_right_bottom_pt1",
        "big_rect_right_bottom_pt2",
        "small_rect_right_top_pt1",
        "small_rect_right_top_pt2",
        "small_rect_right_bottom_pt1",
        "small_rect_right_bottom_pt2",
        "sideline_bottom_right",
        "right_semicircle_left",
        "center_circle_left",
        "center_circle_right"
    };
    if (idx >= 0 && idx < NUM_KEYPOINTS) return names[idx];
    return "unknown";
}

// Keypoint connections for visualization
inline const std::vector<std::pair<int,int>>& get_keypoint_connections() {
    static const std::vector<std::pair<int,int>> connections = {
        {0, 16}, {0, 9}, {16, 25}, {9, 25},
        {1, 2}, {3, 4}, {1, 3}, {2, 4},
        {5, 6}, {7, 8}, {5, 7}, {6, 8},
        {17, 18}, {19, 20}, {17, 19}, {18, 20},
        {21, 22}, {23, 24}, {21, 23}, {22, 24},
        {11, 12}, {13, 14}
    };
    return connections;
}

// Mapping from our 29 keypoints to pitch reference points
inline const std::array<int, 29>& get_keypoint_to_pitch_mapping() {
    static const std::array<int, 29> mapping = {{
        0, 1, 9, 4, 12, 2, 6, 3, 7, 5,
        32, 13, 16, 14, 15, 33, 24, 25, 17, 28,
        20, 26, 22, 27, 23, 29, 34, 30, 31
    }};
    return mapping;
}

// Pitch reference points (from roboflow/sports SoccerPitchConfiguration)
// width=7000cm, length=12000cm, penalty_box=2015×4100, goal_box=550×1832
// centre_circle_radius=915, penalty_spot_distance=1100
inline const std::vector<std::pair<float,float>>& get_pitch_points() {
    // 32 standard pitch reference points (matching SoccerPitchConfiguration.vertices)
    static const std::vector<std::pair<float,float>> points = {
        {0, 0},         // 0: top-left corner
        {0, 1450},      // 1: left penalty area top
        {0, 2584},      // 2: left goal area top
        {0, 4416},      // 3: left goal area bottom
        {0, 5550},      // 4: left penalty area bottom
        {0, 7000},      // 5: bottom-left corner
        {550, 2584},    // 6: left goal area top inner
        {550, 4416},    // 7: left goal area bottom inner
        {1100, 3500},   // 8: left penalty spot
        {2015, 1450},   // 9: left penalty area top inner
        {2015, 2584},   // 10: left penalty+goal area junction top
        {2015, 4416},   // 11: left penalty+goal area junction bottom
        {2015, 5550},   // 12: left penalty area bottom inner
        {6000, 0},      // 13: center line top
        {6000, 2585},   // 14: center circle top
        {6000, 4415},   // 15: center circle bottom
        {6000, 7000},   // 16: center line bottom
        {9985, 1450},   // 17: right penalty area top inner
        {9985, 2584},   // 18: right penalty+goal area junction top
        {9985, 4416},   // 19: right penalty+goal area junction bottom
        {9985, 5550},   // 20: right penalty area bottom inner
        {10900, 3500},  // 21: right penalty spot
        {11450, 2584},  // 22: right goal area top inner
        {11450, 4416},  // 23: right goal area bottom inner
        {12000, 0},     // 24: top-right corner
        {12000, 1450},  // 25: right penalty area top
        {12000, 2584},  // 26: right goal area top
        {12000, 4416},  // 27: right goal area bottom
        {12000, 5550},  // 28: right penalty area bottom
        {12000, 7000},  // 29: bottom-right corner
        {5085, 3500},   // 30: center circle left
        {6915, 3500}    // 31: center circle right
    };
    return points;
}

// Extra pitch points (appended to standard 32, indices 32-34)
inline const std::vector<std::pair<float,float>>& get_extra_pitch_points() {
    static const std::vector<std::pair<float,float>> points = {
        {2932, 3500},  // 32: Left semicircle rightmost point
        {6000, 3500},  // 33: Center point
        {9069, 3500}   // 34: Right semicircle rightmost point
    };
    return points;
}

// Get all pitch points (standard + extra)
inline std::vector<std::pair<float,float>> get_all_pitch_points() {
    auto pts = get_pitch_points();
    auto extra = get_extra_pitch_points();
    pts.insert(pts.end(), extra.begin(), extra.end());
    return pts;
}

// ============================================================================
// Class Colors (BGR)
// ============================================================================
constexpr std::array<int, 3> COLOR_PLAYER  = {0, 255, 0};     // Green
constexpr std::array<int, 3> COLOR_BALL    = {0, 0, 255};     // Red
constexpr std::array<int, 3> COLOR_REFEREE = {255, 0, 0};     // Blue
constexpr std::array<int, 3> COLOR_TEAM1   = {128, 0, 128};   // Purple
constexpr std::array<int, 3> COLOR_TEAM2   = {0, 0, 255};     // Red

// ============================================================================
// Output
// ============================================================================
constexpr int   OUTPUT_FPS        = 30;
constexpr int   OVERLAY_WIDTH     = 500;
constexpr int   OVERLAY_HEIGHT    = 350;
constexpr int   PITCH_DRAW_WIDTH  = 1050;
constexpr int   PITCH_DRAW_HEIGHT = 680;

} // namespace soccer_radar
