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
constexpr const char* KEYPOINT_DETECTION_MODEL  = "models/football_tv2radar.onnx";
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
// Detection Parameters (Exact Soccer_Analysis Parity)
// ============================================================================
// Ball confidence threshold lowered to 0.15 matching supervision two-tier sensitivity
constexpr float CONFIDENCE_THRESHOLD         = 0.15f;
constexpr float PLAYER_CONFIDENCE_THRESHOLD  = 0.25f; // Rejects crowd/background noise
constexpr float REFEREE_CONFIDENCE_THRESHOLD = 0.25f;

// NMS threshold set to 0.70 exact parity with PyTorch torchvision.ops.nms default.
// Prevents closely marked players from vanishing during tackles and corners.
constexpr float NMS_THRESHOLD        = 0.70f;
constexpr float KEYPOINT_CONFIDENCE  = 0.25f; // TV2Radar corner threshold (untouched)
constexpr int   NUM_CORNER_CLASSES   = 13;    // Football-TV2Radar corner classes (0..12)
constexpr int   MIN_CORNERS_FOR_HOMOGRAPHY = 4;

// ============================================================================
// Tracking Parameters (Exact supervision.ByteTrack Parity)
// ============================================================================
constexpr float TRACKER_MATCH_THRESH     = 0.35f;
constexpr float TRACKER_HIGH_CONF_THRESH = 0.25f; // Stage 1 high-confidence pool (D_high)
constexpr float TRACKER_LOW_CONF_THRESH  = 0.10f; // Stage 2 occlusion recovery pool (D_low)
constexpr int   TRACKER_BUFFER_SIZE      = 120;   // 120 frames (4 seconds) lost track retention
constexpr int   TRACKER_MAX_DETECTIONS   = 40;    // Bound assignment cost on noisy crowd shots
constexpr float REID_COSINE_THRESHOLD    = 0.35f; // Maximum cosine distance for appearance recovery
constexpr float REID_EMA_ALPHA           = 0.90f; // Track feature history smoothing
constexpr int   MAX_TRACK_ID             = 100000;

// Configurable detection stride (1 = run every frame, 2 = 50% compute reduction via Kalman prediction)
constexpr int   DETECTION_STRIDE         = 2;

// ============================================================================
// Clustering Parameters (Soccer_Analysis Parity)
// ============================================================================
constexpr int   EMBEDDING_DIM        = 1280; // MobileNetV4 Conv Small output dim
constexpr int   EMBEDDING_BATCH_SIZE = 24;   // Batched crop processing parity
constexpr int   PCA_COMPONENTS       = 3;
constexpr int   NUM_TEAMS            = 2;
constexpr int   KMEANS_MAX_ITER      = 50;
constexpr int   KMEANS_RESTARTS      = 3;
constexpr float KMEANS_EPS           = 1e-4f;
constexpr float MIN_CLUSTER_SILHOUETTE = 0.15f; // Reject visually inseparable team clusters
constexpr int   TEAM_VOTE_WINDOW       = 15;    // Per-track temporal label window

// ============================================================================
// Training Parameters
// ============================================================================
constexpr int TRAINING_FRAME_STRIDE = 12;
constexpr int TRAINING_FRAME_LIMIT  = 120 * 24; // ~2 minutes at 24fps

// ============================================================================
// Ball Interpolation (Exact Soccer_Analysis Parity)
// ============================================================================
constexpr int BALL_INTERPOLATION_LIMIT = 30; // Max gap for constant-acceleration interpolation

// ============================================================================
// Class Colors (BGR)
// ============================================================================
constexpr std::array<int, 3> COLOR_PLAYER  = {0, 255, 0};     // Green
constexpr std::array<int, 3> COLOR_BALL    = {0, 0, 255};     // Red
constexpr std::array<int, 3> COLOR_REFEREE = {255, 0, 0};     // Blue
constexpr std::array<int, 3> COLOR_TEAM1   = {128, 0, 128};   // Purple
constexpr std::array<int, 3> COLOR_TEAM2   = {0, 0, 255};     // Red

// ============================================================================
// Output Configuration
// ============================================================================
constexpr int   OUTPUT_FPS        = 30;
constexpr int   OVERLAY_WIDTH     = 500;
constexpr int   OVERLAY_HEIGHT    = 350;
constexpr int   PITCH_DRAW_WIDTH  = 1050;
constexpr int   PITCH_DRAW_HEIGHT = 680;

} // namespace soccer_radar
