#pragma once

#include <cstdint>
#include <vector>
#include <array>
#include <cmath>

namespace soccer_radar {

// ============================================================================
// Bounding Box
// ============================================================================
struct BBox {
    float x1{0}, y1{0}, x2{0}, y2{0};
    float confidence{0};
    int   class_id{0};
    int   track_id{-1};

    float width()  const { return x2 - x1; }
    float height() const { return y2 - y1; }
    float area()   const { return width() * height(); }
    float cx()     const { return (x1 + x2) * 0.5f; }
    float cy()     const { return (y1 + y2) * 0.5f; }

    bool valid() const { return x2 > x1 && y2 > y1; }
};

// ============================================================================
// Detection Set: lightweight container for a frame's detections
// ============================================================================
struct Detections {
    std::vector<BBox> boxes;

    int size() const { return static_cast<int>(boxes.size()); }
    bool empty() const { return boxes.empty(); }
    void clear() { boxes.clear(); }

    void add(const BBox& b) { boxes.push_back(b); }

    // Filter by class_id
    Detections filter(int cls) const {
        Detections out;
        for (auto& b : boxes) {
            if (b.class_id == cls) out.boxes.push_back(b);
        }
        return out;
    }

    // Get center points (cx, cy)
    std::vector<std::array<float,2>> centers() const {
        std::vector<std::array<float,2>> pts;
        pts.reserve(boxes.size());
        for (auto& b : boxes) {
            pts.push_back({b.cx(), b.cy()});
        }
        return pts;
    }

    // Get bottom-center points (cx, y2) representing the feet on the grass plane (Z=0).
    // Essential for accurate homography projection without head/torso height warping.
    std::vector<std::array<float,2>> bottom_centers() const {
        std::vector<std::array<float,2>> pts;
        pts.reserve(boxes.size());
        for (auto& b : boxes) {
            pts.push_back({b.cx(), b.y2});
        }
        return pts;
    }
};

// ============================================================================
// Timing Diagnostics
// ============================================================================
struct DetectionTiming {
    double preprocess_ms{0.0};
    double onnx_run_ms{0.0};
    double postprocess_ms{0.0};
};

struct TrackingTiming {
    double det_preprocess_ms{0.0};
    double det_onnx_run_ms{0.0};
    double det_postprocess_ms{0.0};
    double tracker_update_ms{0.0};
    double clustering_ms{0.0};
    bool   ran_yolo{false};
};

// ============================================================================
// Corner / Line Intersection Data (Football-TV2Radar representation)
// ============================================================================
struct FieldCorner {
    float x{0}, y{0};      // Center coordinate of detected corner bounding box
    float confidence{0};   // Detection confidence
    int   class_id{0};     // Corner class index (0..12)
};

struct KeypointData {
    std::vector<FieldCorner> corners;

    int num_corners() const { return static_cast<int>(corners.size()); }
    bool empty() const { return corners.empty(); }
    void clear() { corners.clear(); }
};

// ============================================================================
// Track State for ByteTrack (With Temporal Majority Voting Identity Lock)
// ============================================================================
struct TrackState {
    int    track_id{-1};
    BBox   bbox;
    int    frames_since_seen{0};
    int    total_hits{0};
    int    team_label{-1};

    // Temporal voting history across all detections for this track
    int    player_vote_count{0};
    int    referee_vote_count{0};
    int    team0_vote_count{0};
    int    team1_vote_count{0};

    // Kalman state: [cx, cy, w, h, vx, vy, vw, vh]
    std::array<float, 8> state{};
    std::array<float, 64> covariance{};
};

// ============================================================================
// Per-frame track storage
// ============================================================================
struct FrameTracks {
    std::vector<std::pair<int, BBox>> players;
    std::vector<std::pair<int, int>>  player_teams;
    BBox ball;
    std::vector<std::pair<int, BBox>> referees;
};

// ============================================================================
// Metadata for tactical analysis
// ============================================================================
struct TacticalMetadata {
    int  num_players{0};
    int  num_team1{0};
    int  num_team2{0};
    int  num_balls{0};
    int  num_referees{0};
    bool transform_valid{false};
    bool is_replay{false};
};

} // namespace soccer_radar
