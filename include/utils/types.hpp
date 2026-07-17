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

    // Get center points
    std::vector<std::array<float,2>> centers() const {
        std::vector<std::array<float,2>> pts;
        pts.reserve(boxes.size());
        for (auto& b : boxes) {
            pts.push_back({b.cx(), b.cy()});
        }
        return pts;
    }
};

// ============================================================================
// Keypoint Data (per detection)
// ============================================================================
struct KeypointData {
    // Shape: [num_detections][NUM_KEYPOINTS][3] stored flat
    // But typically 1 detection per frame for field keypoints
    std::vector<std::array<float, 3>> points; // x, y, confidence

    int num_keypoints() const { return static_cast<int>(points.size()); }
    bool empty() const { return points.empty(); }
    void clear() { points.clear(); }
};

// ============================================================================
// Track State for ByteTrack
// ============================================================================
struct TrackState {
    int    track_id{-1};
    BBox   bbox;
    int    frames_since_seen{0};
    int    total_hits{0};
    int    team_label{-1};

    // Kalman state: [cx, cy, w, h, vx, vy, vw, vh]
    std::array<float, 8> state{};
    std::array<float, 64> covariance{}; // 8x8 flattened
};

// ============================================================================
// Per-frame track storage
// ============================================================================
struct FrameTracks {
    // player: track_id -> bbox
    std::vector<std::pair<int, BBox>> players;
    // player: track_id -> team_label (class_id after clustering)
    std::vector<std::pair<int, int>>  player_teams;
    // ball: single bbox (may be invalid)
    BBox ball;
    // referees: id -> bbox
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
};

} // namespace soccer_radar
