#pragma once

#include "utils/types.hpp"
#include "utils/constants.hpp"
#include <vector>
#include <unordered_map>

namespace soccer_radar {

class ByteTracker {
public:
    ByteTracker(float match_thresh = TRACKER_MATCH_THRESH,
                int buffer_size = TRACKER_BUFFER_SIZE);
    ~ByteTracker() = default;

    Detections update(const Detections& detections, bool is_predicted_frame = false);
    void update_team_labels(const std::unordered_map<int, int>& team_cache);
    void update_team_votes(int track_id, int predicted_team);
    void reset();

private:
    void kalman_predict(TrackState& track);
    void kalman_update(TrackState& track, const BBox& measurement);
    void kalman_init(TrackState& track, const BBox& bbox);

    static float compute_iou(const BBox& a, const BBox& b);

    // Hungarian Algorithm (optimal bipartite assignment) across IoU cost matrices
    std::vector<std::pair<int,int>> match_tracks_to_detections(
        const std::vector<TrackState>& tracks,
        const std::vector<BBox>& detections,
        float thresh);

    static std::vector<std::pair<int,int>> solve_hungarian(
        const std::vector<std::vector<float>>& cost_matrix,
        float max_cost);

    float match_thresh_;
    int   buffer_size_;
    int   next_track_id_{1};

    std::vector<TrackState> active_tracks_;
    std::vector<TrackState> lost_tracks_;
};

} // namespace soccer_radar
