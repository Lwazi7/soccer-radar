#pragma once

#include "utils/types.hpp"
#include "utils/constants.hpp"
#include <vector>
#include <unordered_map>

namespace soccer_radar {

// True three-stage ByteTrack implementation optimized for constrained devices.
class ByteTracker {
public:
    ByteTracker(float match_thresh = 0.35f,
                int buffer_size = TRACKER_BUFFER_SIZE);
    ~ByteTracker() = default;

    // Update tracker with new detections; returns detections with track_ids assigned
    Detections update(const Detections& detections);

    // Update internal active/lost tracks with team labels from clustering cache
    void update_team_labels(const std::unordered_map<int, int>& team_cache);

    void reset();

private:
    void kalman_predict(TrackState& track);
    void kalman_update(TrackState& track, const BBox& measurement);
    void kalman_init(TrackState& track, const BBox& bbox);

    static float compute_iou(const BBox& a, const BBox& b);

    std::vector<std::pair<int,int>> match_tracks_to_detections(
        const std::vector<TrackState>& tracks,
        const std::vector<BBox>& detections,
        float thresh);

    float match_thresh_;
    int   buffer_size_;
    int   next_track_id_{1};

    std::vector<TrackState> active_tracks_;
    std::vector<TrackState> lost_tracks_;
};

} // namespace soccer_radar
