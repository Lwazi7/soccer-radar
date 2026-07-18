#pragma once

#include "pipelines/detection_pipeline.hpp"
#include "object_tracking/tracker.hpp"
#include "player_clustering/clustering.hpp"
#include "object_annotations/annotator.hpp"
#include "utils/types.hpp"
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <unordered_map>

namespace soccer_radar {

class TrackingPipeline {
public:
    TrackingPipeline();
    ~TrackingPipeline() = default;

    bool initialize(const std::string& model_path,
                    const std::string& embedding_model_path);

    void train_team_assignment(const std::string& video_path, int max_frames = -1);

    void process_frame(const cv::Mat& frame,
                       Detections& players,
                       Detections& balls,
                       Detections& referees,
                       TrackingTiming* timing = nullptr);

    // Stores active detections and teams specifically for the current frame_idx
    void store_tracks(const Detections& players,
                      const Detections& balls,
                      const Detections& referees,
                      int frame_idx,
                      std::unordered_map<int, BBox>& player_tracks_for_frame,
                      std::unordered_map<int, int>& player_teams_for_frame,
                      std::unordered_map<int, BBox>& ball_tracks_map,
                      std::unordered_map<int, BBox>& referee_tracks_for_frame);

    void annotate_frame(cv::Mat& frame,
                        const std::unordered_map<int, BBox>& player_tracks_for_frame,
                        const std::unordered_map<int, int>& player_teams_for_frame,
                        const BBox& ball_track_for_frame,
                        const std::unordered_map<int, BBox>& referee_tracks_for_frame);

private:
    DetectionPipeline detection_pipeline_;
    ByteTracker tracker_;
    ClusteringManager clustering_;
    Annotator annotator_;

    std::unordered_map<int, int> team_cache_;
    std::unordered_map<int, int> last_validation_frame_;
    std::unordered_map<int, std::vector<float>> frame_reid_features_;
    static constexpr int REVALIDATE_INTERVAL = 60;
    int frame_idx_global_{0};

    Detections last_balls_;
    Detections last_referees_;
};

} // namespace soccer_radar
