#include "pipelines/tracking_pipeline.hpp"
#include "utils/video_io.hpp"
#include "utils/constants.hpp"
#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>

namespace soccer_radar {

TrackingPipeline::TrackingPipeline()
    : tracker_(TRACKER_MATCH_THRESH, TRACKER_BUFFER_SIZE) {
}

bool TrackingPipeline::initialize(const std::string& model_path,
                                   const std::string& embedding_model_path) {
    std::cout << "[TrackingPipeline] Initializing..." << std::endl;

    if (!detection_pipeline_.initialize(model_path)) {
        std::cerr << "[TrackingPipeline] Failed to load detection model" << std::endl;
        return false;
    }

    if (!clustering_.get_extractor().load_model(embedding_model_path)) {
        std::cerr << "[TrackingPipeline] Failed to load embedding model" << std::endl;
        return false;
    }

    std::cout << "[TrackingPipeline] Initialization complete" << std::endl;
    return true;
}

void TrackingPipeline::train_team_assignment(const std::string& video_path, int max_frames) {
    std::cout << "[TrackingPipeline] Training team assignment models..." << std::endl;

    VideoReader reader(video_path);
    if (!reader.is_open()) {
        std::cerr << "[TrackingPipeline] Failed to open video: " << video_path << std::endl;
        return;
    }

    std::vector<cv::Mat> all_crops;
    cv::Mat frame;
    int frame_count = 0;
    int sampled_count = 0;
    int total_video_frames = reader.total_frames();

    int upper_frame_limit = (max_frames > 0) ? max_frames : TRAINING_FRAME_LIMIT;
    if (total_video_frames > 0 && upper_frame_limit > total_video_frames) {
        upper_frame_limit = total_video_frames;
    }

    const int TARGET_CROPS = (max_frames > 0 && max_frames <= 60) ? 60 : 120;
    const int stride = (max_frames > 0 && max_frames <= 30) ? 2 : TRAINING_FRAME_STRIDE;

    std::cout << "[TrackingPipeline] Sampling frames up to frame " << upper_frame_limit
              << " with stride " << stride << "..." << std::endl;

    while (reader.read(frame) && frame_count < upper_frame_limit) {
        if (frame_count % stride == 0) {
            Detections players, balls, referees;
            detection_pipeline_.detect_frame(frame, players, balls, referees);

            auto crops = EmbeddingExtractor::get_player_crops(frame, players.boxes);
            all_crops.insert(all_crops.end(), crops.begin(), crops.end());
            sampled_count++;

            if (static_cast<int>(all_crops.size()) >= TARGET_CROPS) {
                std::cout << "  Reached target crop count (" << all_crops.size() << "), stopping early." << std::endl;
                break;
            }
        }
        frame_count++;
    }

    std::cout << "[TrackingPipeline] Collected " << all_crops.size()
              << " player crops from " << sampled_count << " sampled frames" << std::endl;

    if (all_crops.empty()) {
        std::cerr << "[TrackingPipeline] No crops collected for training" << std::endl;
        return;
    }

    if (static_cast<int>(all_crops.size()) > TARGET_CROPS) {
        std::mt19937 rng(42);
        std::shuffle(all_crops.begin(), all_crops.end(), rng);
        all_crops.resize(TARGET_CROPS);
    }

    std::cout << "[TrackingPipeline] Extracting batched embeddings from "
              << all_crops.size() << " crops..." << std::endl;

    std::vector<std::vector<float>> all_embeddings;
    all_embeddings.reserve(all_crops.size());

    const size_t batch_size = 24;
    for (size_t i = 0; i < all_crops.size(); i += batch_size) {
        size_t end = std::min(i + batch_size, all_crops.size());
        std::vector<cv::Mat> batch(all_crops.begin() + i, all_crops.begin() + end);
        auto embs = clustering_.get_extractor().extract_batch(batch);
        all_embeddings.insert(all_embeddings.end(), embs.begin(), embs.end());
    }

    std::cout << "[TrackingPipeline] Training PCA + K-means..." << std::endl;
    clustering_.train(all_embeddings);

    std::cout << "[TrackingPipeline] Team assignment training complete" << std::endl;
}

void TrackingPipeline::process_frame(const cv::Mat& frame,
                                      Detections& players,
                                      Detections& balls,
                                      Detections& referees,
                                      TrackingTiming* timing) {
    if (timing) {
        *timing = TrackingTiming{};
    }

    if (frame_idx_global_ % DETECTION_STRIDE == 0) {
        DetectionTiming dt;
        detection_pipeline_.detect_frame(frame, players, balls, referees, &dt);
        if (timing) {
            timing->det_preprocess_ms  = dt.preprocess_ms;
            timing->det_onnx_run_ms    = dt.onnx_run_ms;
            timing->det_postprocess_ms = dt.postprocess_ms;
            timing->ran_yolo = true;
        }

        // Suppress false-positive balls on close-up jerseys during replays or close shots
        bool is_closeup = false;
        for (const auto& b : players.boxes) {
            if (b.height() > 220.0f) { is_closeup = true; break; }
        }
        if (is_closeup) {
            Detections clean_balls;
            for (const auto& b : balls.boxes) {
                if (b.confidence >= 0.35f) clean_balls.add(b);
            }
            balls = std::move(clean_balls);
        }

        last_balls_ = balls;
        last_referees_ = referees;

        auto t_track0 = std::chrono::steady_clock::now();
        players = tracker_.update(players, false);
        auto t_track1 = std::chrono::steady_clock::now();
        if (timing) {
            timing->tracker_update_ms = std::chrono::duration<double, std::milli>(t_track1 - t_track0).count();
        }
    } else {
        auto t_track0 = std::chrono::steady_clock::now();
        players = tracker_.update(Detections{}, true);
        auto t_track1 = std::chrono::steady_clock::now();
        if (timing) {
            timing->tracker_update_ms = std::chrono::duration<double, std::milli>(t_track1 - t_track0).count();
            timing->ran_yolo = false;
        }
        balls = last_balls_;
        referees = last_referees_;
    }

    auto t_clus0 = std::chrono::steady_clock::now();
    if (!players.empty() && clustering_.is_trained()) {
        std::vector<int> needs_embedding;
        needs_embedding.reserve(players.size());

        for (int i = 0; i < players.size(); ++i) {
            int tid = players.boxes[i].track_id;
            if (tid < 0 || players.boxes[i].class_id == 2) continue; // Skip referees

            auto cache_it = team_cache_.find(tid);
            if (cache_it != team_cache_.end()) {
                auto val_it = last_validation_frame_.find(tid);
                int last_frame = (val_it != last_validation_frame_.end()) ? val_it->second : 0;

                if (frame_idx_global_ - last_frame >= REVALIDATE_INTERVAL) {
                    needs_embedding.push_back(i);
                } else {
                    players.boxes[i].class_id = cache_it->second;
                }
            } else {
                needs_embedding.push_back(i);
            }
        }

        if (!needs_embedding.empty()) {
            std::vector<BBox> new_boxes;
            new_boxes.reserve(needs_embedding.size());
            for (int idx : needs_embedding) {
                new_boxes.push_back(players.boxes[idx]);
            }

            auto crops = EmbeddingExtractor::get_player_crops(frame, new_boxes);

            if (!crops.empty()) {
                auto embeddings = clustering_.get_extractor().extract_batch(crops);
                auto team_labels = clustering_.predict(embeddings);

                for (size_t j = 0; j < needs_embedding.size() && j < team_labels.size(); ++j) {
                    int player_idx = needs_embedding[j];
                    int tid = players.boxes[player_idx].track_id;
                    int team = team_labels[j];

                    tracker_.update_team_votes(tid, team);
                    team_cache_[tid] = team;
                    players.boxes[player_idx].class_id = team;
                    last_validation_frame_[tid] = frame_idx_global_;
                }

                tracker_.update_team_labels(team_cache_);
            }
        }
    }
    auto t_clus1 = std::chrono::steady_clock::now();
    if (timing) {
        timing->clustering_ms = std::chrono::duration<double, std::milli>(t_clus1 - t_clus0).count();
    }

    frame_idx_global_++;
}

void TrackingPipeline::store_tracks(const Detections& players,
                                     const Detections& balls,
                                     const Detections& referees,
                                     int frame_idx,
                                     std::unordered_map<int, BBox>& player_tracks_for_frame,
                                     std::unordered_map<int, int>& player_teams_for_frame,
                                     std::unordered_map<int, BBox>& ball_tracks_map,
                                     std::unordered_map<int, BBox>& referee_tracks_for_frame) {
    player_tracks_for_frame.clear();
    player_teams_for_frame.clear();
    referee_tracks_for_frame.clear();

    for (const auto& box : players.boxes) {
        if (box.track_id >= 0) {
            player_tracks_for_frame[box.track_id] = box;
            player_teams_for_frame[box.track_id] = box.class_id;
        }
    }

    if (!balls.empty()) {
        ball_tracks_map[frame_idx] = balls.boxes[0];
    }

    int ref_id = 0;
    for (const auto& box : referees.boxes) {
        referee_tracks_for_frame[ref_id++] = box;
    }
}

void TrackingPipeline::annotate_frame(cv::Mat& frame,
                                       const std::unordered_map<int, BBox>& player_tracks_for_frame,
                                       const std::unordered_map<int, int>& player_teams_for_frame,
                                       const BBox& ball_track_for_frame,
                                       const std::unordered_map<int, BBox>& referee_tracks_for_frame) {
    FrameTracks ft;
    for (const auto& [tid, bbox] : player_tracks_for_frame) {
        ft.players.emplace_back(tid, bbox);
    }
    for (const auto& [tid, team] : player_teams_for_frame) {
        ft.player_teams.emplace_back(tid, team);
    }
    ft.ball = ball_track_for_frame;
    for (const auto& [rid, bbox] : referee_tracks_for_frame) {
        ft.referees.emplace_back(rid, bbox);
    }

    Detections player_dets = Annotator::tracks_to_player_detections(ft);
    Detections ball_dets = Annotator::tracks_to_ball_detections(ft);
    Detections referee_dets = Annotator::tracks_to_referee_detections(ft);

    annotator_.draw_all(frame, player_dets, ball_dets, referee_dets);
}

} // namespace soccer_radar
