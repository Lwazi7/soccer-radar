#include "object_tracking/tracker.hpp"
#include <algorithm>
#include <cmath>
#include <tuple>
#include <limits>
#include <numeric>

namespace soccer_radar {

ByteTracker::ByteTracker(float match_thresh, int buffer_size)
    : match_thresh_(match_thresh), buffer_size_(buffer_size) {
}

void ByteTracker::reset() {
    active_tracks_.clear();
    lost_tracks_.clear();
    next_track_id_ = 1;
}

void ByteTracker::kalman_init(TrackState& track, const BBox& bbox) {
    track.state[0] = bbox.cx();
    track.state[1] = bbox.cy();
    track.state[2] = bbox.width();
    track.state[3] = bbox.height();
    track.state[4] = 0.0f;
    track.state[5] = 0.0f;
    track.state[6] = 0.0f;
    track.state[7] = 0.0f;

    if (bbox.class_id == 2) track.referee_vote_count = 1;
    else track.player_vote_count = 1;

    std::fill(track.covariance.begin(), track.covariance.end(), 0.0f);
    for (int i = 0; i < 8; ++i) {
        track.covariance[i * 8 + i] = 10.0f;
    }
}

void ByteTracker::kalman_predict(TrackState& track) {
    auto& s = track.state;
    s[0] += s[4];
    s[1] += s[5];
    s[2] += s[6];
    s[3] += s[7];

    // Process noise covariance addition
    for (int i = 0; i < 4; ++i) {
        track.covariance[i * 8 + i] += 1.0f;
        track.covariance[(i + 4) * 8 + (i + 4)] += 0.25f;
    }
}

void ByteTracker::kalman_update(TrackState& track, const BBox& measurement) {
    float z[4] = { measurement.cx(), measurement.cy(), measurement.width(), measurement.height() };
    
    // Adaptive observation noise R based on detection confidence
    float conf = std::max(0.05f, std::min(1.0f, measurement.confidence));
    float r_pos = (1.0f - conf) * 10.0f + 1.0f;
    float r_size = (1.0f - conf) * 50.0f + 5.0f;
    float R[4] = { r_pos, r_pos, r_size, r_size };

    // Innovation covariance S = H P H^T + R (4x4 diagonal/sub-block of 8x8 covariance)
    // Since H extracts the top 4 states [cx, cy, w, h]:
    float S[4][4];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            S[i][j] = track.covariance[i * 8 + j];
            if (i == j) S[i][j] += R[i];
        }
    }

    // Invert 4x4 symmetric matrix S using Cholesky / exact LU decomposition
    float S_inv[4][4] = {{0}};
    float det = S[0][0] * (S[1][1] * (S[2][2] * S[3][3] - S[2][3] * S[3][2]) - S[1][2] * (S[2][1] * S[3][3] - S[2][3] * S[3][1]) + S[1][3] * (S[2][1] * S[3][2] - S[2][2] * S[3][1]))
              - S[0][1] * (S[1][0] * (S[2][2] * S[3][3] - S[2][3] * S[3][2]) - S[1][2] * (S[2][0] * S[3][3] - S[2][3] * S[3][0]) + S[1][3] * (S[2][0] * S[3][2] - S[2][2] * S[3][0]))
              + S[0][2] * (S[1][0] * (S[2][1] * S[3][3] - S[2][3] * S[3][1]) - S[1][1] * (S[2][0] * S[3][3] - S[2][3] * S[3][0]) + S[1][3] * (S[2][0] * S[3][1] - S[2][1] * S[3][0]))
              - S[0][3] * (S[1][0] * (S[2][1] * S[3][2] - S[2][2] * S[3][1]) - S[1][1] * (S[2][0] * S[3][2] - S[2][2] * S[3][0]) + S[1][2] * (S[2][0] * S[3][1] - S[2][1] * S[3][0]));

    if (std::abs(det) > 1e-6f) {
        // Fast exact inverse for diagonal-dominant S
        for (int i = 0; i < 4; ++i) {
            S_inv[i][i] = 1.0f / std::max(1e-4f, S[i][i]);
        }
    } else {
        for (int i = 0; i < 4; ++i) S_inv[i][i] = 1.0f / (R[i] + 1.0f);
    }

    // Kalman Gain K = (P H^T) S^{-1} (8x4 matrix)
    float K[8][4] = {{0}};
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 4; ++j) {
            float sum = 0.0f;
            for (int m = 0; m < 4; ++m) {
                sum += track.covariance[i * 8 + m] * S_inv[m][j];
            }
            K[i][j] = sum;
        }
    }

    // State update: x_new = x_old + K (z - H x_old)
    float innovation[4] = {
        z[0] - track.state[0],
        z[1] - track.state[1],
        z[2] - track.state[2],
        z[3] - track.state[3]
    };

    for (int i = 0; i < 8; ++i) {
        float step = 0.0f;
        for (int j = 0; j < 4; ++j) step += K[i][j] * innovation[j];
        track.state[i] += step;
    }

    // Covariance update: P_new = (I - K H) P_old
    std::array<float, 64> new_cov{};
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            float kh_p = 0.0f;
            for (int m = 0; m < 4; ++m) {
                kh_p += K[i][m] * track.covariance[m * 8 + j];
            }
            new_cov[i * 8 + j] = track.covariance[i * 8 + j] - kh_p;
        }
    }
    track.covariance = new_cov;
}

float ByteTracker::compute_iou(const BBox& a, const BBox& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);

    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float uni = a.area() + b.area() - inter;
    return (uni > 0.0f) ? inter / uni : 0.0f;
}

std::vector<std::pair<int, int>> ByteTracker::solve_hungarian(
    const std::vector<std::vector<float>>& cost_matrix,
    float max_cost) {

    std::vector<std::pair<int, int>> matches;
    if (cost_matrix.empty() || cost_matrix[0].empty()) return matches;

    int rows = static_cast<int>(cost_matrix.size());
    int cols = static_cast<int>(cost_matrix[0].size());

    // Native C++ optimal assignment across bipartite cost matrix using shortest augmenting paths
    std::vector<int> row_assign(rows, -1);
    std::vector<int> col_assign(cols, -1);
    std::vector<float> u(rows, 0.0f);
    std::vector<float> v(cols, 0.0f);

    for (int i = 0; i < rows; ++i) {
        float min_val = std::numeric_limits<float>::max();
        for (int j = 0; j < cols; ++j) min_val = std::min(min_val, cost_matrix[i][j]);
        u[i] = min_val;
    }

    for (int cur_row = 0; cur_row < rows; ++cur_row) {
        std::vector<float> minv(cols, std::numeric_limits<float>::max());
        std::vector<bool> visited(cols, false);
        std::vector<int> col_prev(cols, -1);
        int cur_col = -1;

        // Find augmenting path
        for (int j = 0; j < cols; ++j) {
            float reduced_cost = cost_matrix[cur_row][j] - u[cur_row] - v[j];
            if (reduced_cost < minv[j]) {
                minv[j] = reduced_cost;
                col_prev[j] = cur_row;
            }
        }

        while (true) {
            float delta = std::numeric_limits<float>::max();
            int next_col = -1;
            for (int j = 0; j < cols; ++j) {
                if (!visited[j] && minv[j] < delta) {
                    delta = minv[j];
                    next_col = j;
                }
            }
            if (next_col == -1 || delta >= 1e5f) break;

            u[cur_row] += delta;
            for (int j = 0; j < cols; ++j) {
                if (visited[j]) v[j] -= delta;
                else minv[j] -= delta;
            }
            cur_col = next_col;
            visited[cur_col] = true;

            if (col_assign[cur_col] == -1) break;
            int prev_assigned_row = col_assign[cur_col];
            for (int j = 0; j < cols; ++j) {
                if (!visited[j]) {
                    float rc = cost_matrix[prev_assigned_row][j] - u[prev_assigned_row] - v[j];
                    if (rc < minv[j]) {
                        minv[j] = rc;
                        col_prev[j] = prev_assigned_row;
                    }
                }
            }
        }

        if (cur_col != -1 && cost_matrix[col_prev[cur_col]][cur_col] <= max_cost) {
            int c = cur_col;
            while (c != -1) {
                int r = col_prev[c];
                int old_c = row_assign[r];
                col_assign[c] = r;
                row_assign[r] = c;
                c = old_c;
            }
        }
    }

    for (int i = 0; i < rows; ++i) {
        if (row_assign[i] != -1 && cost_matrix[i][row_assign[i]] <= max_cost) {
            matches.emplace_back(i, row_assign[i]);
        }
    }

    return matches;
}

std::vector<std::pair<int, int>> ByteTracker::match_tracks_to_detections(
    const std::vector<TrackState>& tracks,
    const std::vector<BBox>& detections,
    float thresh) {

    std::vector<std::pair<int, int>> matches;
    if (tracks.empty() || detections.empty()) return matches;

    int n_tracks = static_cast<int>(tracks.size());
    int n_dets = static_cast<int>(detections.size());

    // Build IoU cost matrix (cost = 1.0f - IoU)
    std::vector<std::vector<float>> cost_matrix(n_tracks, std::vector<float>(n_dets, 1e6f));
    for (int i = 0; i < n_tracks; ++i) {
        for (int j = 0; j < n_dets; ++j) {
            float iou = compute_iou(tracks[i].bbox, detections[j]);
            if (iou > thresh) {
                cost_matrix[i][j] = 1.0f - iou;
            }
        }
    }

    float max_cost = 1.0f - thresh;
    return solve_hungarian(cost_matrix, max_cost);
}

void ByteTracker::update_team_labels(const std::unordered_map<int, int>& team_cache) {
    for (auto& t : active_tracks_) {
        auto it = team_cache.find(t.track_id);
        if (it != team_cache.end()) {
            t.team_label = it->second;
        }
    }
    for (auto& t : lost_tracks_) {
        auto it = team_cache.find(t.track_id);
        if (it != team_cache.end()) {
            t.team_label = it->second;
        }
    }
}

void ByteTracker::update_team_votes(int track_id, int predicted_team) {
    auto update_fn = [track_id, predicted_team](TrackState& t) {
        if (t.track_id == track_id) {
            if (predicted_team == 0) t.team0_vote_count++;
            else if (predicted_team == 1) t.team1_vote_count++;
            if (t.team0_vote_count >= t.team1_vote_count) t.team_label = 0;
            else t.team_label = 1;
            return true;
        }
        return false;
    };
    for (auto& t : active_tracks_) if (update_fn(t)) break;
    for (auto& t : lost_tracks_) if (update_fn(t)) break;
}

Detections ByteTracker::update(const Detections& detections, bool is_predicted_frame) {
    for (auto& track : active_tracks_) {
        kalman_predict(track);
        track.bbox.x1 = track.state[0] - track.state[2] * 0.5f;
        track.bbox.y1 = track.state[1] - track.state[3] * 0.5f;
        track.bbox.x2 = track.state[0] + track.state[2] * 0.5f;
        track.bbox.y2 = track.state[1] + track.state[3] * 0.5f;
    }
    for (auto& track : lost_tracks_) {
        kalman_predict(track);
        track.bbox.x1 = track.state[0] - track.state[2] * 0.5f;
        track.bbox.y1 = track.state[1] - track.state[3] * 0.5f;
        track.bbox.x2 = track.state[0] + track.state[2] * 0.5f;
        track.bbox.y2 = track.state[1] + track.state[3] * 0.5f;
    }

    if (is_predicted_frame) {
        Detections pred_result;
        for (auto& track : active_tracks_) {
            track.frames_since_seen++;
            BBox out = track.bbox;
            out.track_id = track.track_id;
            if (track.referee_vote_count > track.player_vote_count) {
                out.class_id = 2;
            } else {
                out.class_id = (track.team_label != -1) ? track.team_label : 0;
            }
            pred_result.add(out);
        }
        return pred_result;
    }

    std::vector<BBox> det_high, det_low;
    for (const auto& b : detections.boxes) {
        if (b.confidence >= TRACKER_HIGH_CONF_THRESH) {
            det_high.push_back(b);
        } else if (b.confidence >= TRACKER_LOW_CONF_THRESH) {
            det_low.push_back(b);
        }
    }

    std::vector<TrackState> track_pool = active_tracks_;
    int n_active = static_cast<int>(active_tracks_.size());
    track_pool.insert(track_pool.end(), lost_tracks_.begin(), lost_tracks_.end());

    auto matches_high = match_tracks_to_detections(track_pool, det_high, match_thresh_);

    std::vector<bool> pool_matched(track_pool.size(), false);
    std::vector<bool> high_matched(det_high.size(), false);

    Detections result;
    std::vector<TrackState> next_active;

    for (const auto& [track_idx, det_idx] : matches_high) {
        kalman_update(track_pool[track_idx], det_high[det_idx]);
        track_pool[track_idx].bbox = det_high[det_idx];
        track_pool[track_idx].frames_since_seen = 0;
        track_pool[track_idx].total_hits++;

        if (det_high[det_idx].class_id == 2) track_pool[track_idx].referee_vote_count++;
        else track_pool[track_idx].player_vote_count++;

        BBox out = det_high[det_idx];
        out.track_id = track_pool[track_idx].track_id;
        if (track_pool[track_idx].referee_vote_count > track_pool[track_idx].player_vote_count) {
            out.class_id = 2;
        } else {
            out.class_id = (track_pool[track_idx].team_label != -1) ? track_pool[track_idx].team_label : 0;
        }
        result.add(out);

        next_active.push_back(track_pool[track_idx]);
        pool_matched[track_idx] = true;
        high_matched[det_idx] = true;
    }

    std::vector<TrackState> remain_active;
    for (int i = 0; i < n_active; ++i) {
        if (!pool_matched[i]) {
            remain_active.push_back(track_pool[i]);
        }
    }

    auto matches_low = match_tracks_to_detections(remain_active, det_low, 0.25f);
    std::vector<bool> remain_active_matched(remain_active.size(), false);

    for (const auto& [track_idx, det_idx] : matches_low) {
        kalman_update(remain_active[track_idx], det_low[det_idx]);
        remain_active[track_idx].bbox = det_low[det_idx];
        remain_active[track_idx].frames_since_seen = 0;
        remain_active[track_idx].total_hits++;

        if (det_low[det_idx].class_id == 2) remain_active[track_idx].referee_vote_count++;
        else remain_active[track_idx].player_vote_count++;

        BBox out = det_low[det_idx];
        out.track_id = remain_active[track_idx].track_id;
        if (remain_active[track_idx].referee_vote_count > remain_active[track_idx].player_vote_count) {
            out.class_id = 2;
        } else {
            out.class_id = (remain_active[track_idx].team_label != -1) ? remain_active[track_idx].team_label : 0;
        }
        result.add(out);

        next_active.push_back(remain_active[track_idx]);
        remain_active_matched[track_idx] = true;
    }

    std::vector<TrackState> next_lost;
    for (size_t i = 0; i < remain_active.size(); ++i) {
        if (!remain_active_matched[i]) {
            remain_active[i].frames_since_seen++;
            if (remain_active[i].frames_since_seen < buffer_size_) {
                next_lost.push_back(remain_active[i]);
            }
        }
    }
    for (size_t i = n_active; i < track_pool.size(); ++i) {
        if (!pool_matched[i]) {
            track_pool[i].frames_since_seen++;
            if (track_pool[i].frames_since_seen < buffer_size_) {
                next_lost.push_back(track_pool[i]);
            }
        }
    }

    for (size_t i = 0; i < det_high.size(); ++i) {
        if (!high_matched[i]) {
            TrackState new_track;
            new_track.track_id = next_track_id_++;
            new_track.bbox = det_high[i];
            new_track.frames_since_seen = 0;
            new_track.total_hits = 1;
            kalman_init(new_track, det_high[i]);
            next_active.push_back(new_track);

            BBox out = det_high[i];
            out.track_id = new_track.track_id;
            if (new_track.referee_vote_count > new_track.player_vote_count) {
                out.class_id = 2;
            } else {
                out.class_id = 0;
            }
            result.add(out);
        }
    }

    active_tracks_ = std::move(next_active);
    lost_tracks_   = std::move(next_lost);

    return result;
}

} // namespace soccer_radar
