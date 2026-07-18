#include "object_tracking/tracker.hpp"
#include <algorithm>
#include <cmath>
#include <tuple>
#include <limits>
#include <numeric>
#include <opencv2/core.hpp>

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
    s[0] += s[4]; s[1] += s[5]; s[2] += s[6]; s[3] += s[7];
    s[2] = std::max(s[2], 1.0f);
    s[3] = std::max(s[3], 1.0f);

    cv::Matx<float, 8, 8> F = cv::Matx<float, 8, 8>::eye();
    for (int i = 0; i < 4; ++i) F(i, i + 4) = 1.0f;
    cv::Matx<float, 8, 8> P;
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c)
            P(r, c) = track.covariance[r * 8 + c];

    cv::Matx<float, 8, 8> Q = cv::Matx<float, 8, 8>::zeros();
    for (int i = 0; i < 4; ++i) {
        Q(i, i) = 1.0f;
        Q(i + 4, i + 4) = 0.25f;
    }
    const auto predicted = F * P * F.t() + Q;
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c)
            track.covariance[r * 8 + c] = predicted(r, c);
}

void ByteTracker::kalman_update(TrackState& track, const BBox& measurement) {
    const cv::Vec4f z(measurement.cx(), measurement.cy(),
                      measurement.width(), measurement.height());

    const float conf = std::clamp(measurement.confidence, 0.05f, 1.0f);
    const cv::Matx44f R = cv::Matx44f::diag(cv::Vec4f(
        (1.0f - conf) * 10.0f + 1.0f,
        (1.0f - conf) * 10.0f + 1.0f,
        (1.0f - conf) * 50.0f + 5.0f,
        (1.0f - conf) * 50.0f + 5.0f));

    cv::Matx<float, 8, 8> P;
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c)
            P(r, c) = track.covariance[r * 8 + c];

    cv::Matx44f S;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            S(r, c) = P(r, c) + R(r, c);

    cv::Matx44f S_inv;
    bool inverted = cv::invert(S, S_inv, cv::DECOMP_CHOLESKY);
    if (!inverted) {
        // Retry with diagonal jitter rather than silently applying a corrupt gain.
        for (int i = 0; i < 4; ++i) S(i, i) += 1e-3f;
        inverted = cv::invert(S, S_inv, cv::DECOMP_SVD);
    }
    if (!inverted || !cv::checkRange(cv::Mat(S_inv))) return;

    cv::Matx<float, 8, 4> PHt;
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 4; ++c)
            PHt(r, c) = P(r, c);
    const cv::Matx<float, 8, 4> K = PHt * S_inv;

    const cv::Vec4f innovation = z - cv::Vec4f(
        track.state[0], track.state[1], track.state[2], track.state[3]);
    const cv::Vec<float, 8> correction = K * innovation;
    for (int i = 0; i < 8; ++i) track.state[i] += correction[i];

    // Joseph stabilized covariance update preserves symmetry/positive semi-definiteness.
    cv::Matx<float, 8, 8> I = cv::Matx<float, 8, 8>::eye();
    cv::Matx<float, 8, 8> KH = cv::Matx<float, 8, 8>::zeros();
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 4; ++c)
            KH(r, c) = K(r, c);
    const auto A = I - KH;
    cv::Matx<float, 8, 8> KRKt = cv::Matx<float, 8, 8>::zeros();
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c)
            for (int k = 0; k < 4; ++k)
                KRKt(r, c) += K(r, k) * R(k, k) * K(c, k);
    cv::Matx<float, 8, 8> P_new = A * P * A.t() + KRKt;
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            const float sym = 0.5f * (P_new(r, c) + P_new(c, r));
            track.covariance[r * 8 + c] = sym;
        }
        track.covariance[r * 8 + r] = std::max(track.covariance[r * 8 + r], 1e-6f);
    }
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

float ByteTracker::cosine_distance(const std::vector<float>& a,
                                   const std::vector<float>& b) {
    if (a.empty() || a.size() != b.size()) return 1.0f;
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
    }
    if (na <= 1e-12f || nb <= 1e-12f) return 1.0f;
    return 1.0f - std::clamp(dot / std::sqrt(na * nb), -1.0f, 1.0f);
}

void ByteTracker::update_appearance(TrackState& track,
                                    const std::vector<float>& feature) {
    if (feature.empty()) return;
    if (track.appearance.size() != feature.size()) {
        track.appearance = feature;
    } else {
        for (size_t i = 0; i < feature.size(); ++i) {
            track.appearance[i] = REID_EMA_ALPHA * track.appearance[i] +
                                  (1.0f - REID_EMA_ALPHA) * feature[i];
        }
    }
    float norm = 0.0f;
    for (float v : track.appearance) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-6f) for (float& v : track.appearance) v /= norm;
}

std::vector<std::pair<int, int>> ByteTracker::solve_hungarian(
    const std::vector<std::vector<float>>& cost_matrix,
    float max_cost) {

    if (cost_matrix.empty() || cost_matrix.front().empty()) return {};
    const int rows = static_cast<int>(cost_matrix.size());
    const int cols = static_cast<int>(cost_matrix.front().size());
    const int n = std::max(rows, cols);
    constexpr double forbidden = 1e6;

    // Standard shortest-augmenting-path Hungarian implementation. Padding makes the
    // rectangular case explicit and avoids dropping rows when tracks > detections.
    std::vector<double> u(n + 1), v(n + 1);
    std::vector<int> p(n + 1), way(n + 1);
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(n + 1, forbidden);
        std::vector<unsigned char> used(n + 1, false);
        do {
            used[j0] = true;
            const int i0 = p[j0];
            double delta = forbidden;
            int j1 = 0;
            for (int j = 1; j <= n; ++j) if (!used[j]) {
                double c = max_cost + 1.0; // dummy assignment
                if (i0 <= rows && j <= cols) c = cost_matrix[i0 - 1][j - 1];
                const double cur = c - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else minv[j] -= delta;
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    std::vector<std::pair<int, int>> matches;
    for (int j = 1; j <= n; ++j) {
        const int i = p[j];
        if (i >= 1 && i <= rows && j <= cols &&
            cost_matrix[i - 1][j - 1] <= max_cost) {
            matches.emplace_back(i - 1, j - 1);
        }
    }
    return matches;
}

std::vector<std::pair<int, int>> ByteTracker::match_tracks_to_detections(
    const std::vector<TrackState>& tracks,
    const std::vector<BBox>& detections,
    const std::vector<std::vector<float>>& detection_features,
    float thresh) {

    if (tracks.empty() || detections.empty()) return {};
    const int n_tracks = static_cast<int>(tracks.size());
    const int n_dets = static_cast<int>(detections.size());
    std::vector<std::vector<float>> cost(n_tracks, std::vector<float>(n_dets, 1e6f));

    for (int i = 0; i < n_tracks; ++i) {
        for (int j = 0; j < n_dets; ++j) {
            const float iou = compute_iou(tracks[i].bbox, detections[j]);
            const bool has_feature = j < static_cast<int>(detection_features.size()) &&
                                     !detection_features[j].empty() && !tracks[i].appearance.empty();
            const float app = has_feature
                ? cosine_distance(tracks[i].appearance, detection_features[j]) : 1.0f;

            // Spatial overlap handles normal motion; appearance may recover a lost player
            // after an occlusion, but is gated tightly to avoid same-kit identity swaps.
            if (iou > thresh) {
                cost[i][j] = has_feature ? 0.75f * (1.0f - iou) + 0.25f * app
                                         : 1.0f - iou;
            } else if (tracks[i].frames_since_seen > 0 && has_feature &&
                       app <= REID_COSINE_THRESHOLD) {
                cost[i][j] = 0.45f + 0.45f * app + 0.10f * (1.0f - iou);
            }
        }
    }
    return solve_hungarian(cost, 0.75f);
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

int ByteTracker::update_team_votes(int track_id, int predicted_team) {
    auto update_fn = [track_id, predicted_team](TrackState& t) {
        if (t.track_id == track_id) {
            if (predicted_team != 0 && predicted_team != 1) return true;
            t.team_vote_history.push_back(predicted_team);
            if (t.team_vote_history.size() > TEAM_VOTE_WINDOW) {
                t.team_vote_history.erase(t.team_vote_history.begin());
            }
            t.team0_vote_count = static_cast<int>(std::count(
                t.team_vote_history.begin(), t.team_vote_history.end(), 0));
            t.team1_vote_count = static_cast<int>(t.team_vote_history.size()) - t.team0_vote_count;
            t.team_label = (t.team0_vote_count >= t.team1_vote_count) ? 0 : 1;
            return true;
        }
        return false;
    };
    for (auto& t : active_tracks_) if (update_fn(t)) return t.team_label;
    for (auto& t : lost_tracks_) if (update_fn(t)) return t.team_label;
    return predicted_team;
}

Detections ByteTracker::update(const Detections& detections, bool is_predicted_frame,
                               const std::vector<std::vector<float>>& detection_features) {
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

    // Keep assignment complexity bounded without allowing arbitrary detector order
    // to decide which observations survive on noisy crowd frames.
    std::vector<size_t> order(detections.boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return detections.boxes[a].confidence > detections.boxes[b].confidence;
    });
    if (order.size() > TRACKER_MAX_DETECTIONS) order.resize(TRACKER_MAX_DETECTIONS);

    std::vector<BBox> det_high, det_low;
    std::vector<std::vector<float>> feat_high, feat_low;
    for (size_t idx : order) {
        const auto& b = detections.boxes[idx];
        const std::vector<float> empty;
        const auto& feature = idx < detection_features.size() ? detection_features[idx] : empty;
        if (b.confidence >= TRACKER_HIGH_CONF_THRESH) {
            det_high.push_back(b); feat_high.push_back(feature);
        } else if (b.confidence >= TRACKER_LOW_CONF_THRESH) {
            det_low.push_back(b); feat_low.push_back(feature);
        }
    }

    std::vector<TrackState> track_pool = active_tracks_;
    int n_active = static_cast<int>(active_tracks_.size());
    track_pool.insert(track_pool.end(), lost_tracks_.begin(), lost_tracks_.end());

    auto matches_high = match_tracks_to_detections(track_pool, det_high, feat_high, match_thresh_);

    std::vector<bool> pool_matched(track_pool.size(), false);
    std::vector<bool> high_matched(det_high.size(), false);

    Detections result;
    std::vector<TrackState> next_active;

    for (const auto& [track_idx, det_idx] : matches_high) {
        kalman_update(track_pool[track_idx], det_high[det_idx]);
        track_pool[track_idx].bbox = det_high[det_idx];
        track_pool[track_idx].frames_since_seen = 0;
        track_pool[track_idx].total_hits++;
        update_appearance(track_pool[track_idx], feat_high[det_idx]);

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

    auto matches_low = match_tracks_to_detections(remain_active, det_low, feat_low, 0.25f);
    std::vector<bool> remain_active_matched(remain_active.size(), false);

    for (const auto& [track_idx, det_idx] : matches_low) {
        kalman_update(remain_active[track_idx], det_low[det_idx]);
        remain_active[track_idx].bbox = det_low[det_idx];
        remain_active[track_idx].frames_since_seen = 0;
        remain_active[track_idx].total_hits++;
        update_appearance(remain_active[track_idx], feat_low[det_idx]);

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
            update_appearance(new_track, feat_high[i]);
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
