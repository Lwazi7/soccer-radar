#include "player_clustering/clustering.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <iostream>
#include <limits>
#include <opencv2/core.hpp>

namespace soccer_radar {

ClusteringManager::ClusteringManager(int n_components, int n_clusters)
    : n_components_(n_components), n_clusters_(n_clusters) {
}

void ClusteringManager::pca_fit(const std::vector<std::vector<float>>& data) {
    if (data.empty() || data.front().empty()) return;
    const int samples = static_cast<int>(data.size());
    const int dim = static_cast<int>(data.front().size());
    cv::Mat rows(samples, dim, CV_32F);
    for (int r = 0; r < samples; ++r) {
        if (static_cast<int>(data[r].size()) != dim) return;
        std::copy(data[r].begin(), data[r].end(), rows.ptr<float>(r));
    }

    // OpenCV's PCA selects the sample-space decomposition when samples << features,
    // avoiding construction/eigendecomposition of a 1280x1280 covariance matrix.
    const int components = std::min({n_components_, samples, dim});
    cv::PCA pca(rows, cv::Mat(), cv::PCA::DATA_AS_ROW, components);
    pca_mean_.assign(pca.mean.ptr<float>(), pca.mean.ptr<float>() + dim);
    pca_components_.assign(static_cast<size_t>(n_components_ * dim), 0.0f);
    for (int c = 0; c < components; ++c) {
        std::copy(pca.eigenvectors.ptr<float>(c),
                  pca.eigenvectors.ptr<float>(c) + dim,
                  pca_components_.begin() + static_cast<size_t>(c * dim));
    }
}

std::vector<std::vector<float>> ClusteringManager::pca_transform(
    const std::vector<std::vector<float>>& data) {

    int dim = static_cast<int>(pca_mean_.size());
    std::vector<std::vector<float>> result;
    result.reserve(data.size());

    for (const auto& sample : data) {
        std::vector<float> projected(n_components_, 0.0f);

        for (int c = 0; c < n_components_; ++c) {
            for (int i = 0; i < dim && i < static_cast<int>(sample.size()); ++i) {
                projected[c] += (sample[i] - pca_mean_[i]) *
                                pca_components_[c * dim + i];
            }
        }

        result.push_back(std::move(projected));
    }

    return result;
}

void ClusteringManager::kmeans_fit(const std::vector<std::vector<float>>& data) {
    if (data.empty()) return;

    int n_samples = static_cast<int>(data.size());
    int dim = static_cast<int>(data[0].size());

    double best_inertia = std::numeric_limits<double>::max();
    std::vector<std::vector<float>> best_centroids;

    // Multi-restart K-Means++ for global cluster optimality.
    for (int init_run = 0; init_run < KMEANS_RESTARTS; ++init_run) {
        std::mt19937 rng(42 + init_run * 17);
        std::vector<std::vector<float>> centroids;

        std::uniform_int_distribution<int> uniform(0, n_samples - 1);
        centroids.push_back(data[uniform(rng)]);

        for (int k = 1; k < n_clusters_; ++k) {
            std::vector<float> distances(n_samples, std::numeric_limits<float>::max());
            for (int i = 0; i < n_samples; ++i) {
                for (const auto& centroid : centroids) {
                    float dist = 0.0f;
                    for (int d = 0; d < dim; ++d) {
                        float diff = data[i][d] - centroid[d];
                        dist += diff * diff;
                    }
                    distances[i] = std::min(distances[i], dist);
                }
            }
            std::discrete_distribution<int> weighted(distances.begin(), distances.end());
            centroids.push_back(data[weighted(rng)]);
        }

        std::vector<int> assignments(n_samples, 0);
        for (int iter = 0; iter < KMEANS_MAX_ITER; ++iter) {
            bool changed = false;
            for (int i = 0; i < n_samples; ++i) {
                float min_dist = std::numeric_limits<float>::max();
                int best_cluster = 0;
                for (int k = 0; k < n_clusters_; ++k) {
                    float dist = 0.0f;
                    for (int d = 0; d < dim; ++d) {
                        float diff = data[i][d] - centroids[k][d];
                        dist += diff * diff;
                    }
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_cluster = k;
                    }
                }
                if (assignments[i] != best_cluster) {
                    assignments[i] = best_cluster;
                    changed = true;
                }
            }

            if (!changed) break;

            std::vector<std::vector<float>> new_centroids(n_clusters_, std::vector<float>(dim, 0.0f));
            std::vector<int> counts(n_clusters_, 0);
            for (int i = 0; i < n_samples; ++i) {
                int k = assignments[i];
                counts[k]++;
                for (int d = 0; d < dim; ++d) new_centroids[k][d] += data[i][d];
            }
            for (int k = 0; k < n_clusters_; ++k) {
                if (counts[k] > 0) {
                    for (int d = 0; d < dim; ++d) new_centroids[k][d] /= static_cast<float>(counts[k]);
                }
            }
            centroids = std::move(new_centroids);
        }

        double total_inertia = 0.0;
        for (int i = 0; i < n_samples; ++i) {
            int k = assignments[i];
            double dist_sq = 0.0;
            for (int d = 0; d < dim; ++d) {
                double diff = static_cast<double>(data[i][d] - centroids[k][d]);
                dist_sq += diff * diff;
            }
            total_inertia += dist_sq;
        }

        if (total_inertia < best_inertia) {
            best_inertia = total_inertia;
            best_centroids = std::move(centroids);
        }
    }

    kmeans_centroids_ = std::move(best_centroids);
}

std::vector<int> ClusteringManager::kmeans_predict(const std::vector<std::vector<float>>& data) {
    if (data.empty() || kmeans_centroids_.empty()) return {};

    int dim = static_cast<int>(kmeans_centroids_[0].size());
    std::vector<int> labels;
    labels.reserve(data.size());

    for (const auto& sample : data) {
        float min_dist = std::numeric_limits<float>::max();
        int best_cluster = 0;

        for (int k = 0; k < n_clusters_ && k < static_cast<int>(kmeans_centroids_.size()); ++k) {
            float dist = 0.0f;
            for (int d = 0; d < dim && d < static_cast<int>(sample.size()); ++d) {
                float diff = sample[d] - kmeans_centroids_[k][d];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = k;
            }
        }

        labels.push_back(best_cluster);
    }

    return labels;
}

void ClusteringManager::train(const std::vector<std::vector<float>>& embeddings) {
    if (embeddings.empty()) return;

    pca_fit(embeddings);
    auto reduced = pca_transform(embeddings);
    kmeans_fit(reduced);

    // Standard mean silhouette score. Training is intentionally small (normally 120
    // crops), so the O(n^2) validity check is cheap and runs only once per video.
    auto labels = kmeans_predict(reduced);
    double total = 0.0;
    int scored = 0;
    for (size_t i = 0; i < reduced.size(); ++i) {
        double same_sum = 0.0, other_sum = 0.0;
        int same_count = 0, other_count = 0;
        for (size_t j = 0; j < reduced.size(); ++j) {
            if (i == j) continue;
            double d2 = 0.0;
            for (size_t k = 0; k < reduced[i].size(); ++k) {
                const double d = reduced[i][k] - reduced[j][k];
                d2 += d * d;
            }
            const double distance = std::sqrt(d2);
            if (labels[j] == labels[i]) { same_sum += distance; ++same_count; }
            else { other_sum += distance; ++other_count; }
        }
        if (same_count > 0 && other_count > 0) {
            const double a = same_sum / same_count;
            const double b = other_sum / other_count;
            total += (b - a) / std::max(a, b);
            ++scored;
        }
    }
    silhouette_score_ = scored > 0 ? static_cast<float>(total / scored) : -1.0f;
    cluster_separation_valid_ = silhouette_score_ >= MIN_CLUSTER_SILHOUETTE;
    if (!cluster_separation_valid_) {
        std::cerr << "[Clustering] Team separation rejected (silhouette="
                  << silhouette_score_ << "); using a single-team fallback." << std::endl;
    }
    is_trained_ = true;
}

std::vector<int> ClusteringManager::predict(const std::vector<std::vector<float>>& embeddings) {
    if (!is_trained_ || embeddings.empty() || !cluster_separation_valid_) {
        return std::vector<int>(embeddings.size(), 0);
    }

    auto reduced = pca_transform(embeddings);
    return kmeans_predict(reduced);
}

std::vector<int> ClusteringManager::process_crops(const std::vector<cv::Mat>& crops, bool train_mode) {
    if (crops.empty()) return {};

    auto embeddings = extractor_.extract_batch(crops);

    if (train_mode) {
        train(embeddings);
    }

    return predict(embeddings);
}

} // namespace soccer_radar
