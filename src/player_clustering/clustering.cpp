#include "player_clustering/clustering.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <iostream>

namespace soccer_radar {

ClusteringManager::ClusteringManager(int n_components, int n_clusters)
    : n_components_(n_components), n_clusters_(n_clusters) {
}

void ClusteringManager::pca_fit(const std::vector<std::vector<float>>& data) {
    if (data.empty()) return;

    int n_samples = static_cast<int>(data.size());
    int dim = static_cast<int>(data[0].size());

    // Compute mean
    pca_mean_.assign(dim, 0.0f);
    for (const auto& sample : data) {
        for (int i = 0; i < dim; ++i) {
            pca_mean_[i] += sample[i];
        }
    }
    for (float& m : pca_mean_) m /= static_cast<float>(n_samples);

    // Compute covariance matrix (centered data)
    std::vector<float> cov(dim * dim, 0.0f);

    for (const auto& sample : data) {
        for (int i = 0; i < dim; ++i) {
            float xi = sample[i] - pca_mean_[i];
            for (int j = i; j < dim; ++j) {
                float xj = sample[j] - pca_mean_[j];
                cov[i * dim + j] += xi * xj;
            }
        }
    }

    // Symmetrize and normalize
    for (int i = 0; i < dim; ++i) {
        for (int j = i; j < dim; ++j) {
            cov[i * dim + j] /= static_cast<float>(std::max(1, n_samples - 1));
            cov[j * dim + i] = cov[i * dim + j];
        }
    }

    pca_components_.resize(static_cast<size_t>(n_components_ * dim), 0.0f);

    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0, 1);

    std::vector<float> temp_cov = cov;

    for (int comp = 0; comp < n_components_; ++comp) {
        std::vector<float> v(dim);
        for (float& vi : v) vi = normal(rng);

        // Power iteration with Gram-Schmidt orthogonalization against previous components
        for (int iter = 0; iter < 30; ++iter) {
            std::vector<float> v_new(dim, 0.0f);

            for (int i = 0; i < dim; ++i) {
                for (int j = 0; j < dim; ++j) {
                    v_new[i] += temp_cov[i * dim + j] * v[j];
                }
            }

            // Gram-Schmidt orthogonalization against previously found principal components
            for (int p = 0; p < comp; ++p) {
                float dot = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    dot += v_new[d] * pca_components_[p * dim + d];
                }
                for (int d = 0; d < dim; ++d) {
                    v_new[d] -= dot * pca_components_[p * dim + d];
                }
            }

            // Normalize
            float norm = 0.0f;
            for (float vi : v_new) norm += vi * vi;
            norm = std::sqrt(norm);
            if (norm > 1e-8f) {
                for (float& vi : v_new) vi /= norm;
            }

            v = std::move(v_new);
        }

        std::copy(v.begin(), v.end(), pca_components_.begin() + comp * dim);

        // Compute eigenvalue: lambda = v^T * cov * v
        float eigenvalue = 0.0f;
        for (int i = 0; i < dim; ++i) {
            float cv_i = 0.0f;
            for (int j = 0; j < dim; ++j) {
                cv_i += temp_cov[i * dim + j] * v[j];
            }
            eigenvalue += v[i] * cv_i;
        }

        // Deflate
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                temp_cov[i * dim + j] -= eigenvalue * v[i] * v[j];
            }
        }
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

    std::mt19937 rng(42);
    kmeans_centroids_.clear();

    std::uniform_int_distribution<int> uniform(0, n_samples - 1);
    kmeans_centroids_.push_back(data[uniform(rng)]);

    for (int k = 1; k < n_clusters_; ++k) {
        std::vector<float> distances(n_samples, std::numeric_limits<float>::max());

        for (int i = 0; i < n_samples; ++i) {
            for (const auto& centroid : kmeans_centroids_) {
                float dist = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    float diff = data[i][d] - centroid[d];
                    dist += diff * diff;
                }
                distances[i] = std::min(distances[i], dist);
            }
        }

        std::discrete_distribution<int> weighted(distances.begin(), distances.end());
        kmeans_centroids_.push_back(data[weighted(rng)]);
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
                    float diff = data[i][d] - kmeans_centroids_[k][d];
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
            for (int d = 0; d < dim; ++d) {
                new_centroids[k][d] += data[i][d];
            }
        }

        for (int k = 0; k < n_clusters_; ++k) {
            if (counts[k] > 0) {
                for (int d = 0; d < dim; ++d) {
                    new_centroids[k][d] /= static_cast<float>(counts[k]);
                }
            }
        }

        kmeans_centroids_ = std::move(new_centroids);
    }
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

    is_trained_ = true;
}

std::vector<int> ClusteringManager::predict(const std::vector<std::vector<float>>& embeddings) {
    if (!is_trained_ || embeddings.empty()) return std::vector<int>(embeddings.size(), 0);

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
