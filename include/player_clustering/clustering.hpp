#pragma once

#include "embeddings.hpp"
#include "utils/types.hpp"
#include "utils/constants.hpp"
#include <opencv2/core.hpp>
#include <vector>

namespace soccer_radar {

// Team clustering using PCA dimensionality reduction + K-means.
// Replaces the original UMAP + K-means pipeline.
// PCA is chosen for constrained devices because:
// 1. It's a simple linear transform (matrix multiply at inference)
// 2. No nearest-neighbor graph construction needed
// 3. Deterministic and fast
// Accuracy is maintained because the embedding space from MobileNetV4
// is already well-structured and only needs mild dimensionality reduction.
class ClusteringManager {
public:
    ClusteringManager(int n_components = PCA_COMPONENTS,
                      int n_clusters = NUM_TEAMS);
    ~ClusteringManager() = default;

    // Train PCA and K-means on collected embeddings
    void train(const std::vector<std::vector<float>>& embeddings);

    // Predict team labels for new embeddings
    std::vector<int> predict(const std::vector<std::vector<float>>& embeddings);

    // Full pipeline: crops -> embeddings -> PCA -> K-means -> labels
    std::vector<int> process_crops(const std::vector<cv::Mat>& crops, bool train_mode = false);

    // Access the embedding extractor
    EmbeddingExtractor& get_extractor() { return extractor_; }
    const EmbeddingExtractor& get_extractor() const { return extractor_; }

    bool is_trained() const { return is_trained_; }
    bool has_valid_team_separation() const { return cluster_separation_valid_; }
    float silhouette_score() const { return silhouette_score_; }

private:
    // PCA implementation (fit and transform)
    void pca_fit(const std::vector<std::vector<float>>& data);
    std::vector<std::vector<float>> pca_transform(
        const std::vector<std::vector<float>>& data);

    // K-means implementation
    void kmeans_fit(const std::vector<std::vector<float>>& data);
    std::vector<int> kmeans_predict(const std::vector<std::vector<float>>& data);

    int n_components_;
    int n_clusters_;
    bool is_trained_{false};
    bool cluster_separation_valid_{false};
    float silhouette_score_{-1.0f};

    // PCA parameters
    std::vector<float> pca_mean_;       // Mean vector (dim)
    std::vector<float> pca_components_; // Principal components (n_components x dim), row-major

    // K-means centroids
    std::vector<std::vector<float>> kmeans_centroids_;

    EmbeddingExtractor extractor_;
};

} // namespace soccer_radar
