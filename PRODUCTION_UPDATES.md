# Production hardening update

This workspace contains a source-level hardening pass over Soccer Radar v2.0. It directly addresses the review items and several issues found while validating the implementation.

## Implemented

- **Object/keypoint input reuse:** detector preprocessors now write directly into model-sized member buffers. Batch embedding input, FP16 conversion, output, and resize scratch buffers are also reused.
- **Stable Kalman filter:** the tracker now performs the complete `F P Fᵀ + Q` prediction, uses OpenCV Cholesky/SVD solves with jitter fallback, and applies the Joseph covariance update. Unsafe release `fast-math` is off by default.
- **Correct rectangular Hungarian assignment:** replaced the previous augmenting-path implementation with a padded, shortest-augmenting-path Hungarian solver. Assignment is bounded to the top 40 confidence-ranked detections.
- **Appearance Re-ID:** MobileNet descriptors are attached to tracks, L2-normalized, updated with an EMA, and combined with IoU for tightly gated lost-track recovery.
- **Temporal team labels:** the per-track majority result now updates the cache; a single fresh prediction can no longer overwrite accumulated votes.
- **Cluster validity:** training computes the standard silhouette score. Separation below `MIN_CLUSTER_SILHOUETTE` uses a safe single-team fallback rather than forcing two unreliable labels.
- **Efficient PCA:** uses OpenCV's sample-space `cv::PCA` instead of allocating and decomposing a 1280×1280 covariance matrix.
- **PROSAC-style homography:** confidence-ranked progressive subset sampling replaces arbitrary sampling. `solve_subset` was verified and retained, with SVD inversion checks, temporal smoothing, and a short stale-transform fallback.
- **Camera motion compensation:** Lucas–Kanade pyramidal optical flow propagates field keypoints between detector refreshes before homography estimation.
- **Ball motion:** missing detections use a constant-acceleration trajectory constrained to reach the next observation; box dimensions remain smoothly interpolated.
- **Hardware fallback visibility:** ONNX Runtime now logs explicitly when XNNPACK/NNAPI are unavailable.
- **Memory:** pass 1 stores compact tactical coordinates rather than a full 1050×680 image for every frame. Tactical images are redrawn during pass 2.
- **Tests:** `tests/core_tests.cpp` covers track assignment/Re-ID, temporal team voting, and ball interpolation. CTest integration is included.

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build build -j
ctest --test-dir build --output-on-failure
```

On Termux, `./build_termux.sh` remains the recommended build entry point. OpenCV's `video` module is now required for optical flow.

## Tunable production thresholds

The main safeguards are in `include/utils/constants.hpp`:

- `TRACKER_MAX_DETECTIONS`
- `REID_COSINE_THRESHOLD`
- `REID_EMA_ALPHA`
- `MIN_CLUSTER_SILHOUETTE`
- `BALL_INTERPOLATION_LIMIT`

These defaults are conservative. Calibrate Re-ID and clustering thresholds against representative match footage before claiming a deployment accuracy target.

## Validation performed

The complete executable was compiled on Debian 13 with GCC 14.2, OpenCV 4.10, and ONNX Runtime 1.22. A second strict-warning build used `-Wall -Wextra -Wpedantic -Wconversion -Wshadow`; project-source warnings were corrected (remaining diagnostics came from ONNX Runtime headers). CTest passed, and the core test executable also passed under AddressSanitizer and UndefinedBehaviorSanitizer.

Model binaries and representative match footage were not present, so end-to-end inference accuracy and target-device FPS still require validation before shipping.
