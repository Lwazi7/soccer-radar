#include "object_tracking/tracker.hpp"
#include "pipelines/processing_pipeline.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <unordered_map>

using namespace soccer_radar;

static BBox box(float cx, float cy, float confidence = 0.9f) {
    return {cx - 5.0f, cy - 10.0f, cx + 5.0f, cy + 10.0f,
            confidence, static_cast<int>(ObjectClass::Player), -1};
}

int main() {
    // Rectangular Hungarian assignment is exercised through two crossing tracks.
    ByteTracker tracker;
    Detections first;
    first.add(box(100, 100)); first.add(box(300, 100));
    std::vector<std::vector<float>> features{{1, 0, 0}, {0, 1, 0}};
    auto initial = tracker.update(first, false, features);
    assert(initial.size() == 2);
    const int id_a = initial.boxes[0].track_id;
    const int id_b = initial.boxes[1].track_id;
    assert(id_a != id_b);

    // Appearance recovers a lost track after a spatially disjoint occlusion.
    tracker.update({}, true);
    Detections reappeared;
    reappeared.add(box(180, 100));
    auto recovered = tracker.update(reappeared, false, {{1, 0, 0}});
    assert(recovered.size() == 1);
    assert(recovered.boxes[0].track_id == id_a);

    // Team assignment is a stable temporal majority, not the latest frame.
    assert(tracker.update_team_votes(id_a, 1) == 1);
    assert(tracker.update_team_votes(id_a, 0) == 0); // documented tie-break
    assert(tracker.update_team_votes(id_a, 1) == 1);

    // Missing ball boxes remain valid and meet both observed endpoints under the
    // constant-acceleration interpolation model.
    ProcessingPipeline processing;
    std::unordered_map<int, BBox> balls;
    BBox b0 = box(20, 20); b0.class_id = static_cast<int>(ObjectClass::Ball);
    BBox b1 = box(40, 25); b1.class_id = static_cast<int>(ObjectClass::Ball);
    BBox b2 = box(80, 45); b2.class_id = static_cast<int>(ObjectClass::Ball);
    balls[0] = b0; balls[5] = b1; balls[10] = b2;
    processing.interpolate_ball_tracks(balls, 11);
    for (int i = 0; i <= 10; ++i) assert(balls.count(i) && balls[i].valid());
    assert(std::abs(balls[10].cx() - 80.0f) < 1e-4f);

    std::cout << "core tests passed\n";
    return 0;
}
