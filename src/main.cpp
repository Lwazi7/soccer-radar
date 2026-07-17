#include "pipelines/detection_pipeline.hpp"
#include "pipelines/keypoint_pipeline.hpp"
#include "pipelines/processing_pipeline.hpp"
#include "pipelines/tactical_pipeline.hpp"
#include "pipelines/tracking_pipeline.hpp"
#include "utils/constants.hpp"
#include "utils/video_io.hpp"
#include "utils/types.hpp"

#include <opencv2/core.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

using namespace soccer_radar;

namespace json {

inline std::string escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;
        }
    }
    return out;
}

inline std::string f(float v, int prec = 2) {
    if (std::isnan(v) || std::isinf(v)) return "null";
    std::ostringstream ss;
    ss.precision(prec);
    ss << std::fixed << v;
    return ss.str();
}

inline std::string bbox_array(const BBox& b) {
    return "[" + f(b.x1) + "," + f(b.y1) + "," + f(b.x2) + "," + f(b.y2) + "]";
}

inline std::string point_array(const std::array<float,2>& p) {
    return "[" + f(p[0]) + "," + f(p[1]) + "]";
}

} // namespace json

class CompleteSoccerAnalysisPipeline {
public:
    CompleteSoccerAnalysisPipeline(const std::string& detection_model_path,
                                    const std::string& keypoint_model_path,
                                    const std::string& embedding_model_path)
        : detection_model_path_(detection_model_path),
          keypoint_model_path_(keypoint_model_path),
          embedding_model_path_(embedding_model_path) {
    }

    bool initialize() {
        std::cout << "=== Initializing Soccer Radar Pipeline ===" << std::endl;
        auto start = std::chrono::steady_clock::now();

        if (!keypoint_pipeline_.initialize(keypoint_model_path_)) {
            std::cerr << "Failed to initialize keypoint pipeline" << std::endl;
            return false;
        }

        if (!tracking_pipeline_.initialize(detection_model_path_, embedding_model_path_)) {
            std::cerr << "Failed to initialize tracking pipeline" << std::endl;
            return false;
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        std::cout << "All models initialized in " << elapsed << "ms" << std::endl;
        return true;
    }

    bool analyze_video(const std::string& video_path,
                       const std::string& output_path,
                       int max_frames = -1) {
        std::cout << "=== Starting Complete Soccer Analysis ===" << std::endl;
        auto total_start = std::chrono::steady_clock::now();

        bool json_output = (output_path.size() >= 5 &&
                           output_path.substr(output_path.size() - 5) == ".json");

        if (!initialize()) return false;

        std::cout << "\n[Step 2/7] Training team assignment models..." << std::endl;
        // Pass max_frames so sampling does not scan 2880 frames when processing small clips
        tracking_pipeline_.train_team_assignment(video_path, max_frames);

        std::cout << "\n[Step 3/7] Opening video..." << std::endl;
        VideoReader reader(video_path);
        if (!reader.is_open()) {
            std::cerr << "Failed to open video: " << video_path << std::endl;
            return false;
        }

        double video_fps = reader.fps();
        if (video_fps <= 0) video_fps = 30.0;

        std::cout << "Video: " << reader.width() << "x" << reader.height()
                  << " @ " << video_fps << " fps, "
                  << reader.total_frames() << " frames" << std::endl;

        VideoWriter writer;
        std::ofstream json_file;

        if (json_output) {
            json_file.open(output_path);
            if (!json_file.is_open()) {
                std::cerr << "Failed to open JSON output: " << output_path << std::endl;
                return false;
            }
            std::cout << "Output mode: JSON" << std::endl;

            json_file << "{\n";
            json_file << "  \"video\": {\n";
            json_file << "    \"path\": \"" << json::escape(video_path) << "\",\n";
            json_file << "    \"width\": " << reader.width() << ",\n";
            json_file << "    \"height\": " << reader.height() << ",\n";
            json_file << "    \"fps\": " << json::f(static_cast<float>(video_fps)) << ",\n";
            json_file << "    \"total_frames\": " << reader.total_frames() << "\n";
            json_file << "  },\n";
            json_file << "  \"frames\": [\n";
        } else {
            if (!writer.open(output_path, reader.width(), reader.height(), OUTPUT_FPS)) {
                std::cerr << "Failed to open output video: " << output_path << std::endl;
                return false;
            }
            std::cout << "Output mode: Video" << std::endl;
        }

        std::cout << "\n[Step 4/7] Processing frames..." << std::endl;

        std::unordered_map<int, BBox> player_tracks;
        std::unordered_map<int, int> player_teams;
        std::unordered_map<int, BBox> ball_tracks;
        std::unordered_map<int, BBox> referee_tracks;

        cv::Mat frame;
        int frame_idx = 0;
        int processed_count = 0;
        bool first_json_frame = true;

        KeypointData cached_keypoints;
        static constexpr int MAX_KEYPOINT_INTERVAL = 15;
        int frames_since_kpt = 0;

        double total_det_ms = 0, total_kpt_ms = 0;
        double total_tactical_ms = 0, total_annotate_ms = 0;

        auto proc_start = std::chrono::steady_clock::now();

        while (reader.read(frame)) {
            if (max_frames > 0 && frame_idx >= max_frames) break;

            if (frames_since_kpt >= MAX_KEYPOINT_INTERVAL || cached_keypoints.empty()) {
                auto t0 = std::chrono::steady_clock::now();
                KeypointData fresh_kpts = keypoint_pipeline_.detect_frame(frame);
                auto t1 = std::chrono::steady_clock::now();
                total_kpt_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

                int visible = 0;
                for (const auto& pt : fresh_kpts.points) {
                    if (pt[2] > KEYPOINT_CONFIDENCE) visible++;
                }
                if (visible >= MIN_KEYPOINTS_FOR_HOMOGRAPHY || cached_keypoints.empty()) {
                    cached_keypoints = std::move(fresh_kpts);
                    frames_since_kpt = 0;
                } else {
                    frames_since_kpt++;
                }
            } else {
                frames_since_kpt++;
            }

            auto t0 = std::chrono::steady_clock::now();
            Detections players, balls, referees;
            tracking_pipeline_.process_frame(frame, players, balls, referees);
            auto t1 = std::chrono::steady_clock::now();
            total_det_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

            tracking_pipeline_.store_tracks(players, balls, referees, frame_idx,
                                             player_tracks, player_teams,
                                             ball_tracks, referee_tracks);

            auto t2 = std::chrono::steady_clock::now();
            TacticalMetadata metadata;
            cv::Mat tactical_frame = tactical_pipeline_.process_frame(
                players, balls, referees, cached_keypoints, metadata);
            auto t3 = std::chrono::steady_clock::now();
            total_tactical_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();

            if (json_output) {
                float timestamp = static_cast<float>(frame_idx) / static_cast<float>(video_fps);

                if (!first_json_frame) json_file << ",\n";
                first_json_frame = false;

                json_file << "    {\n";
                json_file << "      \"frame\": " << frame_idx << ",\n";
                json_file << "      \"timestamp\": " << json::f(timestamp, 3) << ",\n";

                json_file << "      \"players\": [";
                bool first_player = true;
                for (const auto& box : players.boxes) {
                    if (!first_player) json_file << ",";
                    first_player = false;
                    json_file << "\n        {";
                    json_file << "\"track_id\":" << box.track_id;
                    json_file << ",\"bbox\":" << json::bbox_array(box);
                    json_file << ",\"team\":" << box.class_id;
                    json_file << ",\"confidence\":" << json::f(box.confidence);
                    json_file << "}";
                }
                if (!players.empty()) json_file << "\n      ";
                json_file << "],\n";

                json_file << "      \"ball\": ";
                if (!balls.empty()) {
                    const auto& b = balls.boxes[0];
                    json_file << "{\"bbox\":" << json::bbox_array(b)
                              << ",\"confidence\":" << json::f(b.confidence) << "}";
                } else {
                    json_file << "null";
                }
                json_file << ",\n";

                json_file << "      \"referees\": [";
                bool first_ref = true;
                for (const auto& box : referees.boxes) {
                    if (!first_ref) json_file << ",";
                    first_ref = false;
                    json_file << "\n        {";
                    json_file << "\"bbox\":" << json::bbox_array(box);
                    json_file << ",\"confidence\":" << json::f(box.confidence);
                    json_file << "}";
                }
                if (!referees.empty()) json_file << "\n      ";
                json_file << "],\n";

                json_file << "      \"tactical\": {";
                json_file << "\"homography_valid\":" << (metadata.transform_valid ? "true" : "false");
                json_file << ",\"num_team1\":" << metadata.num_team1;
                json_file << ",\"num_team2\":" << metadata.num_team2;
                json_file << "}\n";

                json_file << "    }";
            } else {
                auto t4 = std::chrono::steady_clock::now();
                tracking_pipeline_.annotate_frame(frame, player_tracks, player_teams,
                                                   ball_tracks[frame_idx], referee_tracks);

                cv::Mat output_frame = TacticalPipeline::create_overlay(
                    frame, tactical_frame, OVERLAY_WIDTH, OVERLAY_HEIGHT);
                auto t5 = std::chrono::steady_clock::now();
                total_annotate_ms += std::chrono::duration<double, std::milli>(t5 - t4).count();

                writer.write(output_frame);
            }

            frame_idx++;
            processed_count++;

            if (processed_count % 10 == 0) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - proc_start).count();
                int64_t elapsed_ms = static_cast<int64_t>(elapsed);
                float fps = 1000.0f * processed_count / static_cast<float>(std::max<int64_t>(16, elapsed_ms));
                float avg_det = total_det_ms / processed_count;
                std::cout << "  Frame " << frame_idx
                          << " | " << fps << " fps"
                          << " | det+track: " << avg_det << "ms"
                          << std::endl;
            }
        }

        auto proc_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - proc_start).count();

        std::cout << "\n[Step 5/7] Interpolating ball tracks..." << std::endl;
        std::cout << "\n[Step 6/7] Finalizing output..." << std::endl;

        if (json_output) {
            json_file << "\n  ]\n}\n";
            json_file.close();
        } else {
            writer.close();
        }
        reader.close();

        auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - total_start).count();

        std::cout << "\n=== Soccer Radar Analysis Complete ===" << std::endl;
        std::cout << "Frames processed: " << processed_count << std::endl;
        std::cout << "Processing time: " << proc_elapsed << "ms" << std::endl;
        if (proc_elapsed > 0) {
            std::cout << "Average FPS: " << (1000.0f * processed_count / proc_elapsed) << std::endl;
        }
        std::cout << "\nPer-frame timing breakdown:" << std::endl;
        if (processed_count > 0) {
            std::cout << "  Detection + Tracking: " << (total_det_ms / processed_count) << " ms" << std::endl;
            std::cout << "  Tactical:              " << (total_tactical_ms / processed_count) << " ms" << std::endl;
            if (!json_output) {
                std::cout << "  Annotate + Overlay:    " << (total_annotate_ms / processed_count) << " ms" << std::endl;
            }
        }
        std::cout << "Total time: " << total_elapsed << "ms" << std::endl;
        std::cout << "Output saved to: " << output_path << std::endl;

        return true;
    }

private:
    std::string detection_model_path_;
    std::string keypoint_model_path_;
    std::string embedding_model_path_;

    KeypointPipeline keypoint_pipeline_;
    TrackingPipeline tracking_pipeline_;
    TacticalPipeline tactical_pipeline_;
    ProcessingPipeline processing_pipeline_;
};

int main(int argc, char** argv) {
    std::cout << "=== Soccer Radar v2.0 ===" << std::endl;
    std::cout << "C++ Soccer Analysis Pipeline" << std::endl;
    std::cout << "Optimized for constrained devices (Termux/ARM)" << std::endl;
    std::cout << std::endl;

    std::string video_path = "videos/test_720p.mp4";
    std::string output_path = "";
    int max_frames = -1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--video" && i + 1 < argc) {
            video_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--frames" && i + 1 < argc) {
            max_frames = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: soccer_radar [options]" << std::endl;
            std::cout << "  --video <path>    Input video path (default: videos/test_720p.mp4)" << std::endl;
            std::cout << "  --output <path>   Output: .json for data, .mp4 for video (default: auto .mp4)" << std::endl;
            std::cout << "  --frames <n>      Max frames to process (-1 for all)" << std::endl;
            return 0;
        }
    }

    if (output_path.empty()) {
        output_path = ProcessingPipeline::generate_output_path(video_path, "_analysis");
    }

    std::string detection_model = OBJECT_DETECTION_MODEL;
    std::string keypoint_model = KEYPOINT_DETECTION_MODEL;
    std::string embedding_model = EMBEDDING_MODEL;

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Video: " << video_path << std::endl;
    std::cout << "  Output: " << output_path << std::endl;
    std::cout << "  Detection model: " << detection_model << std::endl;
    std::cout << "  Keypoint model: " << keypoint_model << std::endl;
    std::cout << "  Embedding model: " << embedding_model << std::endl;
    std::cout << "  Max frames: " << (max_frames < 0 ? "all" : std::to_string(max_frames)) << std::endl;
    std::cout << std::endl;

    CompleteSoccerAnalysisPipeline pipeline(detection_model, keypoint_model, embedding_model);

    if (!pipeline.analyze_video(video_path, output_path, max_frames)) {
        std::cerr << "Analysis failed!" << std::endl;
        return 1;
    }

    return 0;
}
