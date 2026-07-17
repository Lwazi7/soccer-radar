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
#include <iomanip>

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

static void print_progress_bar(int current, int total, double elapsed_ms, const std::string& prefix) {
    if (total <= 0) return;
    int width = 30;
    float pct = static_cast<float>(current) / static_cast<float>(total);
    int pos = static_cast<int>(width * pct);

    double fps = (elapsed_ms > 0) ? (1000.0 * current / elapsed_ms) : 0.0;
    int eta_s = (fps > 0) ? static_cast<int>((total - current) / fps) : 0;

    std::cout << "\r" << prefix << " [";
    for (int i = 0; i < width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << current << "/" << total << " ("
              << static_cast<int>(pct * 100.0f) << "%) | "
              << std::fixed << std::setprecision(2) << fps << " fps";
    if (current < total) {
        std::cout << " | ETA: " << eta_s << "s   " << std::flush;
    } else {
        std::cout << "                   \n" << std::flush;
    }
}

class CompleteSoccerAnalysisPipeline {
public:
    CompleteSoccerAnalysisPipeline(const std::string& detection_model_path,
                                    const std::string& keypoint_model_path,
                                    const std::string& embedding_model_path,
                                    bool verbose = false)
        : detection_model_path_(detection_model_path),
          keypoint_model_path_(keypoint_model_path),
          embedding_model_path_(embedding_model_path),
          verbose_(verbose) {
    }

    bool initialize() {
        std::cout << "=== Initializing Soccer Radar Pipeline ===" << std::endl;
        auto start = std::chrono::steady_clock::now();

        if (!keypoint_pipeline_.initialize(keypoint_model_path_)) {
            std::cerr << "Failed to initialize Football-TV2Radar corner pipeline" << std::endl;
            return false;
        }

        if (!tracking_pipeline_.initialize(detection_model_path_, embedding_model_path_)) {
            std::cerr << "Failed to initialize tracking pipeline" << std::endl;
            return false;
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        std::cout << "All models initialized in " << elapsed << "ms\n" << std::endl;
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

        std::cout << "[Step 2/7] Training team assignment models..." << std::endl;
        tracking_pipeline_.train_team_assignment(video_path, max_frames);

        std::cout << "\n[Step 3/7] Opening video..." << std::endl;
        VideoReader reader(video_path);
        if (!reader.is_open()) {
            std::cerr << "Failed to open video: " << video_path << std::endl;
            return false;
        }

        double video_fps = reader.fps();
        if (video_fps <= 0) video_fps = 30.0;
        int total_video_frames = reader.total_frames();
        int target_total = (max_frames > 0 && max_frames < total_video_frames) ? max_frames : total_video_frames;

        std::cout << "Video: " << reader.width() << "x" << reader.height()
                  << " @ " << video_fps << " fps, "
                  << total_video_frames << " frames" << std::endl;

        std::cout << "\n[Step 4/7] Processing frames (Pass 1: Detection, Tracking & Clustering across Video)..." << std::endl;

        // Per-frame track storage to eliminate frozen circles dumped across frames
        std::unordered_map<int, std::unordered_map<int, BBox>> video_player_tracks;
        std::unordered_map<int, std::unordered_map<int, int>>  video_player_teams;
        std::unordered_map<int, BBox>                          video_ball_tracks;
        std::unordered_map<int, std::unordered_map<int, BBox>> video_referee_tracks;
        std::unordered_map<int, cv::Mat>                       tactical_frames;
        std::unordered_map<int, TacticalMetadata>              tactical_metadata;

        cv::Mat frame;
        int frame_idx = 0;
        int processed_count = 0;

        KeypointData cached_corners;
        static constexpr int MAX_CORNER_INTERVAL = 15;
        int frames_since_corner = 0;

        double total_det_preprocess_ms = 0, total_det_onnx_ms = 0, total_det_postprocess_ms = 0;
        double total_tracker_ms = 0, total_clustering_ms = 0;
        int yolo_run_count = 0;

        double total_corner_ms = 0;
        double total_tactical_ms = 0;

        auto proc_start = std::chrono::steady_clock::now();

        while (reader.read(frame)) {
            if (max_frames > 0 && frame_idx >= max_frames) break;

            if (frames_since_corner >= MAX_CORNER_INTERVAL || cached_corners.empty()) {
                auto t0 = std::chrono::steady_clock::now();
                KeypointData fresh_corners = keypoint_pipeline_.detect_frame(frame);
                auto t1 = std::chrono::steady_clock::now();
                total_corner_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

                int visible = 0;
                for (const auto& c : fresh_corners.corners) {
                    if (c.confidence >= KEYPOINT_CONFIDENCE) visible++;
                }
                if (visible >= MIN_CORNERS_FOR_HOMOGRAPHY || cached_corners.empty()) {
                    cached_corners = std::move(fresh_corners);
                    frames_since_corner = 0;
                } else {
                    frames_since_corner++;
                }
            } else {
                frames_since_corner++;
            }

            TrackingTiming tt;
            Detections players, balls, referees;
            tracking_pipeline_.process_frame(frame, players, balls, referees, &tt);

            total_tracker_ms    += tt.tracker_update_ms;
            total_clustering_ms += tt.clustering_ms;
            if (tt.ran_yolo) {
                total_det_preprocess_ms  += tt.det_preprocess_ms;
                total_det_onnx_ms        += tt.det_onnx_run_ms;
                total_det_postprocess_ms += tt.det_postprocess_ms;
                yolo_run_count++;
            }

            tracking_pipeline_.store_tracks(players, balls, referees, frame_idx,
                                             video_player_tracks[frame_idx],
                                             video_player_teams[frame_idx],
                                             video_ball_tracks,
                                             video_referee_tracks[frame_idx]);

            auto t2 = std::chrono::steady_clock::now();
            TacticalMetadata metadata;
            cv::Mat tactical_frame = tactical_pipeline_.process_frame(
                players, balls, referees, cached_corners, metadata);
            auto t3 = std::chrono::steady_clock::now();
            total_tactical_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();

            tactical_frames[frame_idx] = tactical_frame.clone();
            tactical_metadata[frame_idx] = metadata;

            frame_idx++;
            processed_count++;

            auto now = std::chrono::steady_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(now - proc_start).count();

            if (verbose_) {
                if (processed_count % 10 == 0 || processed_count == max_frames || processed_count <= 3) {
                    float fps = 1000.0f * processed_count / static_cast<float>(std::max<int64_t>(16, static_cast<int64_t>(elapsed_ms)));
                    double avg_det_total = (tt.det_preprocess_ms + tt.det_onnx_run_ms + tt.det_postprocess_ms + tt.tracker_update_ms + tt.clustering_ms);
                    std::cout << "  Frame " << frame_idx << "/" << target_total
                              << " | " << std::fixed << std::setprecision(2) << fps << " fps"
                              << " | total det+track: " << avg_det_total << " ms\n"
                              << "    -> Preprocess:   " << tt.det_preprocess_ms << " ms\n"
                              << "    -> YOLO ONNX:    " << tt.det_onnx_run_ms << " ms (" << (tt.ran_yolo ? "YOLO executed" : "Kalman prediction") << ")\n"
                              << "    -> Postprocess:  " << tt.det_postprocess_ms << " ms\n"
                              << "    -> ByteTrack:    " << tt.tracker_update_ms << " ms\n"
                              << "    -> Clustering:   " << tt.clustering_ms << " ms"
                              << std::endl;
                }
            } else {
                print_progress_bar(processed_count, target_total, elapsed_ms, "Pass 1/2 (Tracking): ");
            }
        }

        auto proc_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - proc_start).count();

        std::cout << "\n[Step 5/7] Interpolating ball tracks across video (Exact Soccer_Analysis Parity)..." << std::endl;
        processing_pipeline_.interpolate_ball_tracks(video_ball_tracks, processed_count);

        std::cout << "[Step 6/7] Finalizing output (Pass 2: Overlaying & Compositing onto Video)..." << std::endl;
        reader.open(video_path);

        VideoWriter writer;
        std::ofstream json_file;

        if (json_output) {
            json_file.open(output_path);
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
            writer.open(output_path, reader.width(), reader.height(), OUTPUT_FPS);
        }

        auto annotate_start = std::chrono::steady_clock::now();
        int pass2_idx = 0;
        while (reader.read(frame) && pass2_idx < processed_count) {
            if (json_output) {
                float timestamp = static_cast<float>(pass2_idx) / static_cast<float>(video_fps);
                if (pass2_idx > 0) json_file << ",\n";

                const auto& meta = tactical_metadata[pass2_idx];
                json_file << "    {\n";
                json_file << "      \"frame\": " << pass2_idx << ",\n";
                json_file << "      \"timestamp\": " << json::f(timestamp, 3) << ",\n";
                json_file << "      \"tactical\": {";
                json_file << "\"homography_valid\":" << (meta.transform_valid ? "true" : "false");
                json_file << ",\"num_team1\":" << meta.num_team1;
                json_file << ",\"num_team2\":" << meta.num_team2;
                json_file << "}\n    }";
            } else {
                // Pass exact per-frame track maps to eliminate frozen boxes
                tracking_pipeline_.annotate_frame(frame,
                                                   video_player_tracks[pass2_idx],
                                                   video_player_teams[pass2_idx],
                                                   video_ball_tracks[pass2_idx],
                                                   video_referee_tracks[pass2_idx]);

                cv::Mat output_frame = TacticalPipeline::create_overlay(
                    frame, tactical_frames[pass2_idx], OVERLAY_WIDTH, OVERLAY_HEIGHT);
                writer.write(output_frame);
            }
            pass2_idx++;

            if (!verbose_) {
                auto now = std::chrono::steady_clock::now();
                double ann_elapsed_ms = std::chrono::duration<double, std::milli>(now - annotate_start).count();
                print_progress_bar(pass2_idx, processed_count, ann_elapsed_ms, "Pass 2/2 (Overlay):  ");
            }
        }

        auto annotate_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - annotate_start).count();
        double total_annotate_ms = static_cast<double>(annotate_elapsed);

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
        std::cout << "Pass 1 (Tracking/Clustering) time: " << proc_elapsed << "ms" << std::endl;
        if (!json_output) {
            std::cout << "Pass 2 (Annotation/Overlay) time:  " << annotate_elapsed << "ms" << std::endl;
        }
        if (proc_elapsed > 0) {
            std::cout << "Average Pass 1 FPS: " << (1000.0f * processed_count / proc_elapsed) << std::endl;
        }
        std::cout << "\nPer-frame timing breakdown across " << processed_count << " frames:" << std::endl;
        if (processed_count > 0) {
            double avg_yolo_prep = (yolo_run_count > 0) ? (total_det_preprocess_ms / yolo_run_count) : 0.0;
            double avg_yolo_onnx = (yolo_run_count > 0) ? (total_det_onnx_ms / yolo_run_count) : 0.0;
            double avg_yolo_post = (yolo_run_count > 0) ? (total_det_postprocess_ms / yolo_run_count) : 0.0;
            double avg_track     = total_tracker_ms / processed_count;
            double avg_clus      = total_clustering_ms / processed_count;

            std::cout << "  [Detection + Tracking Sub-steps]:\n"
                      << "    - YOLO Preprocess (letterbox + RGB):  " << (total_det_preprocess_ms / processed_count) << " ms (avg over all frames, " << avg_yolo_prep << " ms/run across " << yolo_run_count << " runs)\n"
                      << "    - YOLO ONNX Run (soccana_object):     " << (total_det_onnx_ms / processed_count) << " ms (avg over all frames, " << avg_yolo_onnx << " ms/run across " << yolo_run_count << " runs)\n"
                      << "    - YOLO Postprocess (8400 anchors):    " << (total_det_postprocess_ms / processed_count) << " ms (avg over all frames, " << avg_yolo_post << " ms/run)\n"
                      << "    - ByteTrack Update (Kalman + IoU):     " << avg_track << " ms\n"
                      << "    - Team Clustering (MobileNetV4):       " << avg_clus << " ms\n"
                      << "  [Other Pipeline Sub-steps]:\n"
                      << "    - Tactical Analysis (TV2Radar):       " << (total_tactical_ms / processed_count) << " ms\n";
            if (!json_output) {
                std::cout << "    - Annotate + Overlay (Pass 2):        " << (total_annotate_ms / processed_count) << " ms\n";
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
    bool verbose_{false};

    KeypointPipeline keypoint_pipeline_;
    TrackingPipeline tracking_pipeline_;
    TacticalPipeline tactical_pipeline_;
    ProcessingPipeline processing_pipeline_;
};

int main(int argc, char** argv) {
    std::cout << "=== Soccer Radar v2.0 ===" << std::endl;
    std::cout << "C++ Soccer Analysis Pipeline (Football-TV2Radar + Soccana Combination)" << std::endl;
    std::cout << "Optimized for constrained devices (Termux/ARM)" << std::endl;
    std::cout << std::endl;

    std::string video_path = "videos/test_720p.mp4";
    std::string output_path = "";
    int max_frames = -1;
    bool verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--video" && i + 1 < argc) {
            video_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--frames" && i + 1 < argc) {
            max_frames = std::stoi(argv[++i]);
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--help") {
            std::cout << "Usage: soccer_radar [options]" << std::endl;
            std::cout << "  --video <path>    Input video path (default: videos/test_720p.mp4)" << std::endl;
            std::cout << "  --output <path>   Output: .json for data, .mp4 for video (default: auto .mp4)" << std::endl;
            std::cout << "  --frames <n>      Max frames to process (-1 for all)" << std::endl;
            std::cout << "  --verbose         Show per-frame diagnostic breakdown instead of progress bar" << std::endl;
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
    std::cout << "  Corner model (TV2Radar): " << keypoint_model << std::endl;
    std::cout << "  Embedding model: " << embedding_model << std::endl;
    std::cout << "  Max frames: " << (max_frames < 0 ? "all" : std::to_string(max_frames)) << std::endl;
    std::cout << "  Verbose timing: " << (verbose ? "ON" : "OFF (Live Progress Bar)") << std::endl;
    std::cout << std::endl;

    CompleteSoccerAnalysisPipeline pipeline(detection_model, keypoint_model, embedding_model, verbose);

    if (!pipeline.analyze_video(video_path, output_path, max_frames)) {
        std::cerr << "Analysis failed!" << std::endl;
        return 1;
    }

    return 0;
}
