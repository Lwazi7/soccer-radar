#pragma once

#include <opencv2/core.hpp>

namespace soccer_radar {

// Letterbox: pad a 1280x720 image to 1280x736 (8px top + 8px bottom)
// This matches the soccana model export size.
// Uses zero-allocation path when output is pre-allocated.
void letterbox_frame(const cv::Mat& src, cv::Mat& dst,
                     int target_w, int target_h,
                     int pad_top, int pad_bottom,
                     int pad_left, int pad_right);

// Reverse letterbox: convert detection coordinates from model space
// back to original frame space.
void unletterbox_bbox(float& x1, float& y1, float& x2, float& y2,
                      int pad_left, int pad_top);

} // namespace soccer_radar
