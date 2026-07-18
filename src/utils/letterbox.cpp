#include "utils/letterbox.hpp"
#include "utils/constants.hpp"
#include <cstring>
#include <stdexcept>

namespace soccer_radar {

void letterbox_frame(const cv::Mat& src, cv::Mat& dst,
                     int target_w, int target_h,
                     int pad_top, int pad_bottom,
                     int pad_left, int pad_right) {
    // For our specific case: 1280x720 -> 1280x736. Fail explicitly rather
    // than writing outside the destination if an unsupported resolution is supplied.
    if (src.empty() || src.cols + pad_left + pad_right != target_w ||
        src.rows + pad_top + pad_bottom != target_h) {
        throw std::invalid_argument("letterbox_frame: source dimensions and padding do not match model input");
    }

    if (dst.rows != target_h || dst.cols != target_w || dst.type() != src.type()) {
        dst.create(target_h, target_w, src.type());
    }

    // Fill with gray (114, 114, 114) - standard letterbox color
    dst.setTo(cv::Scalar(114, 114, 114));

    // Copy source into the padded region
    // Since we're only padding vertically (pad_left = pad_right = 0),
    // we can do a direct row copy which is very fast
    if (pad_left == 0 && pad_right == 0 && src.cols == target_w) {
        // Fast path: direct memory copy of rows
        std::memcpy(dst.ptr(pad_top), src.ptr(0),
                    static_cast<size_t>(src.rows) * src.cols * src.elemSize());
    } else {
        // General path: use ROI copy
        cv::Mat roi = dst(cv::Rect(pad_left, pad_top, src.cols, src.rows));
        src.copyTo(roi);
    }
}

void unletterbox_bbox(float& x1, float& y1, float& x2, float& y2,
                      int pad_left, int pad_top) {
    x1 -= static_cast<float>(pad_left);
    y1 -= static_cast<float>(pad_top);
    x2 -= static_cast<float>(pad_left);
    y2 -= static_cast<float>(pad_top);
}

} // namespace soccer_radar
