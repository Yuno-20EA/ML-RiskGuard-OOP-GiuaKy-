#pragma once
// ============================================================
//  RiskGuard ML Framework — Layer.hpp
//  Abstract Base Class cho mọi tầng trong Neural Network.
//  Sử dụng C++20 Concept để ràng buộc kiểu tham số rõ ràng.
// ============================================================

#include <vector>
#include <string>
#include <span>
#include "Matrix.hpp"

namespace riskguard {


// ── Abstract Base Class ───────────────────────────────────────
class Layer {
public:
    // ★ Destructor PHẢI là virtual để đảm bảo gọi đúng destructor
    //   của lớp con khi xóa qua con trỏ Layer*
    virtual ~Layer() = default;

    // ── Interface thuần ảo bắt buộc implement ─────────────────

    /// Forward pass: nhận input, trả về output, lưu cache cho backward
    /// @param input  Ma trận đầu vào (batch_size × in_features)
    /// @return       Ma trận đầu ra (batch_size × out_features)
    [[nodiscard]]
    virtual Matrix forward(const Matrix& input) = 0;

    /// Backward pass: nhận gradient đầu ra, trả về gradient đầu vào
    /// Đồng thời tích lũy gradient cho tham số nội bộ (weights, biases)
    /// @param output_gradient  ∂Loss/∂output từ tầng phía sau
    /// @return                 ∂Loss/∂input để truyền cho tầng phía trước
    [[nodiscard]]
    virtual Matrix backward(const Matrix& output_gradient) = 0;

    /// Trả về danh sách con trỏ đến các tham số học được (weights, biases).
    /// Lớp không có tham số (Sigmoid, ReLU) trả về vector rỗng.
    virtual std::vector<Matrix*> get_parameters() = 0;

    /// Cập nhật tham số bằng Gradient Descent: param -= lr * gradient
    /// @param learning_rate  Tốc độ học
    virtual void update_parameters(double learning_rate) = 0;

    /// Tên định danh của tầng — dùng bởi ModelManager để Serialize
    [[nodiscard]]
    virtual std::string get_type_name() const = 0;

    // ── Interface mặc định (có thể override) ──────────────────

    /// Reset cache giữa các epoch (nếu cần)
    virtual void reset_cache() {}

    /// Chuyển sang chế độ inference (tắt dropout, batch norm, v.v.)
    virtual void set_inference_mode(bool /*inference*/) {}
};

} // namespace riskguard
