#pragma once
#include "../include/core/layer.hpp"

// Lớp kích hoạt Sigmoid (Dùng để chốt tỷ lệ phần trăm từ 0.0 -> 1.0)
// Rất quan trọng ở lớp Output cuối cùng để ép điểm số rủi ro về giới hạn chuẩn [0%, 100%]
class SigmoidLayer : public Layer {
private:
    Matrix last_output; // Lưu lại kết quả đầu ra
public:
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& grad) override;
    std::string get_type_name() const override { return "SigmoidLayer"; }
};

// Lớp kích hoạt ReLU (Loại bỏ các giá trị âm, giữ nguyên giá trị dương)
// Thường dùng ở các lớp ẩn (Hidden Layers) để giúp mạng nơ-ron tìm ra các quy luật phi tuyến tính phức tạp
class ReLULayer : public Layer {
private:
    Matrix last_input;
public:
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& grad) override;
    std::string get_type_name() const override { return "ReLULayer"; }
};
