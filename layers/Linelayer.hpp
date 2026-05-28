#pragma once
#include "../include/core/layer.hpp"

// Lớp Tuyến Tính (Linear Layer / Fully Connected Layer)
// Đóng vai trò là các "khớp nối" lưu trữ "Trí nhớ" (Weights & Biases) của AI
class LinearLayer : public Layer {
private:
    Matrix weights;     // Trọng số: Mức độ quan trọng của từng yếu tố (VD: Mức nợ xấu quan trọng hơn Độ tuổi)
    Matrix biases;      // Độ lệch: Hệ số điều chỉnh độ nhạy chung, đóng vai trò như mức cơ sở
    Matrix d_weights;   // Đạo hàm của trọng số (để AI biết cần chỉnh trọng số tăng hay giảm)
    Matrix d_biases;    // Đạo hàm của độ lệch
    Matrix last_input;  // Lưu lại lịch sử hồ sơ đầu vào để phục vụ truyền ngược (Backpropagation)

public:
    // Khởi tạo kích thước (Số lượng đặc trưng đầu vào -> Số lượng đặc trưng đầu ra)
    LinearLayer(int in_features, int out_features);
    
    // Tính toán tỷ lệ rủi ro qua một lớp: Đầu ra = Đầu vào * Trọng số + Độ lệch
    Matrix forward(const Matrix& input) override;
    
    // Tính đạo hàm sai số để tìm cách "sửa chữa" lớp này và truyền lỗi về lớp trước đó
    Matrix backward(const Matrix& grad) override;
    
    // Lấy Weights và Biases để ModelManager xuất ra file (Lưu kiến thức)
    std::vector<Matrix*> get_parameters() override;
    
    // Quá trình Học tập: Cập nhật lại Weights/Biases dựa trên hệ số học (learning rate)
    void update_parameters(double lr) override;
    
    std::string get_type_name() const override { return "LinearLayer"; }
};
