#pragma once
#include "matrix.hpp"
#include <string>
#include <vector>

// Layer là khuôn mẫu cho các tầng nơ-ron
// Trong bài toán tài chính, các tầng này sẽ học cách bóc tách các đặc trưng (Thu nhập, Khoản nợ, Lịch sử, Độ tuổi)
class Layer {
public:
    virtual ~Layer();
    
    // Lan truyền tiến: Nhận dữ liệu đầu vào (hồ sơ khách hàng) và tính toán ra các đặc trưng/kết quả
    virtual Matrix forward(const Matrix& input) = 0;
    
    // Lan truyền ngược: Nhận sai số từ tầng sau, truyền ngược lại để tính đạo hàm (cập nhật nhận thức của AI)
    virtual Matrix backward(const Matrix& grad) = 0;
    
    // Lấy các tham số (bộ não - weights, biases) để lưu trữ
    virtual std::vector<Matrix*> get_parameters();
    
    // Cập nhật lại kiến thức sau khi biết sai số
    virtual void update_parameters(double lr);
    
    // Tên của tầng để định dạng khi lưu file
    virtual std::string get_type_name() const = 0;
};
