#pragma once
#include <vector>
#include <stdexcept>
#include <span>
#include <functional>
#include <limits>

// Lớp Matrix (Ma trận): Xương sống toán học của hệ thống AI
// Toàn bộ dữ liệu hồ sơ (Thu nhập, nợ...) và bộ não AI (Trọng số) đều được biểu diễn dưới dạng ma trận
class Matrix {
public:
    int rows; // Số hàng (Ví dụ: Số lượng hồ sơ khách hàng/Batch size)
    int cols; // Số cột (Ví dụ: Số lượng chỉ số của mỗi khách hàng)
    std::vector<double> data_; // Mảng 1 chiều lưu trữ dữ liệu để tối ưu tốc độ tính toán

    // Khởi tạo ma trận với giá trị mặc định
    Matrix(int r = 0, int c = 0, double init_val = 0.0);
    
    int get_rows() const { return rows; }
    int get_cols() const { return cols; }
    
    // Truy cập phần tử tại hàng r, cột c một cách an toàn
    double& at(int r, int c);
    const double& at(int r, int c) const;

    // Các phép toán ma trận cơ bản cho mạng nơ-ron
    Matrix add(const Matrix& other) const;      // Cộng ma trận (Dùng để cộng Bias)
    Matrix multiply(const Matrix& other) const; // Nhân ma trận (Tích vô hướng - Tính tổng trọng số)
    Matrix transpose() const;                   // Chuyển vị ma trận (Cần thiết khi truyền ngược đạo hàm)
    Matrix apply(std::function<double(double)> func) const; // Áp dụng hàm toán học lên từng phần tử (Dùng cho Activation)

    // Khởi tạo ma trận đặc biệt
    static Matrix zeros(int r, int c);  // Tạo ma trận toàn số 0 (Khởi tạo Bias)
    static Matrix random(int r, int c); // Tạo ma trận ngẫu nhiên chuẩn tắc (Khởi tạo Weights ban đầu cho AI học)
};
