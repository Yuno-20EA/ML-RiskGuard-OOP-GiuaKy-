#include "../include/core/matrix.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

// Hàm tạo: Cấp phát bộ nhớ cho ma trận
Matrix::Matrix(int r, int c, double init_val) : rows(r), cols(c) {
    if (r < 0 || c < 0) throw std::invalid_argument("Kích thước ma trận không hợp lệ");
    data_.assign(r * c, init_val);
}

// Truy cập phần tử có kiểm tra biên giới hạn để chống lỗi tràn bộ nhớ (Segmentation fault)
double& Matrix::at(int r, int c) {
    if (r < 0 || r >= rows || c < 0 || c >= cols) throw std::out_of_range("Chỉ số ma trận vượt quá giới hạn");
    return data_[r * cols + c];
}

const double& Matrix::at(int r, int c) const {
    if (r < 0 || r >= rows || c < 0 || c >= cols) throw std::out_of_range("Chỉ số ma trận vượt quá giới hạn");
    return data_[r * cols + c];
}

// Cộng hai ma trận cùng kích thước (Thường dùng khi tính: X * W + b, b chính là bias được cộng vào)
Matrix Matrix::add(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) throw std::invalid_argument("Kích thước không khớp để thực hiện phép cộng");
    Matrix result(rows, cols);
    std::transform(data_.begin(), data_.end(), other.data_.begin(), result.data_.begin(), std::plus<double>());
    return result;
}

// Nhân ma trận (Thuật toán cốt lõi trong Lan truyền tiến: Đầu vào khách hàng * Bảng Trọng số)
Matrix Matrix::multiply(const Matrix& other) const {
    if (cols != other.rows) throw std::invalid_argument("Kích thước không khớp để thực hiện phép nhân ma trận");
    Matrix result(rows, other.cols);
    for (int i = 0; i < rows; ++i) {
        for (int k = 0; k < cols; ++k) {
            double temp = at(i, k);
            for (int j = 0; j < other.cols; ++j) {
                result.at(i, j) += temp * other.at(k, j);
            }
        }
    }
    return result;
}

// Chuyển vị ma trận (Đảo hàng thành cột, rất quan trọng trong tính toán Lan truyền ngược - Backprop)
Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result.at(j, i) = at(i, j);
        }
    }
    return result;
}

// Áp dụng hàm tùy chỉnh (vd: hàm phi tuyến Sigmoid, ReLU) lên toàn bộ phần tử
Matrix Matrix::apply(std::function<double(double)> func) const {
    Matrix result(rows, cols);
    std::transform(data_.begin(), data_.end(), result.data_.begin(), func);
    return result;
}

// Tạo ma trận 0 (Thường dùng làm Bias khởi điểm)
Matrix Matrix::zeros(int r, int c) {
    return Matrix(r, c, 0.0);
}

// Tạo ma trận ngẫu nhiên phân phối chuẩn (Gaussian distribution)
// Mục đích: Phá vỡ tính đối xứng, giúp AI không bị "kẹt" (cùng học một thứ) khi mới bắt đầu học
Matrix Matrix::random(int r, int c) {
    Matrix result(r, c);
    std::mt19937 gen(42); // Seed cố định 42 để kết quả học ổn định khi test
    std::normal_distribution<double> dist(0.0, 1.0);
    std::generate(result.data_.begin(), result.data_.end(), [&](){ return dist(gen); });
    return result;
}
