#pragma once
// ============================================================
//  RiskGuard ML Framework — Matrix.hpp
//  Core math layer optimized for C++20
// ============================================================

#include <vector>
#include <span>
#include <stdexcept>
#include <string>

namespace riskguard {

class Matrix {
public:
    // ── Constructors ──────────────────────────────────────────
    Matrix() : rows(0), cols(0) {}
    Matrix(int r, int c, double init_val = 0.0);

    // ── Accessors ─────────────────────────────────────────────
    int get_rows() const { return rows; }
    int get_cols() const { return cols; }
    int size() const { return rows * cols; }

    // Truy cập phần tử (r * cols + c)
    double& operator()(int r, int c);
    const double& operator()(int r, int c) const;

    // Yêu cầu 2: Lấy một hàng dữ liệu trả về std::span (C++20)
    std::span<double> get_row(int r);
    std::span<const double> get_row(int r) const;

    // ── Yêu cầu 3: Các phép toán cần phải có ──────────────────
    Matrix multiply(const Matrix& other) const;
    void addBias(const Matrix& bias);
    Matrix transpose() const;
    Matrix elementwiseMultiply(const Matrix& other) const;

    // Tiện ích
    std::vector<double>& get_data() { return data; }
    const std::vector<double>& get_data() const { return data; }

private:
    int rows;
    int cols;
    // Yêu cầu 1: Mảng phẳng
    std::vector<double> data;
};

} // namespace riskguard
