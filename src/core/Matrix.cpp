#include "riskguard/core/Matrix.hpp"
#include <stdexcept>
#include <execution>
#include <algorithm>

namespace riskguard {

Matrix::Matrix(int r, int c, double init_val) : rows(r), cols(c), data(r * c, init_val) {
    if (r < 0 || c < 0) {
        throw std::runtime_error("Matrix dimensions cannot be negative.");
    }
}

double& Matrix::operator()(int r, int c) {
    if (r < 0 || r >= rows || c < 0 || c >= cols) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    // Yêu cầu 1: Truy cập theo công thức index = r * cols + c
    return data[r * cols + c];
}

const double& Matrix::operator()(int r, int c) const {
    if (r < 0 || r >= rows || c < 0 || c >= cols) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    // Yêu cầu 1: Truy cập theo công thức index = r * cols + c
    return data[r * cols + c];
}

std::span<double> Matrix::get_row(int r) {
    if (r < 0 || r >= rows) {
        throw std::out_of_range("Row index out of bounds");
    }
    // Yêu cầu 2: Trả về std::span thay vì copy vector (C++20)
    return std::span<double>(data.data() + r * cols, cols);
}

std::span<const double> Matrix::get_row(int r) const {
    if (r < 0 || r >= rows) {
        throw std::out_of_range("Row index out of bounds");
    }
    // Yêu cầu 2: Trả về std::span thay vì copy vector (C++20)
    return std::span<const double>(data.data() + r * cols, cols);
}

Matrix Matrix::multiply(const Matrix& other) const {
    // Bắt buộc kiểm tra kích thước cols == other.rows
    if (cols != other.rows) {
        throw std::runtime_error("Matrix multiply error: cols must equal other.rows");
    }

    Matrix result(rows, other.cols, 0.0);
    double* res_ptr = result.data.data();
    const double* a_ptr = this->data.data();
    const double* b_ptr = other.data.data();
    
    // Yêu cầu 1: Tối ưu hóa CPU Cache bằng cách duyệt theo thứ tự i, k, j
    for (int i = 0; i < rows; ++i) {
        for (int k = 0; k < cols; ++k) {
            double temp = a_ptr[i * cols + k];
            for (int j = 0; j < other.cols; ++j) {
                res_ptr[i * other.cols + j] += temp * b_ptr[k * other.cols + j];
            }
        }
    }
    return result;
}

void Matrix::addBias(const Matrix& bias) {
    // Yêu cầu 3: Cộng vector bias vào từng cột của ma trận (Broadcasting)
    if (bias.rows == 1 && bias.cols == cols) {
        // Bias dạng (1 x cols), cộng tương ứng vào tất cả các dòng
        for (int i = 0; i < rows; ++i) {
            auto row_span = get_row(i);
            auto bias_span = bias.get_row(0);
            for (int j = 0; j < cols; ++j) {
                row_span[j] += bias_span[j];
            }
        }
    } else if (bias.cols == 1 && bias.rows == rows) {
        // Bias dạng (rows x 1), cộng giá trị tương ứng của dòng đó
        for (int i = 0; i < rows; ++i) {
            double b_val = bias(i, 0);
            auto row_span = get_row(i);
            for (int j = 0; j < cols; ++j) {
                row_span[j] += b_val;
            }
        }
    } else {
        throw std::runtime_error("addBias error: Invalid bias dimensions");
    }
}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(j, i) = this->operator()(i, j);
        }
    }
    return result;
}

Matrix Matrix::elementwiseMultiply(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::runtime_error("elementwiseMultiply error: dimension mismatch");
    }

    Matrix result(rows, cols);
    // Duyệt qua mảng phẳng 1 chiều (tối ưu nhất cho Hadamard product)
    std::transform(std::execution::par_unseq,
                   data.begin(), data.end(),
                   other.data.begin(),
                   result.data.begin(),
                   std::multiplies<double>());
    return result;
}

void Matrix::gemm(const Matrix& A, const Matrix& B, Matrix& C, bool transA, bool transB) {
    int a_rows = transA ? A.cols : A.rows;
    int a_cols = transA ? A.rows : A.cols;
    int b_rows = transB ? B.cols : B.rows;
    int b_cols = transB ? B.rows : B.cols;

    if (a_cols != b_rows) {
        throw std::runtime_error("Matrix gemm error: inner dimensions must match");
    }
    if (C.rows != a_rows || C.cols != b_cols) {
        throw std::runtime_error("Matrix gemm error: C dimensions must match result dimensions");
    }

    const double* a_ptr = A.data.data();
    const double* b_ptr = B.data.data();
    double* c_ptr = C.data.data();

    // Để tối ưu cho các trường hợp cụ thể của ML (thường là transA hoặc transB)
    if (!transA && !transB) {
        // C = A * B
        for (int i = 0; i < a_rows; ++i) {
            for (int k = 0; k < a_cols; ++k) {
                double temp = a_ptr[i * a_cols + k];
                for (int j = 0; j < b_cols; ++j) {
                    c_ptr[i * b_cols + j] += temp * b_ptr[k * b_cols + j];
                }
            }
        }
    } else if (transA && !transB) {
        // C = A^T * B
        for (int k = 0; k < a_rows; ++k) {
            for (int i = 0; i < a_cols; ++i) {
                double temp = a_ptr[k * a_cols + i];
                for (int j = 0; j < b_cols; ++j) {
                    c_ptr[i * b_cols + j] += temp * b_ptr[k * b_cols + j];
                }
            }
        }
    } else if (!transA && transB) {
        // C = A * B^T
        for (int i = 0; i < a_rows; ++i) {
            for (int j = 0; j < b_cols; ++j) {
                double sum = 0.0;
                for (int k = 0; k < a_cols; ++k) {
                    sum += a_ptr[i * a_cols + k] * b_ptr[j * b_cols + k];
                }
                c_ptr[i * b_cols + j] += sum;
            }
        }
    } else {
        // C = A^T * B^T
        for (int i = 0; i < a_rows; ++i) {
            for (int j = 0; j < b_cols; ++j) {
                double sum = 0.0;
                for (int k = 0; k < a_cols; ++k) {
                    sum += a_ptr[k * a_cols + i] * b_ptr[j * b_cols + k];
                }
                c_ptr[i * b_cols + j] += sum;
            }
        }
    }
}

} // namespace riskguard
