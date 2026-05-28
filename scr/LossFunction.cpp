#include "../layers/LossFunction.hpp"
#include <stdexcept>
#include <cmath>
#include <limits>
#include <algorithm>

// MSELoss: Dành cho bài toán hồi quy (Dự đoán một con số cụ thể, ví dụ hạn mức tín dụng tối đa)
double MSELoss::compute(const Matrix& pred, const Matrix& true_) {
    if (pred.rows != true_.rows || pred.cols != true_.cols)
        throw std::invalid_argument("MSELoss: pred and true dimensions mismatch");
    int n = pred.rows * pred.cols;
    if (n == 0) return std::numeric_limits<double>::quiet_NaN();
    
    double loss = 0.0;
    for (int i = 0; i < n; ++i) {
        double diff = pred.data_[i] - true_.data_[i];
        loss += diff * diff;
    }
    return loss / n;
}

Matrix MSELoss::gradient(const Matrix& pred, const Matrix& true_) {
    if (pred.rows != true_.rows || pred.cols != true_.cols)
        throw std::invalid_argument("MSELoss: pred and true dimensions mismatch");
    int n = pred.rows * pred.cols;
    Matrix grad(pred.rows, pred.cols);
    if (n == 0) return grad;

    for (int i = 0; i < n; ++i) {
        grad.data_[i] = 2.0 * (pred.data_[i] - true_.data_[i]) / n;
    }
    return grad;
}

// BCELoss (Binary Cross Entropy): Dành cho bài toán phân loại nhị phân 
// Rất phù hợp để dự đoán xác suất Nợ xấu (0% - 100%), từ đó kết luận Duyệt hay Từ chối
double BCELoss::compute(const Matrix& pred, const Matrix& true_) {
    if (pred.rows != true_.rows || pred.cols != true_.cols)
        throw std::invalid_argument("BCELoss: pred and true dimensions mismatch");
    int n = pred.rows * pred.cols;
    if (n == 0) return std::numeric_limits<double>::quiet_NaN();
    
    double loss = 0.0;
    for (int i = 0; i < n; ++i) {
        // Giới hạn giá trị p trong khoảng an toàn để tránh log(0) gây lỗi toán học tràn số
        double p = std::max(epsilon, std::min(1.0 - epsilon, pred.data_[i]));
        double t = true_.data_[i]; // Nhãn thực tế: 1 (Bùng nợ) hoặc 0 (Trả tốt)
        loss += - (t * std::log(p) + (1.0 - t) * std::log(1.0 - p));
    }
    return loss / n;
}

Matrix BCELoss::gradient(const Matrix& pred, const Matrix& true_) {
    if (pred.rows != true_.rows || pred.cols != true_.cols)
        throw std::invalid_argument("BCELoss: pred and true dimensions mismatch");
    int n = pred.rows * pred.cols;
    Matrix grad(pred.rows, pred.cols);
    if (n == 0) return grad;

    for (int i = 0; i < n; ++i) {
        double p = std::max(epsilon, std::min(1.0 - epsilon, pred.data_[i]));
        double t = true_.data_[i];
        // Đạo hàm của BCELoss để truyền ngược về trước điều chỉnh trọng số (nhận thức rủi ro)
        grad.data_[i] = ((p - t) / (p * (1.0 - p))) / n;
    }
    return grad;
}
