#include "../layers/SigmoidLayer.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

// Forward Sigmoid: Ép dải số thực bất kỳ về khoảng [0, 1] 
// VD: Số 5.0 -> ~99% Rủi ro. Số -2.0 -> ~11% Rủi ro
Matrix SigmoidLayer::forward(const Matrix& input) {
    last_output = input.apply([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
    return last_output;
}

// Backward Sigmoid: Tính đạo hàm của Sigmoid để truyền ngược sai số
Matrix SigmoidLayer::backward(const Matrix& grad) {
    if(grad.rows != last_output.rows || grad.cols != last_output.cols) 
        throw std::invalid_argument("Sai lệch kích thước khi truyền ngược qua Sigmoid");
    Matrix d_input(grad.rows, grad.cols);
    for(size_t i=0; i<grad.data_.size(); ++i) {
        double out = last_output.data_[i];
        // Công thức Đạo hàm của Sigmoid: out * (1 - out)
        d_input.data_[i] = grad.data_[i] * out * (1.0 - out);
    }
    return d_input;
}

// Forward ReLU: Hàm kích hoạt phổ biến nhất (Triệt tiêu số âm thành 0)
Matrix ReLULayer::forward(const Matrix& input) {
    last_input = input;
    return input.apply([](double x) { return std::max(0.0, x); });
}

// Backward ReLU: Tính đạo hàm (Nếu input cũ > 0 thì cho lỗi đi qua, nếu <= 0 thì chặn lỗi lại)
Matrix ReLULayer::backward(const Matrix& grad) {
    if(grad.rows != last_input.rows || grad.cols != last_input.cols) 
        throw std::invalid_argument("Sai lệch kích thước khi truyền ngược qua ReLU");
    Matrix d_input(grad.rows, grad.cols);
    for(size_t i=0; i<grad.data_.size(); ++i) {
        d_input.data_[i] = last_input.data_[i] > 0 ? grad.data_[i] : 0.0;
    }
    return d_input;
}
