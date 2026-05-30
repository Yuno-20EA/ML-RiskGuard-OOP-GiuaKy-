#include "riskguard/layers/SigmoidLayer.hpp"
#include <cmath>
#include <algorithm>

namespace riskguard {

Matrix SigmoidLayer::forward(const Matrix& input) {
    Matrix output(input.get_rows(), input.get_cols());
    
    for (size_t i = 0; i < input.get_data().size(); ++i) {
        double x = input.get_data()[i];
        
        // 1. Tối ưu hàm kích hoạt Sigmoid (Ép biên - Clamping)
        // Tránh lỗi bùng nổ hàm mũ (Numerical Overflow) khi x quá âm
        x = std::clamp(x, -15.0, 15.0);
        
        output.get_data()[i] = 1.0 / (1.0 + std::exp(-x));
    }
    
    last_output = output;
    return output;
}

Matrix SigmoidLayer::backward(const Matrix& output_gradient) {
    Matrix input_gradient(output_gradient.get_rows(), output_gradient.get_cols());
    
    for (size_t i = 0; i < output_gradient.get_data().size(); ++i) {
        double sigmoid_x = last_output.get_data()[i];
        // Đạo hàm Sigmoid: S(x) * (1 - S(x))
        double grad = sigmoid_x * (1.0 - sigmoid_x);
        
        input_gradient.get_data()[i] = output_gradient.get_data()[i] * grad;
    }
    
    return input_gradient;
}

} // namespace riskguard
