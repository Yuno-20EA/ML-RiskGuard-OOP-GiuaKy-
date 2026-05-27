#include "riskguard/layers/LinearLayer.hpp"
#include <stdexcept>

namespace riskguard {

LinearLayer::LinearLayer(int input_dim, int output_dim) 
    : weights(input_dim, output_dim), biases(1, output_dim), 
      weights_gradient(input_dim, output_dim), biases_gradient(1, output_dim) {
    // Lưu ý: Trong thực tế sẽ có hàm khởi tạo Xavier/He ở đây
}

Matrix LinearLayer::forward(const Matrix& input) {
    last_input = input; // Lưu đầu vào cho backward
    Matrix output = input.multiply(weights);
    output.addBias(biases);
    return output;
}

// 1. Thuật toán Lan truyền ngược (Backpropagation) cho LinearLayer
Matrix LinearLayer::backward(const Matrix& output_gradient) {
    // Tính toán weights_gradient = transpose(last_input) * output_gradient
    // Áp dụng luật chuỗi (Chain Rule) cho Y = X*W + b
    weights_gradient = last_input.transpose().multiply(output_gradient);

    // Tính toán biases_gradient = sum_columns(output_gradient)
    // Gom gradient từ tất cả các samples trong batch
    for (int j = 0; j < output_gradient.get_cols(); ++j) {
        double sum = 0.0;
        for (int i = 0; i < output_gradient.get_rows(); ++i) {
            sum += output_gradient(i, j);
        }
        biases_gradient(0, j) = sum;
    }

    // Tính toán input_gradient = output_gradient * transpose(weights) để truyền tiếp về tầng trước
    Matrix input_gradient = output_gradient.multiply(weights.transpose());
    
    return input_gradient;
}

// 2. Phương thức cập nhật trọng số bằng Gradient Descent
void LinearLayer::updateWeights(double learning_rate) {
    // Thực hiện phép toán weights = weights - learning_rate * weights_gradient
    for (size_t i = 0; i < weights.get_data().size(); ++i) {
        weights.get_data()[i] -= learning_rate * weights_gradient.get_data()[i];
    }
    
    // Tương tự cho biases
    for (size_t i = 0; i < biases.get_data().size(); ++i) {
        biases.get_data()[i] -= learning_rate * biases_gradient.get_data()[i];
    }
}

// 3. Kỹ thuật Gradient Clipping
void LinearLayer::clipGradients(double max_norm) {
    // Clip weights_gradient
    for (size_t i = 0; i < weights_gradient.get_data().size(); ++i) {
        double& val = weights_gradient.get_data()[i];
        if (val > max_norm) val = max_norm;
        else if (val < -max_norm) val = -max_norm;
    }
    
    // Clip biases_gradient
    for (size_t i = 0; i < biases_gradient.get_data().size(); ++i) {
        double& val = biases_gradient.get_data()[i];
        if (val > max_norm) val = max_norm;
        else if (val < -max_norm) val = -max_norm;
    }
}

std::vector<Matrix*> LinearLayer::get_parameters() {
    return {&weights, &biases};
}

} // namespace riskguard
