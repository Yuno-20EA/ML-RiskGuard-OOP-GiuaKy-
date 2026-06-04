#include "riskguard/layers/LinearLayer.hpp"
#include <stdexcept>
#include <algorithm>
#include <execution>
#include <random>
#include <cmath>

namespace riskguard {

LinearLayer::LinearLayer(int input_dim, int output_dim) 
    : weights(input_dim, output_dim), biases(1, output_dim), 
      weights_gradient(input_dim, output_dim), biases_gradient(1, output_dim) {
    
    // Khởi tạo trọng số ngẫu nhiên theo phân phối chuẩn (Xavier/He style initialization)
    // Giúp phá vỡ tính đối xứng của mạng và hội tụ nhanh hơn.
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Variance = 1.0 / input_dim (Xavier đơn giản)
    double stddev = std::sqrt(1.0 / input_dim);
    std::normal_distribution<double> dist(0.0, stddev);

    auto& w_data = weights.get_data();
    for (double& w : w_data) {
        w = dist(gen);
    }
    
    // Biases thường được khởi tạo bằng 0
    std::fill(biases.get_data().begin(), biases.get_data().end(), 0.0);
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
    // Sử dụng gemm (Zero-copy) để tính in-place không cần khởi tạo ma trận trung gian
    // last_input: (batch, input_dim) -> transposeA = true -> (input_dim, batch)
    // output_gradient: (batch, output_dim)
    // weights_gradient: (input_dim, output_dim)
    // weights_gradient = 0 (khởi tạo về 0 trước khi gemm)
    std::fill(weights_gradient.get_data().begin(), weights_gradient.get_data().end(), 0.0);
    Matrix::gemm(last_input, output_gradient, weights_gradient, true, false);

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
    // Cấp phát trước kết quả để gọi gemm in-place
    Matrix input_gradient(output_gradient.get_rows(), weights.get_rows());
    Matrix::gemm(output_gradient, weights, input_gradient, false, true);
    
    return input_gradient;
}

// 2. Phương thức cập nhật trọng số bằng Gradient Descent
void LinearLayer::updateWeights(double learning_rate) {
    // Thực hiện phép toán weights = weights - learning_rate * weights_gradient
    // Tận dụng std::execution::par_unseq
    std::transform(weights.get_data().begin(), weights.get_data().end(),
                   weights_gradient.get_data().begin(),
                   weights.get_data().begin(),
                   [learning_rate](double w, double gw) { return w - learning_rate * gw; });
    
    // Tương tự cho biases
    std::transform(biases.get_data().begin(), biases.get_data().end(),
                   biases_gradient.get_data().begin(),
                   biases.get_data().begin(),
                   [learning_rate](double b, double gb) { return b - learning_rate * gb; });
}

// 3. Kỹ thuật Gradient Clipping
void LinearLayer::clipGradients(double max_norm) {
    auto clip_func = [max_norm](double& val) {
        val = std::clamp(val, -max_norm, max_norm);
    };

    // Clip weights_gradient
    std::for_each(weights_gradient.get_data().begin(), weights_gradient.get_data().end(),
                  clip_func);
    
    // Clip biases_gradient
    std::for_each(biases_gradient.get_data().begin(), biases_gradient.get_data().end(),
                  clip_func);
}

std::vector<Matrix*> LinearLayer::get_parameters() {
    return {&weights, &biases};
}

} // namespace riskguard
