#include "riskguard/NeuralNetwork.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace riskguard {

void NeuralNetwork::add_layer(std::unique_ptr<Layer> layer) {
    layers.push_back(std::move(layer));
}

Matrix NeuralNetwork::forward(const Matrix& input) {
    Matrix current_output = input;
    for (auto& layer : layers) {
        current_output = layer->forward(current_output);
    }
    return current_output;
}

void NeuralNetwork::backward(const Matrix& output_gradient) {
    Matrix current_gradient = output_gradient;
    // Lan truyền ngược từ tầng cuối cùng lên tầng đầu tiên
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        current_gradient = (*it)->backward(current_gradient);
    }
}

void NeuralNetwork::update_parameters(double learning_rate) {
    for (auto& layer : layers) {
        layer->update_parameters(learning_rate);
    }
}

// 3. Hàm tính lỗi tại tầng cuối cùng sử dụng Binary Cross-Entropy
// Phù hợp cho phân loại nhị phân rủi ro tín dụng (0 hoặc 1)
double NeuralNetwork::calculateBCELoss(const Matrix& predictions, const Matrix& targets) {
    if (predictions.get_rows() != targets.get_rows() || predictions.get_cols() != targets.get_cols()) {
        throw std::runtime_error("BCE Loss Error: Kích thước của predictions và targets không khớp.");
    }

    double total_loss = 0.0;
    int batch_size = predictions.get_rows();
    
    // Sử dụng epsilon siêu nhỏ để tránh tính toán log(0) gây lỗi undefined
    const double epsilon = 1e-15;

    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < predictions.get_cols(); ++j) {
            double y_pred = predictions(i, j);
            double y_true = targets(i, j);

            // Công thức tính lỗi BCE: - (y_true * log(y_pred + epsilon) + (1 - y_true) * log(1 - y_pred + epsilon))
            total_loss -= (y_true * std::log(y_pred + epsilon) + (1.0 - y_true) * std::log(1.0 - y_pred + epsilon));
        }
    }

    // Trả về trung bình lỗi của cả batch
    return total_loss / batch_size;
}

// Phương thức tiện ích tính đạo hàm của hàm lỗi để mồi cho backward propagation
Matrix NeuralNetwork::calculateBCEGradient(const Matrix& predictions, const Matrix& targets) {
    Matrix gradient(predictions.get_rows(), predictions.get_cols());
    const double epsilon = 1e-15;
    int batch_size = predictions.get_rows();

    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < predictions.get_cols(); ++j) {
            double y_pred = predictions(i, j);
            double y_true = targets(i, j);

            y_pred = std::max(epsilon, std::min(1.0 - epsilon, y_pred));

            // Đạo hàm thuần của Binary Cross Entropy dL/dY = (p - y) / (p * (1 - p))
            // Để tránh lỗi chia cho 0 và NaN khi kết hợp với Sigmoid backward, 
            // ta gộp tử số thành (p - y) và chặn mẫu số bằng epsilon.
            double denominator = std::max(y_pred * (1.0 - y_pred), epsilon);
            gradient(i, j) = (y_pred - y_true) / denominator;
            
            // Chia cho độ lớn batch để bình chuẩn hóa gradient
            gradient(i, j) /= batch_size;
        }
    }
    return gradient;
}

} // namespace riskguard
