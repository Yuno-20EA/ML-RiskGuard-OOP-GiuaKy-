#include "riskguard/network/NeuralNetwork.hpp"
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

double NeuralNetwork::calculateBCELoss(const Matrix& predictions, const Matrix& targets) {
    if (predictions.get_rows() != targets.get_rows() || predictions.get_cols() != targets.get_cols()) {
        throw std::runtime_error("BCE Loss Error: Kích thước của predictions và targets không khớp.");
    }

    double total_loss = 0.0;
    int batch_size = predictions.get_rows();
    const double epsilon = 1e-15;

    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < predictions.get_cols(); ++j) {
            double y_pred = predictions(i, j);
            double y_true = targets(i, j);
            total_loss -= (y_true * std::log(y_pred + epsilon) + (1.0 - y_true) * std::log(1.0 - y_pred + epsilon));
        }
    }
    return total_loss / batch_size;
}

Matrix NeuralNetwork::calculateBCEGradient(const Matrix& predictions, const Matrix& targets) {
    Matrix gradient(predictions.get_rows(), predictions.get_cols());
    const double epsilon = 1e-15;
    int batch_size = predictions.get_rows();

    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < predictions.get_cols(); ++j) {
            double y_pred = predictions(i, j);
            double y_true = targets(i, j);

            y_pred = std::max(epsilon, std::min(1.0 - epsilon, y_pred));
            // Đạo hàm BCE chuẩn: (pred - true) / (pred * (1 - pred))
            // Không chia batch_size lần 2 — Loss đã chia rồi
            double denominator = std::max(y_pred * (1.0 - y_pred), epsilon);
            gradient(i, j) = (y_pred - y_true) / (denominator * batch_size);
        }
    }
    return gradient;
}

std::vector<Matrix*> NeuralNetwork::get_first_layer_parameters() const {
    for (const auto& layer : layers) {
        auto params = layer->get_parameters();
        if (!params.empty()) {
            return params; // Trả về tham số của lớp học được (Linear) đầu tiên
        }
    }
    return {};
}

} // namespace riskguard
