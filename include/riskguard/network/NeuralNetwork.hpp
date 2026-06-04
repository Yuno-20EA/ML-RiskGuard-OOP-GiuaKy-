#pragma once
#include "riskguard/core/Layer.hpp"
#include "riskguard/core/Matrix.hpp"
#include <vector>
#include <memory>

namespace riskguard {

class NeuralNetwork {
public:
    NeuralNetwork() = default;

    void add_layer(std::unique_ptr<Layer> layer);
    
    Matrix forward(const Matrix& input);
    void backward(const Matrix& output_gradient);
    void update_parameters(double learning_rate);

    // Hàm tính lỗi và đạo hàm lỗi (Binary Cross-Entropy)
    double calculateBCELoss(const Matrix& predictions, const Matrix& targets);
    Matrix calculateBCEGradient(const Matrix& predictions, const Matrix& targets);

    // Trích xuất trọng số của lớp mạng đầu tiên để phục vụ XAI
    std::vector<Matrix*> get_first_layer_parameters() const;

    // Trích xuất danh sách lớp mạng để phục vụ ModelManager
    const std::vector<std::unique_ptr<Layer>>& get_layers() const { return layers; }

private:
    std::vector<std::unique_ptr<Layer>> layers;
};

} // namespace riskguard
