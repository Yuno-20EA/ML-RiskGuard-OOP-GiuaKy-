#pragma once
#include "riskguard/core/Layer.hpp"
#include "riskguard/core/Matrix.hpp"
#include <vector>

namespace riskguard {

class LinearLayer : public Layer {
public:
    LinearLayer(int input_dim, int output_dim);
    
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& output_gradient) override;
    
    std::vector<Matrix*> get_parameters() override;
    
    // Yêu cầu: Phương thức cập nhật trọng số
    void updateWeights(double learning_rate);
    void update_parameters(double learning_rate) override { updateWeights(learning_rate); }
    
    // Kỹ thuật Gradient Clipping
    void clipGradients(double max_norm);
    
    std::string get_type_name() const override { return "Linear"; }

private:
    Matrix weights;
    Matrix biases;
    Matrix weights_gradient;
    Matrix biases_gradient;
    Matrix last_input; // Lưu lại input để phục vụ backpropagation
};

} // namespace riskguard
