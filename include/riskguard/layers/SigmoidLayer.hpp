#pragma once
#include "riskguard/core/Layer.hpp"
#include "riskguard/core/Matrix.hpp"
#include <vector>

namespace riskguard {

class SigmoidLayer : public Layer {
public:
    SigmoidLayer() = default;
    
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& output_gradient) override;
    
    std::vector<Matrix*> get_parameters() override { return {}; }
    void update_parameters(double /*learning_rate*/) override {}
    
    std::string get_type_name() const override { return "Sigmoid"; }

private:
    Matrix last_output; // Cache lại output để dùng cho backward pass
};

} // namespace riskguard
