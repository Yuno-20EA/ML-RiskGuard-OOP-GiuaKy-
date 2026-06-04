#pragma once
#include <vector>
#include "riskguard/network/NeuralNetwork.hpp"

namespace riskguard {

class RiskEvaluator {
public:
    // Dự đoán tỷ lệ duyệt vay cho một khách hàng dựa trên các đặc trưng đã chuẩn hóa
    static double predict_approval_rate(const std::vector<double>& input_features, NeuralNetwork& trained_model);
};

} // namespace riskguard
