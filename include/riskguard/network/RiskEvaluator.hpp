#pragma once
#include <vector>
#include <string>
#include "riskguard/network/NeuralNetwork.hpp"

namespace riskguard {

class RiskEvaluator {
public:
    // Dự đoán tỷ lệ duyệt vay cho một khách hàng dựa trên các đặc trưng đã chuẩn hóa
    static double predict_approval_rate(const std::vector<double>& input_features, NeuralNetwork& trained_model);

    // XAI: Sinh lý do từ chối/chấp thuận dựa trên logic nghiệp vụ
    static std::string evaluate_risk_factors(double income, double debt, double delinquency, double approval_rate);
};

} // namespace riskguard
