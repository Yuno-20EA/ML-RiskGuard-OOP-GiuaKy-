#include "riskguard/network/RiskEvaluator.hpp"
#include "riskguard/core/Matrix.hpp"
#include <stdexcept>
#include <string>

namespace riskguard {

double RiskEvaluator::predict_approval_rate(const std::vector<double>& input_features, NeuralNetwork& trained_model) {
    if (input_features.empty()) {
        throw std::invalid_argument("Input features vector cannot be empty.");
    }

    Matrix input_matrix(1, static_cast<int>(input_features.size()));
    for (size_t i = 0; i < input_features.size(); ++i) {
        input_matrix(0, static_cast<int>(i)) = input_features[i];
    }

    Matrix output_matrix = trained_model.forward(input_matrix);

    if (output_matrix.get_rows() != 1 || output_matrix.get_cols() != 1) {
        throw std::runtime_error("Unexpected output dimensions from NeuralNetwork forward pass.");
    }

    return 1.0 - output_matrix(0, 0);
}

std::string RiskEvaluator::evaluate_risk_factors(double income, double debt, double delinquency, double approval_rate) {
    if (approval_rate >= 0.5) {
        return "Hồ sơ tín dụng an toàn. Các chỉ số đều ở mức tốt.";
    }

    if (income <= 1e-7 && debt > 0) {
        return "Từ chối: Khách hàng có dư nợ nhưng thu nhập bằng 0 (Tỷ lệ nợ DTI vô hạn).";
    }

    if (income > 0 && (debt / income) > 0.4) {
        return "Từ chối: Tỷ lệ nợ trên thu nhập (DTI) vượt ngưỡng an toàn (> 40%).";
    }

    if (delinquency > 0) {
        return "Từ chối: Lịch sử tín dụng xấu, phát hiện " + std::to_string(static_cast<int>(delinquency)) + " lần trễ hạn thanh toán.";
    }

    return "Từ chối: Mô hình AI đánh giá rủi ro mặc định nợ cao.";
}

} // namespace riskguard
