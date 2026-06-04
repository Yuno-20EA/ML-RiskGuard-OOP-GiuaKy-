#include "riskguard/network/RiskEvaluator.hpp"
#include "riskguard/core/Matrix.hpp"
#include <stdexcept>
#include <algorithm>
#include <iostream>

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

    return output_matrix(0, 0);
}

std::string RiskEvaluator::evaluate_risk_factors(const std::vector<double>& normalized_features, const NeuralNetwork& trained_model, double risk_prob) {
    if (risk_prob < 0.5) {
        return "Hồ sơ tín dụng an toàn. Các chỉ số đều ở mức tốt.";
    }

    auto params = trained_model.get_first_layer_parameters();
    if (params.empty() || params[0]->get_rows() != 4) {
        return "Hệ thống phát hiện rủi ro hỗn hợp cao.";
    }

    Matrix* weights = params[0]; // Ma trận (4 x Output_Dim)
    
    // Tính điểm đóng góp (Contribution Score) cho 4 thuộc tính
    // Income (0), Debt (1), Delinquency (2), Age (3)
    double max_contribution = -1e9;
    int max_index = -1;

    for (int i = 0; i < 4; ++i) {
        double feature_val = normalized_features[i];
        double weight_sum = 0.0;
        
        // Cộng dồn trọng số của thuộc tính i tới tất cả các neuron lớp tiếp theo
        for (int j = 0; j < weights->get_cols(); ++j) {
            weight_sum += (*weights)(i, j);
        }

        double contribution = feature_val * weight_sum;
        // Chúng ta muốn tìm yếu tố có lực tác động lớn nhất (dương lớn nhất)
        if (contribution > max_contribution) {
            max_contribution = contribution;
            max_index = i;
        }
    }

    switch (max_index) {
        case 0:
            return "Thu nhập không ổn định hoặc thấp hơn chuẩn an toàn.";
        case 1:
            return "Dư nợ hiện tại quá cao, tạo áp lực trả nợ lớn.";
        case 2:
            return "Lịch sử tín dụng xấu, có nhiều lần trễ hạn thanh toán.";
        case 3:
            return "Rủi ro liên quan đến độ tuổi theo dữ liệu thống kê.";
        default:
            return "Rủi ro tín dụng cao do nhiều yếu tố kết hợp.";
    }
}

} // namespace riskguard
