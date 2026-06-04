#include "riskguard/network/RiskEvaluator.hpp"
#include "riskguard/core/Matrix.hpp"
#include <stdexcept>

namespace riskguard {

double RiskEvaluator::predict_approval_rate(const std::vector<double>& input_features, NeuralNetwork& trained_model) {
    if (input_features.empty()) {
        throw std::invalid_argument("Input features vector cannot be empty.");
    }

    // Chuyển đổi mảng std::vector<double> (1 x N) sang ma trận để lan truyền xuôi
    Matrix input_matrix(1, static_cast<int>(input_features.size()));
    for (size_t i = 0; i < input_features.size(); ++i) {
        input_matrix(0, static_cast<int>(i)) = input_features[i];
    }

    // Thực hiện lan truyền xuôi qua mạng một cách an toàn bằng tham chiếu không hằng
    Matrix output_matrix = trained_model.forward(input_matrix);

    // Đảm bảo đầu ra là một giá trị vô hướng duy nhất (Ma trận 1x1)
    if (output_matrix.get_rows() != 1 || output_matrix.get_cols() != 1) {
        throw std::runtime_error("Unexpected output dimensions from NeuralNetwork forward pass.");
    }

    // Trả về xác suất duyệt vay cuối cùng
    return output_matrix(0, 0);
}

} // namespace riskguard
