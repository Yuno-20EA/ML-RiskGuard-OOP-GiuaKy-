#include "riskguard/utils/DataPipeline.hpp"
#include <cmath>

namespace riskguard {

// Ngưỡng an toàn để chống lỗi chia cho 0 hoặc chia số quá nhỏ (Numerical Instability)
constexpr double EPSILON = 1e-7;

void DataPipeline::set_age_params(double mean, double std_dev) {
    age_mean = mean;
    age_std_dev = (std::abs(std_dev) > EPSILON) ? std_dev : 1.0;
}

void DataPipeline::set_income_params(double mean, double std_dev) {
    income_mean = mean;
    income_std_dev = (std::abs(std_dev) > EPSILON) ? std_dev : 1.0;
}

void DataPipeline::set_credit_score_params(double mean, double std_dev) {
    credit_score_mean = mean;
    credit_score_std_dev = (std::abs(std_dev) > EPSILON) ? std_dev : 1.0;
}

void DataPipeline::set_job_status_mapping(const std::unordered_map<std::string, double>& mapping) {
    job_status_map = mapping;
}

std::vector<double> DataPipeline::transform(const Customer& customer) const {
    std::vector<double> features;
    // Tối ưu bộ nhớ: cấp phát trước 4 không gian để tránh việc cấp phát lại
    features.reserve(4);

    // Chuẩn hóa Z-score cho các biến liên tục
    features.push_back((customer.get_age() - age_mean) / age_std_dev);
    features.push_back((customer.get_income() - income_mean) / income_std_dev);
    features.push_back((customer.get_credit_score() - credit_score_mean) / credit_score_std_dev);

    // Mã hóa nhãn (Label encoding) cho biến phân loại
    auto it = job_status_map.find(customer.get_job_status());
    if (it != job_status_map.end()) {
        features.push_back(it->second);
    } else {
        // Giá trị dự phòng cho các giá trị phân loại không xác định (Out-of-Vocabulary)
        features.push_back(-1.0);
    }

    return features;
}

} // namespace riskguard
