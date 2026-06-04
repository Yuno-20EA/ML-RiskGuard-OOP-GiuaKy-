#include "riskguard/utils/DataPipeline.hpp"

namespace riskguard {

// ── Thiết lập tham số chuẩn hóa ────────────────────────────────────────────

void DataPipeline::set_income_params(double mean, double std_dev) {
    income_mean    = mean;
    income_std_dev = (std::abs(std_dev) > EPSILON) ? std_dev : 1.0;
}

void DataPipeline::set_debt_params(double mean, double std_dev) {
    debt_mean    = mean;
    debt_std_dev = (std::abs(std_dev) > EPSILON) ? std_dev : 1.0;
}

void DataPipeline::set_delinquency_params(double mean, double std_dev) {
    delinquency_mean    = mean;
    delinquency_std_dev = (std::abs(std_dev) > EPSILON) ? std_dev : 1.0;
}

void DataPipeline::set_age_params(double mean, double std_dev) {
    age_mean    = mean;
    age_std_dev = (std::abs(std_dev) > EPSILON) ? std_dev : 1.0;
}

// ── Phép Z-score + kẹp biên [-3, 3] (hàm tĩnh nội bộ) ─────────────────────

double DataPipeline::zscore_clip(double value, double mean, double std_dev) {
    double z = (value - mean) / std_dev;
    return std::clamp(z, CLIP_MIN, CLIP_MAX);
}

// ── Hàm biến đổi chính ──────────────────────────────────────────────────────

std::vector<double> DataPipeline::transform(double income, double debt,
                                             double delinquency, double age) const {
    std::vector<double> features;
    features.reserve(4);

    // Chuẩn hóa Z-score từng đặc trưng và kẹp biên trong [-3.0, 3.0]
    features.push_back(zscore_clip(income,      income_mean,      income_std_dev));
    features.push_back(zscore_clip(debt,        debt_mean,        debt_std_dev));
    features.push_back(zscore_clip(delinquency, delinquency_mean, delinquency_std_dev));
    features.push_back(zscore_clip(age,         age_mean,         age_std_dev));

    return features;
}

} // namespace riskguard
