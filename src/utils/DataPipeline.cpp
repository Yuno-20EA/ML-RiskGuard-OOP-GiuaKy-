#include "riskguard/utils/DataPipeline.hpp"

namespace riskguard {

void DataPipeline::set_income_params(double mean, double std_dev) {
    income_mean = mean;
    income_std_dev = (std::abs(std_dev) > EPSILON) ? std_dev : 1.0;
}

void DataPipeline::set_debt_params(double mean, double std_dev) {
    debt_mean = mean;
    debt_std_dev = (std::abs(std_dev) > EPSILON) ? std_dev : 1.0;
}

void DataPipeline::set_delinquency_params(double mean, double std_dev) {
    delinquency_mean = mean;
    delinquency_dev = (std::abs(std_dev) > EPSILON) ? std_dev : 1.0;
}

void DataPipeline::set_age_params(double mean, double std_dev) {
    age_mean = mean;
    age_std_dev = (std::abs(std_dev) > EPSILON) ? std_dev : 1.0;
}

double DataPipeline::zscore_clip(double value, double mean, double std_dev) {
    double z = (value - mean) / std_dev;
    return std::clamp(z, CLIP_MIN, CLIP_MAX);
}

void DataPipeline::fit(const Matrix& raw_data) {
    int rows = raw_data.get_rows();
    int cols = raw_data.get_cols();
    if (rows == 0 || cols < 4) return;

    double sum[4] = {0.0};
    for (int i = 0; i < rows; ++i) {
        sum[0] += raw_data(i, 0); // Income
        sum[1] += raw_data(i, 1); // Debt
        sum[2] += raw_data(i, 2); // Delinquency
        sum[3] += raw_data(i, 3); // Age
    }

    double mean[4];
    for (int j = 0; j < 4; ++j) {
        mean[j] = sum[j] / rows;
    }

    double var_sum[4] = {0.0};
    for (int i = 0; i < rows; ++i) {
        var_sum[0] += std::pow(raw_data(i, 0) - mean[0], 2);
        var_sum[1] += std::pow(raw_data(i, 1) - mean[1], 2);
        var_sum[2] += std::pow(raw_data(i, 2) - mean[2], 2);
        var_sum[3] += std::pow(raw_data(i, 3) - mean[3], 2);
    }

    set_income_params(mean[0], std::sqrt(var_sum[0] / rows));
    set_debt_params(mean[1], std::sqrt(var_sum[1] / rows));
    set_delinquency_params(mean[2], std::sqrt(var_sum[2] / rows));
    set_age_params(mean[3], std::sqrt(var_sum[3] / rows));
}

std::vector<double> DataPipeline::transform(double income, double debt,
                                            double delinquency, double age) const {
    std::vector<double> norm_features;
    norm_features.reserve(4);

    norm_features.push_back(zscore_clip(income, income_mean, income_std_dev));
    norm_features.push_back(zscore_clip(debt, debt_mean, debt_std_dev));
    norm_features.push_back(zscore_clip(delinquency, delinquency_mean, delinquency_dev));
    norm_features.push_back(zscore_clip(age, age_mean, age_std_dev));

    return norm_features;
}

std::vector<double> DataPipeline::transform(const Customer& customer) const {
    std::vector<double> features;
    features.reserve(4);
    features.push_back((customer.get_income() - income_mean) / income_std_dev);
    features.push_back((customer.get_debt() - debt_mean) / debt_std_dev);
    features.push_back((customer.get_delinquency() - delinquency_mean) / delinquency_dev);
    features.push_back((customer.get_age() - age_mean) / age_std_dev);
    return features;
}

Matrix DataPipeline::transform(const Matrix& raw_data) const {
    int rows = raw_data.get_rows();
    int cols = raw_data.get_cols();
    Matrix result(rows, cols);

    for (int i = 0; i < rows; ++i) {
        if (cols > 0) result(i, 0) = zscore_clip(raw_data(i, 0), income_mean, income_std_dev);
        if (cols > 1) result(i, 1) = zscore_clip(raw_data(i, 1), debt_mean, debt_std_dev);
        if (cols > 2) result(i, 2) = zscore_clip(raw_data(i, 2), delinquency_mean, delinquency_dev);
        if (cols > 3) result(i, 3) = zscore_clip(raw_data(i, 3), age_mean, age_std_dev);
        
        // Sao chép nguyên si các cột nhãn (nếu có)
        for (int j = 4; j < cols; ++j) {
            result(i, j) = raw_data(i, j);
        }
    }
    return result;
}

} // namespace riskguard
