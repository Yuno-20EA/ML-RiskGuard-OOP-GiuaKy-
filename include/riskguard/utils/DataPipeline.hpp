#pragma once
#include "riskguard/core/Customer.hpp"
#include <vector>
#include <unordered_map>
#include <string>

namespace riskguard {

class DataPipeline {
public:
    DataPipeline() = default;

    // Các hàm thiết lập tham số chuẩn hóa
    void set_age_params(double mean, double std_dev);
    void set_income_params(double mean, double std_dev);
    void set_credit_score_params(double mean, double std_dev);

    // Thiết lập bảng ánh xạ cho biến phân loại
    void set_job_status_mapping(const std::unordered_map<std::string, double>& mapping);

    // Chuyển đổi chính: chuẩn hóa và mã hóa các đặc trưng
    std::vector<double> transform(const Customer& customer) const;

private:
    // Các tham số chuẩn hóa với giá trị mặc định an toàn
    double age_mean{0.0};
    double age_std_dev{1.0};

    double income_mean{0.0};
    double income_std_dev{1.0};

    double credit_score_mean{0.0};
    double credit_score_std_dev{1.0};

    // Bảng ánh xạ nhãn (Label Encoding)
    std::unordered_map<std::string, double> job_status_map;
};

} // namespace riskguard
