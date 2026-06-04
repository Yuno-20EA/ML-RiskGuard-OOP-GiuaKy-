#pragma once
#include <vector>
#include <algorithm>
#include <cmath>

namespace riskguard {

class DataPipeline {
public:
    DataPipeline() = default;

    // Các hàm thiết lập tham số chuẩn hóa (mean, std_dev)
    void set_income_params(double mean, double std_dev);
    void set_debt_params(double mean, double std_dev);
    void set_delinquency_params(double mean, double std_dev);
    void set_age_params(double mean, double std_dev);

    // Chuẩn hóa Z-score có kẹp biên trong khoảng [-3.0, 3.0] để chặn điểm ngoại lệ
    std::vector<double> transform(double income, double debt,
                                  double delinquency, double age) const;

private:
    static constexpr double EPSILON        = 1e-7;
    static constexpr double CLIP_MIN       = -3.0;
    static constexpr double CLIP_MAX       =  3.0;

    double income_mean{0.0};       double income_std_dev{1.0};
    double debt_mean{0.0};         double debt_std_dev{1.0};
    double delinquency_mean{0.0};  double delinquency_std_dev{1.0};
    double age_mean{0.0};          double age_std_dev{1.0};

    // Chuẩn hóa Z-score rồi kẹp biên
    static double zscore_clip(double value, double mean, double std_dev);
};

} // namespace riskguard
