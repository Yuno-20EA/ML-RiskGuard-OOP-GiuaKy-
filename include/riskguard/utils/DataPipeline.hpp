#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include "riskguard/core/Matrix.hpp"

namespace riskguard {

class DataPipeline {
public:
    DataPipeline() = default;

    // Tự động tính toán Mean và StdDev từ ma trận dữ liệu thô (fit)
    // Giả định 4 cột đầu tiên là: Income, Debt, Delinquency, Age
    void fit(const Matrix& raw_data);

    // Các hàm thiết lập tham số thủ công (dành cho test case hoặc khi đã biết trước phân phối)
    void set_income_params(double mean, double std_dev);
    void set_debt_params(double mean, double std_dev);
    void set_delinquency_params(double mean, double std_dev);
    void set_age_params(double mean, double std_dev);

    // Lấy tham số phục vụ cho ModelManager
    double get_income_mean() const { return income_mean; }
    double get_income_std_dev() const { return income_std_dev; }
    double get_debt_mean() const { return debt_mean; }
    double get_debt_std_dev() const { return debt_std_dev; }
    double get_delinquency_mean() const { return delinquency_mean; }
    double get_delinquency_std_dev() const { return delinquency_std_dev; }
    double get_age_mean() const { return age_mean; }
    double get_age_std_dev() const { return age_std_dev; }

    // Chuẩn hóa Z-score có kẹp biên trong khoảng [-3.0, 3.0]
    // 1. Cho một khách hàng đơn lẻ
    std::vector<double> transform(double income, double debt,
                                  double delinquency, double age) const;
    
    // 2. Cho toàn bộ tập dữ liệu (training)
    // Chỉ chuẩn hóa 4 cột đầu, giữ nguyên cột nhãn (nếu có)
    Matrix transform(const Matrix& raw_data) const;

private:
    static constexpr double EPSILON        = 1e-7;
    static constexpr double CLIP_MIN       = -3.0;
    static constexpr double CLIP_MAX       =  3.0;

    double income_mean{0.0};       double income_std_dev{1.0};
    double debt_mean{0.0};         double debt_std_dev{1.0};
    double delinquency_mean{0.0};  double delinquency_std_dev{1.0};
    double age_mean{0.0};          double age_std_dev{1.0};

    // Hàm nội bộ tính z-score và kẹp biên
    static double zscore_clip(double value, double mean, double std_dev);
};

} // namespace riskguard
