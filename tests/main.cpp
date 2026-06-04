// ============================================================
//  RiskGuard ML Framework — tests/main.cpp
//  Bộ Unit Test tự động bảo vệ số học hệ thống
// ============================================================
#include "test_runner.hpp"
#include "logger.hpp"

// Unity Build: gộp tất cả test case vào một translation unit
#include "test_matrix.cpp"
#include "test_layers.cpp"
#include "test_csv_reader.cpp"

// ── Thêm include riêng cho test DataPipeline & RiskEvaluator ────────────────
#include "riskguard/utils/DataPipeline.hpp"
#include "riskguard/network/RiskEvaluator.hpp"
#include "riskguard/network/NeuralNetwork.hpp"
#include "riskguard/layers/LinearLayer.hpp"
#include "riskguard/layers/SigmoidLayer.hpp"
#include <cassert>
#include <stdexcept>

using namespace riskguard;

// ============================================================
//  Test Case 1: Kiểm thử tính đúng đắn của Z-score
// ============================================================
class ZScoreNormalizationTest : public TestCase {
public:
    ZScoreNormalizationTest() : TestCase("DataPipeline", "ZScoreNormalizationTest") {}

    void run_logic() override {
        DataPipeline pipeline;
        // Thiết lập tham số cố định từ phân phối dataset.csv
        pipeline.set_income_params(70000.0, 30000.0);
        pipeline.set_debt_params(15000.0, 10000.0);
        pipeline.set_delinquency_params(0.5, 0.5);
        pipeline.set_age_params(45.0, 15.0);

        // Mẫu dữ liệu lý tưởng nằm chính xác tại mean → Z-score phải = 0.0
        auto features = pipeline.transform(70000.0, 15000.0, 0.5, 45.0);

        OOP_ASSERT_EQ(static_cast<int>(features.size()), 4);
        OOP_ASSERT_NEAR(features[0], 0.0, 1e-9); // Income z = 0
        OOP_ASSERT_NEAR(features[1], 0.0, 1e-9); // Debt z = 0
        OOP_ASSERT_NEAR(features[2], 0.0, 1e-9); // Delinquency z = 0
        OOP_ASSERT_NEAR(features[3], 0.0, 1e-9); // Age z = 0
    }
};

// ============================================================
//  Test Case 2: Kiểm thử kẹp biên (Clipping) giá trị ngoại lệ
// ============================================================
class OutlierClippingTest : public TestCase {
public:
    OutlierClippingTest() : TestCase("DataPipeline", "OutlierClippingTest") {}

    void run_logic() override {
        DataPipeline pipeline;
        pipeline.set_income_params(70000.0, 30000.0);
        pipeline.set_debt_params(15000.0, 10000.0);
        pipeline.set_delinquency_params(0.5, 0.5);
        pipeline.set_age_params(45.0, 15.0);

        // Thu nhập cực đại vô lý: 10 tỷ → Z-score thô = (1e10 - 70000) / 30000 ≈ 333.000
        // Sau khi clamp, đặc trưng income phải được kẹp cứng về 3.0
        auto features = pipeline.transform(10'000'000'000.0, 15000.0, 0.5, 45.0);

        OOP_ASSERT_NEAR(features[0], 3.0, 1e-9); // income phải bị kẹp tại 3.0
        OOP_ASSERT_NEAR(features[1], 0.0, 1e-9); // debt vẫn bình thường
    }
};

// ============================================================
//  Test Case 3: Kiểm thử ngoại lệ RiskEvaluator (vector rỗng)
// ============================================================
class EmptyFeatureVectorTest : public TestCase {
public:
    EmptyFeatureVectorTest() : TestCase("RiskEvaluator", "EmptyFeatureVectorTest") {}

    void run_logic() override {
        NeuralNetwork model;
        model.add_layer(std::make_unique<LinearLayer>(4, 8));
        model.add_layer(std::make_unique<SigmoidLayer>());
        model.add_layer(std::make_unique<LinearLayer>(8, 1));
        model.add_layer(std::make_unique<SigmoidLayer>());

        std::vector<double> empty_features;
        bool threw_correct_exception = false;

        try {
            RiskEvaluator::predict_approval_rate(empty_features, model);
        } catch (const std::invalid_argument&) {
            // Phải bắt đúng kiểu ngoại lệ std::invalid_argument
            threw_correct_exception = true;
        } catch (...) {
            throw std::runtime_error("Sai loại ngoại lệ — kỳ vọng std::invalid_argument");
        }

        if (!threw_correct_exception) {
            throw std::runtime_error("RiskEvaluator không ném ngoại lệ khi nhận vector rỗng");
        }
    }
};

// ── Hàm main: đăng ký toàn bộ test cases và kích hoạt runner ────────────────
int main() {
    Logger::log_info("Hệ thống kiểm thử tự động bắt đầu khởi tạo...");

    TestRunner& runner = TestRunner::get_instance();

    // Nhóm 1: Kiểm thử Ma Trận
    runner.register_test(std::make_unique<MatrixMultiplicationTest>());
    runner.register_test(std::make_unique<MatrixBroadcastingTest>());
    runner.register_test(std::make_unique<MatrixTransposeTest>());

    // Nhóm 2: Kiểm thử Logic AI (Layers & Loss)
    runner.register_test(std::make_unique<SigmoidActivationTest>());
    runner.register_test(std::make_unique<LinearForwardTest>());
    runner.register_test(std::make_unique<BCELossGradientTest>());

    // Nhóm 3: Kiểm thử Xử lý Dữ liệu (CSV)
    runner.register_test(std::make_unique<FileNotFoundTest>());
    runner.register_test(std::make_unique<MinMaxScalerTest>());

    // Nhóm 4: Kiểm thử Pipeline Số học & RiskEvaluator
    runner.register_test(std::make_unique<ZScoreNormalizationTest>());
    runner.register_test(std::make_unique<OutlierClippingTest>());
    runner.register_test(std::make_unique<EmptyFeatureVectorTest>());

    return runner.run_all();
}
