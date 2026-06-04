// ============================================================
//  RiskGuard ML Framework — tests/main.cpp
//  Hệ thống Unit Test tự động (Unity Build Pattern)
// ============================================================
#include "test_runner.hpp"
#include "logger.hpp"

// Unity Build: gộp tất cả các translation unit test vào một file
#include "test_matrix.cpp"
#include "test_layers.cpp"
#include "test_csv_reader.cpp"

// ── Include riêng cho test DataPipeline & RiskEvaluator ─────────────────────
#include "riskguard/utils/DataPipeline.hpp"
#include "riskguard/network/RiskEvaluator.hpp"
#include "riskguard/network/NeuralNetwork.hpp"
#include "riskguard/layers/LinearLayer.hpp"
#include "riskguard/layers/SigmoidLayer.hpp"
#include <cmath>
#include <stdexcept>

using namespace riskguard;

// ── Hàm tiện ích nội bộ: dựng pipeline đã cấu hình sẵn ─────────────────────
static DataPipeline make_configured_pipeline() {
    DataPipeline p;
    p.set_income_params(70000.0, 30000.0);
    p.set_debt_params(15000.0, 10000.0);
    p.set_delinquency_params(0.5, 0.5);
    p.set_age_params(45.0, 15.0);
    return p;
}

// ── Hàm tiện ích nội bộ: dựng mạng nơ-ron 4→8→1 ────────────────────────────
static NeuralNetwork make_network() {
    NeuralNetwork net;
    net.add_layer(std::make_unique<LinearLayer>(4, 8));
    net.add_layer(std::make_unique<SigmoidLayer>());
    net.add_layer(std::make_unique<LinearLayer>(8, 1));
    net.add_layer(std::make_unique<SigmoidLayer>());
    return net;
}

// ============================================================
//  Test Case 1 — Z-score tại điểm mean → đầu ra phải = 0.0
// ============================================================
class ZScoreAtMeanTest : public TestCase {
public:
    ZScoreAtMeanTest() : TestCase("DataPipeline", "ZScoreAtMeanTest") {}
    void run_logic() override {
        auto pipeline = make_configured_pipeline();
        // Đưa vào chính xác các giá trị mean → Z-score = 0.0 trước khi clamp
        auto f = pipeline.transform(70000.0, 15000.0, 0.5, 45.0);
        OOP_ASSERT_EQ(static_cast<int>(f.size()), 4);
        OOP_ASSERT_NEAR(f[0], 0.0, 1e-9);
        OOP_ASSERT_NEAR(f[1], 0.0, 1e-9);
        OOP_ASSERT_NEAR(f[2], 0.0, 1e-9);
        OOP_ASSERT_NEAR(f[3], 0.0, 1e-9);
    }
};

// ============================================================
//  Test Case 2 — Giá trị income cực đại → clamp tại +3.0
// ============================================================
class IncomeOutlierClampTest : public TestCase {
public:
    IncomeOutlierClampTest() : TestCase("DataPipeline", "IncomeOutlierClampTest") {}
    void run_logic() override {
        auto pipeline = make_configured_pipeline();
        // Income = 10 tỷ: Z-score thô ≈ 333,000 → phải bị kẹp về 3.0
        auto f = pipeline.transform(10'000'000'000.0, 15000.0, 0.5, 45.0);
        OOP_ASSERT_NEAR(f[0],  3.0, 1e-9);
        OOP_ASSERT_NEAR(f[1],  0.0, 1e-9);
        OOP_ASSERT_NEAR(f[2],  0.0, 1e-9);
        OOP_ASSERT_NEAR(f[3],  0.0, 1e-9);
    }
};

// ============================================================
//  Test Case 3 — Delinquency cực lớn → clamp tại +3.0
// ============================================================
class DelinquencyOutlierClampTest : public TestCase {
public:
    DelinquencyOutlierClampTest() : TestCase("DataPipeline", "DelinquencyOutlierClampTest") {}
    void run_logic() override {
        auto pipeline = make_configured_pipeline();
        // Delinquency = 100 (cực đại được phép): Z-score = (100 - 0.5) / 0.5 = 199 → clamp 3.0
        auto f = pipeline.transform(70000.0, 15000.0, 100.0, 45.0);
        OOP_ASSERT_NEAR(f[2], 3.0, 1e-9);
    }
};

// ============================================================
//  Test Case 4 — Xác minh Z-score đúng toán học tại 1 std_dev
// ============================================================
class ZScoreOneStdDevTest : public TestCase {
public:
    ZScoreOneStdDevTest() : TestCase("DataPipeline", "ZScoreOneStdDevTest") {}
    void run_logic() override {
        auto pipeline = make_configured_pipeline();
        // Income = 70000 + 30000 = 100000 → Z = 1.0 (nằm trong [-3,3])
        auto f = pipeline.transform(100000.0, 15000.0, 0.5, 45.0);
        OOP_ASSERT_NEAR(f[0], 1.0, 1e-9);
        // Age = 45 - 15 = 30 → Z = -1.0
        auto f2 = pipeline.transform(70000.0, 15000.0, 0.5, 30.0);
        OOP_ASSERT_NEAR(f2[3], -1.0, 1e-9);
    }
};

// ============================================================
//  Test Case 5 — RiskEvaluator ném std::invalid_argument khi vector rỗng
// ============================================================
class EmptyFeatureExceptionTest : public TestCase {
public:
    EmptyFeatureExceptionTest() : TestCase("RiskEvaluator", "EmptyFeatureExceptionTest") {}
    void run_logic() override {
        auto model = make_network();
        std::vector<double> empty;
        bool caught = false;
        try {
            RiskEvaluator::predict_approval_rate(empty, model);
        } catch (const std::invalid_argument&) {
            caught = true;
        } catch (...) {
            throw std::runtime_error("Sai loai ngoai le — ky vong std::invalid_argument");
        }
        if (!caught)
            throw std::runtime_error("RiskEvaluator khong nem ngoai le khi vector rong");
    }
};

// ============================================================
//  Test Case 6 — Đầu ra RiskEvaluator luôn nằm trong [0.0, 1.0]
// ============================================================
class OutputRangeTest : public TestCase {
public:
    OutputRangeTest() : TestCase("RiskEvaluator", "OutputRangeTest") {}
    void run_logic() override {
        auto pipeline = make_configured_pipeline();
        auto model    = make_network();

        // Thử nhiều tổ hợp đặc trưng khác nhau
        const std::vector<std::tuple<double,double,double,double>> samples = {
            {70000.0, 15000.0, 0.5,  45.0},  // Khách hàng trung bình
            {10000.0, 90000.0, 80.0, 25.0},  // Rủi ro rất cao
            {200000.0, 500.0,  0.0,  60.0},  // Rủi ro thấp
        };

        for (const auto& [inc, dbt, del, age] : samples) {
            auto f = pipeline.transform(inc, dbt, del, age);
            double prob = RiskEvaluator::predict_approval_rate(f, model);
            if (prob < 0.0 || prob > 1.0)
                throw std::runtime_error("Xac suat ngoai khoang [0.0, 1.0]: " +
                                         std::to_string(prob));
        }
    }
};

// ============================================================
//  Test Case 7 — Kiểm tra thuật toán giải thích XAI
// ============================================================
class XAIEvaluationTest : public TestCase {
public:
    XAIEvaluationTest() : TestCase("RiskEvaluator", "XAIEvaluationTest") {}
    void run_logic() override {
        auto model = make_network();
        std::vector<double> features = {2.0, 3.0, 1.0, 0.5}; // Đặc trưng giả định
        
        // Cố tình đẩy trọng số lớp đầu tiên của thuộc tính Debt (index 1) lên mức siêu cao
        auto params = model.get_first_layer_parameters();
        if (!params.empty()) {
            Matrix* weights = params[0];
            for (int j = 0; j < weights->get_cols(); ++j) {
                (*weights)(1, j) = 10.0; // Làm cho nợ có lực đóng góp mạnh nhất
                (*weights)(0, j) = 0.1;
                (*weights)(2, j) = 0.1;
                (*weights)(3, j) = 0.1;
            }
        }
        
        std::string reason = RiskEvaluator::evaluate_risk_factors(features, model, 0.8);
        if (reason.find("Dư nợ hiện tại quá cao") == std::string::npos) {
            throw std::runtime_error("XAI không tìm ra nguyên nhân Dư nợ cao. Output: " + reason);
        }
    }
};

// ── Main: Đăng ký toàn bộ Test Cases → Kích hoạt Runner ────────────────────
int main() {
    Logger::log_info("He thong kiem thu tu dong khoi dong...");

    TestRunner& runner = TestRunner::get_instance();

    // Nhóm 1 — Ma Trận
    runner.register_test(std::make_unique<MatrixMultiplicationTest>());
    runner.register_test(std::make_unique<MatrixBroadcastingTest>());
    runner.register_test(std::make_unique<MatrixTransposeTest>());

    // Nhóm 2 — Logic AI (Layers & Loss)
    runner.register_test(std::make_unique<SigmoidActivationTest>());
    runner.register_test(std::make_unique<LinearForwardTest>());
    runner.register_test(std::make_unique<BCELossGradientTest>());

    // Nhóm 3 — Xử lý Dữ liệu CSV
    runner.register_test(std::make_unique<FileNotFoundTest>());
    runner.register_test(std::make_unique<RawCSVLoaderTest>());

    // Nhóm 4 — Pipeline Số học (Z-score & Clipping)
    runner.register_test(std::make_unique<ZScoreAtMeanTest>());
    runner.register_test(std::make_unique<IncomeOutlierClampTest>());
    runner.register_test(std::make_unique<DelinquencyOutlierClampTest>());
    runner.register_test(std::make_unique<ZScoreOneStdDevTest>());

    // Nhóm 5 — RiskEvaluator (Exception & Output Range)
    runner.register_test(std::make_unique<EmptyFeatureExceptionTest>());
    runner.register_test(std::make_unique<OutputRangeTest>());
    runner.register_test(std::make_unique<XAIEvaluationTest>());

    return runner.run_all();
}
