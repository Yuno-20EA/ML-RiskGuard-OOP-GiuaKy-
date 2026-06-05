// ============================================================
//  RiskGuard ML Framework — tests/test_evaluator.cpp
//  Unit Tests cho RiskEvaluator: predict_approval_rate & evaluate_risk_factors
//  (Tách riêng khỏi tests/main.cpp để bảo vệ tính đóng gói test)
// ============================================================
// NOTE: File này được gộp qua Unity Build (#include "test_evaluator.cpp")
//       nên không khai báo lại using namespace hay #include runner.

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
            throw std::runtime_error("Sai loai ngoai le -- ky vong std::invalid_argument");
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

        const std::vector<std::tuple<double,double,double,double>> samples = {
            {70000.0,  15000.0, 0.5,  45.0},   // Khách hàng trung bình
            {10000.0,  90000.0, 80.0, 25.0},   // Rủi ro rất cao
            {200000.0, 500.0,   0.0,  60.0},   // Rủi ro thấp
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
//  Test Case 7 — Kiểm tra đầy đủ 4 nhánh logic XAI
//  Dùng giá trị tài chính thô để kích hoạt từng nhánh từ chối
// ============================================================
class XAIEvaluationTest : public TestCase {
public:
    XAIEvaluationTest() : TestCase("RiskEvaluator", "XAIEvaluationTest") {}
    void run_logic() override {
        // ── Nhánh 1: income = 0, debt > 0 → DTI vô hạn ─────────────────────
        {
            constexpr double raw_income      = 0.0;
            constexpr double raw_debt        = 50000.0;
            constexpr double raw_delinquency = 0.0;
            constexpr double approval_rate   = 0.2;

            std::string reason = RiskEvaluator::evaluate_risk_factors(
                raw_income, raw_debt, raw_delinquency, approval_rate);
            // Kiểm tra ASCII substring tránh lỗi UTF-8 find()
            if (reason.find("DTI") == std::string::npos) {
                throw std::runtime_error(
                    "XAI Branch1: DTI vo han khong duoc phat hien. Output: " + reason);
            }
        }

        // ── Nhánh 2: DTI > 40% (debt/income > 0.4) → vượt ngưỡng an toàn ──
        {
            constexpr double raw_income      = 2000.0;
            constexpr double raw_debt        = 90000.0;   // DTI = 45× >> 0.4
            constexpr double raw_delinquency = 0.0;
            constexpr double approval_rate   = 0.2;

            std::string reason = RiskEvaluator::evaluate_risk_factors(
                raw_income, raw_debt, raw_delinquency, approval_rate);
            if (reason.find("DTI") == std::string::npos &&
                reason.find("40%") == std::string::npos) {
                throw std::runtime_error(
                    "XAI Branch2: DTI qua cao khong duoc phat hien. Output: " + reason);
            }
        }

        // ── Nhánh 3: Trễ hạn nhiều lần → lịch sử tín dụng xấu ──────────────
        {
            constexpr double raw_income      = 100000.0;
            constexpr double raw_debt        = 10000.0;   // DTI = 10% (an toàn)
            constexpr double raw_delinquency = 12.0;      // 12 lần trễ hạn
            constexpr double approval_rate   = 0.2;

            std::string reason = RiskEvaluator::evaluate_risk_factors(
                raw_income, raw_debt, raw_delinquency, approval_rate);
            // "12" xuất hiện trong chuỗi trả về → xác nhận delinquency count đúng
            if (reason.find("12") == std::string::npos) {
                throw std::runtime_error(
                    "XAI Branch3: So lan tre han (12) khong xuat hien. Output: " + reason);
            }
        }

        // ── Nhánh 4: Hồ sơ an toàn (approval_rate >= 0.5) ──────────────────
        {
            constexpr double raw_income      = 200000.0;
            constexpr double raw_debt        = 10000.0;   // DTI = 5%
            constexpr double raw_delinquency = 0.0;
            constexpr double approval_rate   = 0.85;

            std::string reason = RiskEvaluator::evaluate_risk_factors(
                raw_income, raw_debt, raw_delinquency, approval_rate);
            if (reason.find("an to") == std::string::npos) {
                throw std::runtime_error(
                    "XAI Branch4: Ho so tot khong tra ve 'an toan'. Output: " + reason);
            }
        }
    }
};
