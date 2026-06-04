// ============================================================
//  RiskGuard ML Framework — app/main.cpp
//  Giao diện Cyberpunk CLI tương tác nhập liệu khách hàng
// ============================================================
#include "riskguard/utils/Dashboard.hpp"
#include "riskguard/utils/DataPipeline.hpp"
#include "riskguard/network/NeuralNetwork.hpp"
#include "riskguard/network/RiskEvaluator.hpp"
#include "riskguard/layers/LinearLayer.hpp"
#include "riskguard/layers/SigmoidLayer.hpp"
#include "riskguard/utils/DataLoader.hpp"
#include <iostream>
#include <memory>

using namespace riskguard;

// ── Khởi tạo kiến trúc mạng nơ-ron chuẩn 4→8→1 ────────────────────────────
static NeuralNetwork build_network() {
    NeuralNetwork net;
    net.add_layer(std::make_unique<LinearLayer>(4, 8));
    net.add_layer(std::make_unique<SigmoidLayer>());
    net.add_layer(std::make_unique<LinearLayer>(8, 1));
    net.add_layer(std::make_unique<SigmoidLayer>());
    return net;
}

// ── Thiết lập tham số thống kê từ phân phối dataset.csv ─────────────────────
static DataPipeline build_pipeline() {
    DataPipeline pipeline;
    // Các tham số Mean/StdDev được tính từ phân phối thực tế của dataset.csv
    pipeline.set_income_params(70000.0, 30000.0);
    pipeline.set_debt_params(15000.0, 10000.0);
    pipeline.set_delinquency_params(0.5, 0.5);
    pipeline.set_age_params(45.0, 15.0);
    return pipeline;
}

// ── Vòng lặp xử lý một lần đánh giá khách hàng ─────────────────────────────
static void run_single_assessment(Dashboard& db, DataPipeline& pipeline, NeuralNetwork& model) {
    std::cout << "\n";

    // Thu thập dữ liệu thô từ bàn phím qua bộ giáp nhập liệu an toàn
    double income      = db.getSafeDouble("  ▶ Nhập Thu nhập hàng năm (VND): ", 0.0, 1e9);
    double debt        = db.getSafeDouble("  ▶ Nhập Tổng dư nợ hiện tại (VND): ", 0.0, 1e9);
    double delinquency = db.getSafeDouble("  ▶ Nhập Số lần trễ hạn (0 - 100): ", 0.0, 100.0);
    int    age         = db.getSafeInt   ("  ▶ Nhập Tuổi của khách hàng: ", 18, 100);

    // Chuẩn hóa Z-score + kẹp biên [-3.0, 3.0] qua DataPipeline
    std::vector<double> features = pipeline.transform(income, debt,
                                                       delinquency,
                                                       static_cast<double>(age));

    // Dự đoán xác suất vỡ nợ (default) qua RiskEvaluator
    double risk_prob = RiskEvaluator::predict_approval_rate(features, model);

    // Hiển thị kết quả dưới dạng thẻ thẩm định ASCII Art
    db.displayAssessmentCard(risk_prob, "Du lieu nguoi dung nhap tu ban phim");
}

// ── Điểm khởi đầu toàn bộ hệ thống ─────────────────────────────────────────
int main() {
    try {
        // Khởi tạo đối tượng Dashboard (Tự kích hoạt ANSI & UTF-8 trên Windows)
        Dashboard db;
        NeuralNetwork model = build_network();
        DataPipeline  pipeline = build_pipeline();

        // Danh sách tùy chọn Menu động
        const std::vector<std::string> menu_options = {
            "Nhan ho so & Tham dinh rui ro tin dung",
            "Thoat he thong"
        };

        while (true) {
            db.showMenu(menu_options);

            int choice = db.getSafeInt("", 1, static_cast<int>(menu_options.size()));

            if (choice == 1) {
                run_single_assessment(db, pipeline, model);
            } else {
                std::cout << "\n\033[1;36m[SYSTEM] Hệ thống đã đóng an toàn. Tạm biệt!\033[0m\n\n";
                break;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "\033[1;31m[FATAL EXCEPTION] " << e.what() << "\033[0m\n";
        return 1;
    }

    return 0;
}
