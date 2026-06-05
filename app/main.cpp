// ============================================================
//  RiskGuard ML Framework — app/main.cpp
//  Giao diện Cyberpunk CLI tương tác — Staged Animated Inference
// ============================================================
#include "riskguard/utils/Dashboard.hpp"
#include "riskguard/utils/DataPipeline.hpp"
#include "riskguard/utils/DataLoader.hpp"
#include "riskguard/utils/ModelManager.hpp"
#include "riskguard/network/NeuralNetwork.hpp"
#include "riskguard/network/RiskEvaluator.hpp"
#include "riskguard/layers/LinearLayer.hpp"
#include "riskguard/layers/SigmoidLayer.hpp"
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <memory>
#include <vector>
#include <string>

using namespace riskguard;

// ── Mã màu nội bộ (ANSI) ────────────────────────────────────────────────────
namespace ansi {
    constexpr auto RESET    = "\033[0m";
    constexpr auto BOLD     = "\033[1m";
    constexpr auto YELLOW   = "\033[33m";
    constexpr auto CYAN     = "\033[1;36m";
    constexpr auto MAGENTA  = "\033[1;35m";
    constexpr auto GREEN    = "\033[1;32m";
    constexpr auto GRAY     = "\033[90m";
}

// ── Vẽ thanh tiến trình trên một dòng (dùng \r để ghi đè) ──────────────────
static void render_progress_bar(std::string_view label,
                                 std::string_view color,
                                 int total_steps,
                                 int step_delay_ms) {
    constexpr int BAR_WIDTH = 36;
    for (int step = 0; step <= total_steps; ++step) {
        int filled = (step * BAR_WIDTH) / total_steps;
        int pct    = (step * 100) / total_steps;

        std::cout << "\r  " << color << label << ansi::RESET
                  << " [";
        for (int k = 0; k < BAR_WIDTH; ++k)
            std::cout << (k < filled ? "█" : ansi::GRAY + std::string("░") + ansi::RESET);
        std::cout << "] "
                  << color << std::setw(3) << pct << "%" << ansi::RESET
                  << std::flush;

        std::this_thread::sleep_for(std::chrono::milliseconds(step_delay_ms));
    }
    std::cout << "\n";
}

// ── Hiệu ứng phân tích 3 giai đoạn [AI COGNITIVE ASSESSMENT] ───────────────
static void run_cognitive_assessment(const std::vector<double>& features) {
    std::cout << "\n"
              << ansi::CYAN << ansi::BOLD
              << " ╔══════════════════════════════════════════════════╗\n"
              << " ║         HỆ THỐNG PHÂN TÍCH TÍN DỤNG AI           ║\n"
              << " ╚══════════════════════════════════════════════════╝"
              << ansi::RESET << "\n\n";

    // ── Giai đoạn 1: Làm sạch & Kẹp biên dữ liệu (Sanitization) ─────────────
    std::cout << ansi::YELLOW << ansi::BOLD
              << " [XỬ LÝ]   Làm sạch & Kẹp biên dữ liệu đầu vào\n"
              << ansi::RESET;
    render_progress_bar("[XU LY] ", ansi::YELLOW, 20, 40);

    // In kết quả Z-score đã làm sạch
    const std::string labels[] = {"Thu nhap", "Du no  ", "Tre han", "Tuoi   "};
    std::cout << ansi::GRAY << "  ├─ Vector đặc trưng (Z-score kẹp biên [-3.0, 3.0]):\n";
    for (std::size_t i = 0; i < features.size(); ++i) {
        std::cout << "  │   " << labels[i] << " = "
                  << ansi::YELLOW << std::fixed << std::setprecision(6)
                  << features[i] << ansi::RESET << "\n";
    }
    std::cout << ansi::GRAY << "  └─ Làm sạch dữ liệu hoàn tất.\n" << ansi::RESET << "\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    // ── Giai đoạn 2: Lan truyền ma trận trọng số (Propagation) ───────────────
    std::cout << ansi::CYAN << ansi::BOLD
              << " [TÍNH TOÁN] Lan truyền ma trận đặc trưng (1×4 → 1×8)\n"
              << ansi::RESET;
    render_progress_bar("[TINH TOAN]", ansi::CYAN, 20, 35);
    std::cout << ansi::GRAY
              << "  └─ Ma trận trọng số lớp ẩn đã được nhân xong.\n"
              << ansi::RESET << "\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(250));

    // ── Giai đoạn 3: Kích hoạt đầu ra Sigmoid (Activation) ───────────────────
    std::cout << ansi::MAGENTA << ansi::BOLD
              << " [KẾT LUẬN]  Kích hoạt hàm Sigmoid lớp đầu ra\n"
              << ansi::RESET;
    render_progress_bar("[KET LUAN]", ansi::MAGENTA, 20, 30);
    std::cout << ansi::GRAY
              << "  └─ Xác suất đầu ra đã được ép về khoảng [0.0, 1.0].\n"
              << ansi::RESET << "\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

// ── Khởi tạo kiến trúc mạng nơ-ron 4→8→1 ───────────────────────────────────
static NeuralNetwork build_network() {
    NeuralNetwork net;
    net.add_layer(std::make_unique<LinearLayer>(4, 8));
    net.add_layer(std::make_unique<SigmoidLayer>());
    net.add_layer(std::make_unique<LinearLayer>(8, 1));
    net.add_layer(std::make_unique<SigmoidLayer>());
    return net;
}

// ── Xử lý một phiên thẩm định đầy đủ ───────────────────────────────────────
static void run_single_assessment(Dashboard& db, const DataPipeline& pipeline, NeuralNetwork& model) {
    std::cout << "\n"
              << ansi::CYAN << " ── Nhập thông tin hồ sơ khách hàng ──────────────────\n"
              << ansi::RESET;

    double income      = db.getSafeDouble(
        "  ▶ Thu nhập hàng năm  (VND, 0 - 100.000.000.000): ",
        0.0, 100'000'000'000.0);
    double debt        = db.getSafeDouble(
        "  ▶ Tổng dư nợ hiện tại (VND, 0 - 20.000.000.000): ",
        0.0, 20'000'000'000.0);
    double delinquency = db.getSafeDouble(
        "  ▶ Số lần trễ hạn thanh toán (0 - 100): ",
        0.0, 100.0);
    int    age         = db.getSafeInt(
        "  ▶ Tuổi khách hàng (1 - 200): ", 1, 200);

    std::vector<double> features = pipeline.transform(income, debt,
                                                       delinquency,
                                                       static_cast<double>(age));

    run_cognitive_assessment(features);

    double approval_rate = RiskEvaluator::predict_approval_rate(features, model);
    double risk_prob = 1.0 - approval_rate;
    
    std::string reason = RiskEvaluator::evaluate_risk_factors(income, debt, delinquency, approval_rate);

    // Hiển thị trực tiếp risk_prob cho người dùng dễ hiểu
    db.displayAssessmentCard(risk_prob, reason);
}

// ── Khởi tạo & Huấn luyện mạng tự động ──────────────────────────────────────
static void startup_training(Dashboard& db, NeuralNetwork& model, DataPipeline& pipeline) {
#ifdef RISKGUARD_PROJECT_ROOT
    std::string csv_path = std::string(RISKGUARD_PROJECT_ROOT) + "/data/dataset.csv";
    std::string model_path = std::string(RISKGUARD_PROJECT_ROOT) + "/model.json";
#else
    std::string csv_path = "data/dataset.csv"; // Fallback cho trường hợp biên dịch thủ công
    std::string model_path = "model.json";
#endif

    std::cout << ansi::YELLOW << "[HỆ THỐNG] Kiểm tra mô hình lưu trữ tại " << model_path << "..." << ansi::RESET << std::endl;
    if (ModelManager::loadModel(model_path, model, pipeline)) {
        std::cout << ansi::GREEN << "[HỆ THỐNG] Đã nạp thành công bộ trọng số và phân phối dữ liệu từ model.json. Bỏ qua huấn luyện!" << ansi::RESET << std::endl;
        return;
    }

    DataLoader loader;
    std::cout << ansi::YELLOW << "[HỆ THỐNG] Đang tải dữ liệu từ " << csv_path << "..." << ansi::RESET << std::endl;
    Matrix raw_data = loader.loadRawCSV(csv_path);

    if (raw_data.get_rows() == 0) {
        std::cerr << ansi::BOLD << "\033[1;31m[LỖI] Không thể nạp dataset. CSV trống hoặc sai đường dẫn.\033[0m\n";
        return;
    }

    // 1. Phân tích tham số tự động cho DataPipeline (Auto-Fit)
    std::cout << ansi::YELLOW << "[HỆ THỐNG] Phân tích phân phối thống kê từ dữ liệu..." << ansi::RESET << std::endl;
    pipeline.fit(raw_data);

    // 2. Chuẩn hóa tập dữ liệu bằng Z-score
    Matrix normalized_data = pipeline.transform(raw_data);

    // 3. Tách Features (X) và Targets (Y)
    int rows = normalized_data.get_rows();
    int cols = normalized_data.get_cols();
    Matrix X(rows, 4);
    Matrix Y(rows, 1);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < 4; ++j) {
            X(i, j) = normalized_data(i, j);
        }
        Y(i, 0) = normalized_data(i, 4); // Cột thứ 5 là nhãn
    }

    // 4. Huấn luyện mạng
    int epochs = 50;
    double learning_rate = 0.05;
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        Matrix predictions = model.forward(X);
        double loss = model.calculateBCELoss(predictions, Y);
        Matrix gradient = model.calculateBCEGradient(predictions, Y);
        
        model.backward(gradient);
        model.update_parameters(learning_rate);
        
        // Tính Accuracy (Ngưỡng 0.5)
        int correct = 0;
        for (int i = 0; i < rows; ++i) {
            int pred_class = (predictions(i, 0) >= 0.5) ? 1 : 0;
            int target_class = (Y(i, 0) >= 0.5) ? 1 : 0;
            if (pred_class == target_class) correct++;
        }
        double accuracy = static_cast<double>(correct) / rows;
        
        db.showTrainingProgress(epoch, epochs, loss, accuracy);
        std::this_thread::sleep_for(std::chrono::milliseconds(20)); // Delay nhỏ để thấy hiệu ứng
    }

    std::cout << ansi::YELLOW << "\n[HỆ THỐNG] Đang lưu mô hình..." << ansi::RESET << std::endl;
    if (ModelManager::saveModel(model_path, model, pipeline)) {
        std::cout << ansi::GREEN << "[HỆ THỐNG] Đã lưu mô hình thành công vào model.json." << ansi::RESET << std::endl;
    }
}

// ── Entry Point ──────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    // Xử lý tham số dòng lệnh --help
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "--help" || arg == "-h") {
            std::cout << "RiskGuard ML Framework v2.0\n\n"
                      << "Cach su dung:\n"
                      << "  riskguard [OPTIONS]\n\n"
                      << "Options:\n"
                      << "  -h, --help    Hien thi thong tin huong dan nay va thoat.\n\n"
                      << "Chương trình sẽ tự động đọc data/dataset.csv để huấn luyện.\n";
            return 0;
        }
    }

    try {
        Dashboard     db;
        NeuralNetwork model = build_network();
        DataPipeline  pipeline;

        // Khởi động huấn luyện (Runtime Training)
        startup_training(db, model, pipeline);

        const std::vector<std::string> menu_options = {
            "Nhận hồ sơ & Thẩm định rủi ro tín dụng",
            "Thoát hệ thống"
        };

        while (true) {
            db.showMenu(menu_options);
            int choice = db.getSafeInt("", 1, static_cast<int>(menu_options.size()));

            if (choice == 1) {
                run_single_assessment(db, pipeline, model);
            } else {
                std::cout << "\n" << ansi::CYAN
                          << "[HỆ THỐNG] Hệ thống đã đóng an toàn. Hẹn gặp lại!\n"
                          << ansi::RESET << "\n";
                break;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << ansi::BOLD << "\033[1;31m[FATAL] " << e.what() << ansi::RESET << "\n";
        return 1;
    }
    return 0;
}
