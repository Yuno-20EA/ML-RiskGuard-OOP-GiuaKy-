// ============================================================
//  RiskGuard ML Framework — app/main.cpp  v2.1
//  Giao diện Cyberpunk CLI tương tác — Staged Animated Inference
//  Nâng cấp: Early Stopping · LR Decay · --verbose · Lịch sử phiên
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
#include <algorithm>
#include <limits>
#include <cmath>

using namespace riskguard;

// ── Mã màu nội bộ (ANSI) ────────────────────────────────────────────────────
namespace ansi {
    constexpr auto RESET    = "\033[0m";
    constexpr auto BOLD     = "\033[1m";
    constexpr auto YELLOW   = "\033[33m";
    constexpr auto CYAN     = "\033[1;36m";
    constexpr auto MAGENTA  = "\033[1;35m";
    constexpr auto GREEN    = "\033[1;32m";
    constexpr auto RED      = "\033[1;31m";
    constexpr auto GRAY     = "\033[90m";
    constexpr auto WHITE    = "\033[1;37m";
}

// ── Bản ghi lịch sử một phiên thẩm định ─────────────────────────────────────
struct AssessmentRecord {
    int    session_id;
    double income;
    double debt;
    double delinquency;
    int    age;
    double risk_prob;       // [0.0, 1.0]
    std::string decision;   // "DUYỆT" | "TỪ CHỐI"
    std::string reason;
};

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

// ── Format số VND với dấu phân cách hàng nghìn ──────────────────────────────
static std::string fmt_vnd(double val) {
    long long v = static_cast<long long>(val);
    std::string s = std::to_string(v);
    int n = static_cast<int>(s.size());
    std::string result;
    for (int i = 0; i < n; ++i) {
        if (i > 0 && (n - i) % 3 == 0) result += '.';
        result += s[i];
    }
    return result + " VND";
}

// ── Hiệu ứng phân tích 3 giai đoạn [AI COGNITIVE ASSESSMENT] ───────────────
static void run_cognitive_assessment(const std::vector<double>& features, bool verbose) {
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

    if (verbose) {
        const std::string labels[] = {"Thu nhap", "Du no  ", "Tre han", "Tuoi   "};
        std::cout << ansi::GRAY << "  ├─ Vector đặc trưng (Z-score kẹp biên [-3.0, 3.0]):\n";
        for (std::size_t i = 0; i < features.size(); ++i) {
            std::cout << "  │   " << labels[i] << " = "
                      << ansi::YELLOW << std::fixed << std::setprecision(6)
                      << features[i] << ansi::RESET << "\n";
        }
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

// ── Nhập số an toàn với gợi ý định dạng và khả năng nhập lại ────────────────
static double prompt_vnd(const std::string& label, double lo, double hi,
                          const std::string& example) {
    double value;
    while (true) {
        std::cout << ansi::CYAN << "  ▶ " << label << "\n"
                  << ansi::GRAY << "    (ví dụ: " << example
                  << " | giới hạn: 0 – " << fmt_vnd(hi) << ")\n"
                  << ansi::CYAN << "  → " << ansi::RESET;
        if (std::cin >> value && value >= lo && value <= hi) return value;
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << ansi::RED << ansi::BOLD
                  << "  [LỖI] Giá trị phải nằm trong khoảng "
                  << lo << " – " << fmt_vnd(hi) << ".\n"
                  << ansi::RESET;
    }
}

// ── Hiển thị bảng lịch sử tất cả hồ sơ trong phiên ─────────────────────────
static void show_session_history(const std::vector<AssessmentRecord>& history) {
    if (history.empty()) {
        std::cout << "\n" << ansi::YELLOW
                  << "  [THÔNG BÁO] Chưa có hồ sơ nào được thẩm định trong phiên này.\n"
                  << ansi::RESET;
        return;
    }

    std::cout << "\n" << ansi::CYAN << ansi::BOLD
              << " ╔══════════════════════════════════════════════════════════════════╗\n"
              << " ║               LỊCH SỬ THẨM ĐỊNH — PHIÊN HIỆN TẠI               ║\n"
              << " ╚══════════════════════════════════════════════════════════════════╝\n"
              << ansi::RESET;

    std::cout << ansi::GRAY
              << " ┌────┬──────────────┬──────────────┬──────┬─────┬────────┬──────────────┐\n"
              << " │ #  │ Thu Nhập     │ Dư Nợ        │ TrHạ │ Tuổi│ RủiRo  │ Quyết Định   │\n"
              << " ├────┼──────────────┼──────────────┼──────┼─────┼────────┼──────────────┤\n"
              << ansi::RESET;

    for (const auto& r : history) {
        std::string dec_color = (r.decision == "DUYỆT") ? ansi::GREEN : ansi::RED;
        std::cout << ansi::GRAY << " │ " << ansi::RESET
                  << ansi::WHITE << std::setw(2) << r.session_id << ansi::RESET
                  << ansi::GRAY << " │ " << ansi::RESET
                  << std::setw(12) << std::fixed << std::setprecision(0) << r.income
                  << ansi::GRAY << " │ " << ansi::RESET
                  << std::setw(12) << r.debt
                  << ansi::GRAY << " │ " << ansi::RESET
                  << std::setw(4) << static_cast<int>(r.delinquency)
                  << ansi::GRAY << " │ " << ansi::RESET
                  << std::setw(3) << r.age
                  << ansi::GRAY << " │ " << ansi::RESET
                  << ansi::YELLOW << std::setw(6) << std::setprecision(1)
                  << (r.risk_prob * 100.0) << "%" << ansi::RESET
                  << ansi::GRAY << " │ " << ansi::RESET
                  << dec_color << ansi::BOLD << std::setw(12) << r.decision
                  << ansi::RESET
                  << ansi::GRAY << " │\n" << ansi::RESET;
    }

    std::cout << ansi::GRAY
              << " └────┴──────────────┴──────────────┴──────┴─────┴────────┴──────────────┘\n"
              << ansi::RESET;
    std::cout << ansi::GRAY << "  Tổng hồ sơ: " << ansi::WHITE << history.size()
              << ansi::GRAY << "  |  Duyệt: " << ansi::GREEN
              << std::count_if(history.begin(), history.end(),
                               [](const auto& r){ return r.decision == "DUYỆT"; })
              << ansi::GRAY << "  |  Từ chối: " << ansi::RED
              << std::count_if(history.begin(), history.end(),
                               [](const auto& r){ return r.decision == "TỪ CHỐI"; })
              << ansi::RESET << "\n";
}

// ── Xử lý một phiên thẩm định đầy đủ ───────────────────────────────────────
static void run_single_assessment(Dashboard& db,
                                   const DataPipeline& pipeline,
                                   NeuralNetwork& model,
                                   std::vector<AssessmentRecord>& history,
                                   bool verbose) {
    std::cout << "\n"
              << ansi::CYAN << " ── Nhập thông tin hồ sơ khách hàng ──────────────────\n"
              << ansi::RESET;

    double income, debt, delinquency;
    int    age;

    // ── Vòng lặp xác nhận: cho phép nhập lại toàn bộ nếu sai ────────────────
    while (true) {
        income = prompt_vnd("Thu nhập hàng năm (VND)",
                            0.0, 100'000'000'000.0, "50.000.000");
        debt   = prompt_vnd("Tổng dư nợ hiện tại (VND)",
                            0.0, 20'000'000'000.0,  "15.000.000");

        std::cout << ansi::CYAN << "  ▶ Số lần trễ hạn thanh toán (0 – 100)\n"
                  << ansi::CYAN << "  → " << ansi::RESET;
        while (!(std::cin >> delinquency) || delinquency < 0.0 || delinquency > 100.0) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << ansi::RED << "  [LỖI] Nhập số từ 0 đến 100.\n" << ansi::CYAN << "  → " << ansi::RESET;
        }

        std::cout << ansi::CYAN << "  ▶ Tuổi khách hàng (1 – 200)\n"
                  << ansi::CYAN << "  → " << ansi::RESET;
        while (!(std::cin >> age) || age < 1 || age > 200) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << ansi::RED << "  [LỖI] Nhập số nguyên từ 1 đến 200.\n" << ansi::CYAN << "  → " << ansi::RESET;
        }

        // ── Bước xác nhận tóm tắt ────────────────────────────────────────────
        std::cout << "\n" << ansi::YELLOW << ansi::BOLD
                  << " ── Xác nhận thông tin nhập ────────────────────────────\n"
                  << ansi::RESET
                  << ansi::GRAY  << "  Thu nhập   : " << ansi::WHITE << fmt_vnd(income)   << "\n"
                  << ansi::GRAY  << "  Dư nợ      : " << ansi::WHITE << fmt_vnd(debt)     << "\n"
                  << ansi::GRAY  << "  Trễ hạn    : " << ansi::WHITE << static_cast<int>(delinquency) << " lần\n"
                  << ansi::GRAY  << "  Tuổi       : " << ansi::WHITE << age << " tuổi\n"
                  << ansi::RESET;

        double dti = (income > 1e-7) ? (debt / income * 100.0) : 999.0;
        std::cout << ansi::GRAY << "  DTI         : "
                  << (dti > 40.0 ? ansi::RED : ansi::GREEN)
                  << std::fixed << std::setprecision(1) << dti << "%\n" << ansi::RESET;

        std::cout << "\n" << ansi::CYAN << "  [Y/n] Xác nhận và tiến hành thẩm định? " << ansi::RESET;
        char confirm;
        std::cin >> confirm;
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        if (confirm == 'n' || confirm == 'N') {
            std::cout << ansi::YELLOW << "  [HỆ THỐNG] Nhập lại toàn bộ thông tin...\n\n" << ansi::RESET;
            continue;
        }
        break;
    }

    std::vector<double> features = pipeline.transform(income, debt,
                                                       delinquency,
                                                       static_cast<double>(age));

    run_cognitive_assessment(features, verbose);

    double approval_rate = RiskEvaluator::predict_approval_rate(features, model);
    double risk_prob     = 1.0 - approval_rate;

    std::string reason   = RiskEvaluator::evaluate_risk_factors(income, debt, delinquency, approval_rate);
    std::string decision = (approval_rate >= 0.5) ? "DUYỆT" : "TỪ CHỐI";

    db.displayAssessmentCard(risk_prob, reason);

    // Ghi vào lịch sử phiên
    history.push_back({
        static_cast<int>(history.size()) + 1,
        income, debt, delinquency, age,
        risk_prob, decision, reason
    });
}

// ── Khởi tạo & Huấn luyện mạng tự động ─────────────────────────────────────
static void startup_training(Dashboard& db, NeuralNetwork& model,
                              DataPipeline& pipeline, bool verbose) {
#ifdef RISKGUARD_PROJECT_ROOT
    std::string csv_path   = std::string(RISKGUARD_PROJECT_ROOT) + "/data/dataset.csv";
    std::string model_path = std::string(RISKGUARD_PROJECT_ROOT) + "/model.json";
#else
    std::string csv_path   = "data/dataset.csv";
    std::string model_path = "model.json";
#endif

    std::cout << ansi::YELLOW << "[HỆ THỐNG] Kiểm tra mô hình tại " << model_path << "..." << ansi::RESET << "\n";
    if (ModelManager::loadModel(model_path, model, pipeline)) {
        std::cout << ansi::GREEN << "[HỆ THỐNG] Đã nạp mô hình v2.1.0 (arch=4-8-1). Bỏ qua huấn luyện!\n" << ansi::RESET;
        return;
    }

    DataLoader loader;
    std::cout << ansi::YELLOW << "[HỆ THỐNG] Đang tải dữ liệu từ " << csv_path << "...\n" << ansi::RESET;
    Matrix raw_data = loader.loadRawCSV(csv_path);

    if (raw_data.get_rows() == 0) {
        std::cerr << ansi::BOLD << "\033[1;31m[LỖI] Không thể nạp dataset. CSV trống hoặc sai đường dẫn.\033[0m\n";
        return;
    }

    pipeline.fit(raw_data);
    Matrix normalized_data = pipeline.transform(raw_data);

    int rows = normalized_data.get_rows();
    Matrix X(rows, 4);
    Matrix Y(rows, 1);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < 4; ++j) X(i, j) = normalized_data(i, j);
        Y(i, 0) = normalized_data(i, 4);
    }

    // ── Huấn luyện với Early Stopping + Learning Rate Step Decay ─────────────
    constexpr int    MAX_EPOCHS    = 200;
    constexpr int    PATIENCE      = 15;   // Dừng nếu loss không cải thiện sau N epoch
    constexpr double INITIAL_LR    = 0.05;
    constexpr int    LR_DECAY_STEP = 30;   // Giảm LR sau mỗi N epoch
    constexpr double LR_DECAY_RATE = 0.7;

    double best_loss     = std::numeric_limits<double>::max();
    int    stagnant_ctr  = 0;
    double learning_rate = INITIAL_LR;

    std::cout << ansi::YELLOW << "[HỆ THỐNG] Bắt đầu huấn luyện (max=" << MAX_EPOCHS
              << " epochs, patience=" << PATIENCE << ")...\n" << ansi::RESET;

    for (int epoch = 1; epoch <= MAX_EPOCHS; ++epoch) {
        // LR step decay mỗi LR_DECAY_STEP epoch
        if (epoch > 1 && (epoch - 1) % LR_DECAY_STEP == 0) {
            learning_rate *= LR_DECAY_RATE;
            if (verbose)
                std::cout << "\n" << ansi::GRAY << "  [LR Decay] lr → "
                          << std::fixed << std::setprecision(6) << learning_rate
                          << " (epoch " << epoch << ")\n" << ansi::RESET;
        }

        Matrix predictions = model.forward(X);
        double loss        = model.calculateBCELoss(predictions, Y);
        Matrix gradient    = model.calculateBCEGradient(predictions, Y);

        model.backward(gradient);
        model.update_parameters(learning_rate);

        // Accuracy
        int correct = 0;
        for (int i = 0; i < rows; ++i) {
            int pred_class   = (predictions(i, 0) >= 0.5) ? 1 : 0;
            int target_class = (Y(i, 0) >= 0.5)           ? 1 : 0;
            if (pred_class == target_class) ++correct;
        }
        double accuracy = static_cast<double>(correct) / rows;

        db.showTrainingProgress(epoch, MAX_EPOCHS, loss, accuracy);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // Early Stopping
        if (loss < best_loss - 1e-5) {
            best_loss    = loss;
            stagnant_ctr = 0;
        } else {
            ++stagnant_ctr;
            if (stagnant_ctr >= PATIENCE) {
                std::cout << "\n" << ansi::GREEN
                          << "[HỆ THỐNG] Early Stopping kích hoạt tại epoch " << epoch
                          << " (loss=" << std::fixed << std::setprecision(4) << best_loss << ").\n"
                          << ansi::RESET;
                break;
            }
        }
    }

    std::cout << ansi::YELLOW << "\n[HỆ THỐNG] Đang lưu mô hình...\n" << ansi::RESET;
    if (ModelManager::saveModel(model_path, model, pipeline))
        std::cout << ansi::GREEN << "[HỆ THỐNG] Đã lưu mô hình thành công vào " << model_path << ".\n" << ansi::RESET;
}

// ── Entry Point ──────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    bool verbose = false;

    // Parse CLI flags
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "RiskGuard ML Framework v2.1\n\n"
                      << "Cach su dung:\n"
                      << "  riskguard [OPTIONS]\n\n"
                      << "Options:\n"
                      << "  -h, --help      In huong dan va thoat.\n"
                      << "  --verbose       In chi tiet Z-score, LR decay va epoch log.\n\n"
                      << "Chuong trinh tu dong doc data/dataset.csv de huan luyen.\n";
            return 0;
        }
        if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        }
    }

    try {
        Dashboard     db;
        NeuralNetwork model    = build_network();
        DataPipeline  pipeline;

        startup_training(db, model, pipeline, verbose);

        std::vector<AssessmentRecord> history;

        const std::vector<std::string> menu_options = {
            "Nhan ho so & Tham dinh rui ro tin dung",
            "Xem lich su tham dinh phien nay",
            "Thoat he thong"
        };

        while (true) {
            db.showMenu(menu_options);
            int choice = db.getSafeInt("", 1, static_cast<int>(menu_options.size()));

            if (choice == 1) {
                run_single_assessment(db, pipeline, model, history, verbose);
            } else if (choice == 2) {
                show_session_history(history);
            } else {
                std::cout << "\n" << ansi::CYAN
                          << "[HE THONG] He thong da dong an toan. Hen gap lai!\n"
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
