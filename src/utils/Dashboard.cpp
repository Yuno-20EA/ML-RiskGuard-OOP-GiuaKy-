#include "riskguard/utils/Dashboard.hpp"
#include <iomanip>
#include <vector>

#define RESET       "\033[0m"
#define BOLD        "\033[1m"
#define MAGENTA     "\033[35m"
#define YELLOW      "\033[33m"
#define GREEN       "\033[32m"
#define RED         "\033[31m"

/**
 * Vẽ tiêu đề hệ thống bằng ASCII Art.
 * Dùng MAGENTA để tạo cảm giác công nghệ cao (Cyberpunk style).
 */
void Dashboard::drawHeader() {
    std::cout << MAGENTA << BOLD;
    std::cout << " _______________________________________________________________" << std::endl;
    std::cout << " |                                                               |" << std::endl;
    std::cout << " |  ____  ___  ____  _  __  ____ _   _  ____  ____  ____         |" << std::endl;
    std::cout << " |  | _ \\|_ _|/ ___|| |/ / / ___| | | |/ ___|| _ \\| _ \\        |" << std::endl;
    std::cout << " |  | |_) || | \\___ \\| ' < | |  _| | | |\\___ \\| |_) | | | |       |" << std::endl;
    std::cout << " |  |  _ < | |  ___) | . \\ | |_| | |_| | ___) |  _ <| |_| |       |" << std::endl;
    std::cout << " |  |_| \\_\\___||____/|_|\\_\\ \\____|\\___/|____/|_| \\_\\____/       |" << std::endl;
    std::cout << " |                                                               |" << std::endl;
    std::cout << " |              CORE ML RISK MANAGEMENT SYSTEM                   |" << std::endl;
    std::cout << " |_______________________________________________________________|" << RESET << std::endl;
}

/**
 * Hiển thị Menu điều hướng. Sử dụng icon để tăng tính trực quan.
 */
void Dashboard::showMenu() {
    drawHeader();
    std::cout << "\n" << BOLD << MAGENTA << " ✧ DANH MỤC ĐIỀU KHIỂN ✧" << RESET << std::endl;
    std::cout << "  " << YELLOW << "▶" << RESET << " Nạp & Chuẩn hóa dữ liệu (Use Case 1)" << std::endl;
    std::cout << "  " << YELLOW << "▶" << RESET << " Huấn luyện 'Bộ não' AI (Use Case 2)" << std::endl;
    std::cout << "  " << YELLOW << "▶" << RESET << " Thẩm định hồ sơ cấp tốc (Use Case 3)" << std::endl;
    std::cout << "  " << RED << "✖" << RESET << " Thoát hệ thống" << std::endl;
    std::cout << "\n" << BOLD << MAGENTA << "⚡ Lựa chọn của bạn: " << RESET;
}

/**
 * Hiển thị thanh tiến trình training AI.
 * @param epoch: vòng lặp hiện tại.
 * @param totalEpochs: tổng số vòng lặp.
 * @param loss: sai số huấn luyện.
 */
void Dashboard::showTrainingProgress(int epoch, int totalEpochs, double loss) {
    float progress = (float)epoch / totalEpochs;
    int barWidth = 40;

    std::cout << "\r" << YELLOW << "Training: " << RESET << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "■";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% "
              << "| Loss: " << std::fixed << std::setprecision(4) << loss << std::flush;

    if (epoch == totalEpochs) std::cout << GREEN << " [HOÀN TẤT]" << RESET << std::endl;
}

/**
 * Hiển thị bảng kết quả thẩm định rủi ro.
 * Sử dụng logic so sánh để trả về màu sắc trạng thái (Xanh: Duyệt, Đỏ: Từ chối).
 */
void Dashboard::displayAssessmentCard(double riskProb, const std::string& mainReason) {
    std::string status = (riskProb < 0.5) ? "DUYỆT VAY" : "TỪ CHỐI";
    std::string color = (riskProb < 0.5) ? GREEN : RED;

    std::cout << "\n" << color << BOLD;
    std::cout << "┌──────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│                 KẾT QUẢ THẨM ĐỊNH TÍN DỤNG               │" << std::endl;
    std::cout << "├──────────────────────────────────────────────────────────┤" << std::endl;
    std::cout << "│  TRẠNG THÁI: " << std::left << std::setw(44) << status << "│" << std::endl;
    std::cout << "│  XÁC SUẤT RỦI RO: " << std::fixed << std::setprecision(2)
              << (riskProb * 100) << "%" << std::setw(34) << " " << "│" << std::endl;
    std::cout << "├──────────────────────────────────────────────────────────┤" << std::endl;
    std::cout << "│  NGUYÊN NHÂN CHÍNH: " << std::left << std::setw(37) << mainReason << "│" << std::endl;
    std::cout << "└──────────────────────────────────────────────────────────┘" << RESET << std::endl;
}