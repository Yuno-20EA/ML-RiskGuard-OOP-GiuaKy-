#include "riskguard/utils/Dashboard.hpp"
#include <iomanip>
#include <vector>

#define RESET       "\033[0m"
#define BOLD        "\033[1m"
#define RED         "\033[31m"  
#define GREEN       "\033[32m"  
#define YELLOW      "\033[33m"  
#define CYAN        "\033[36m"  

void Dashboard::drawHeader() {
    std::cout << CYAN << BOLD;
    std::cout << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║    ____  _     _      ____                       _       ║" << std::endl;
    std::cout << "║   |  _ \\(_)___| | __ / ___|_   _  __ _ _ __   __| |      ║" << std::endl;
    std::cout << "║   | |_) | / __| |/ /| |  _| | | |/ _` | '__| / _` |      ║" << std::endl;
    std::cout << "║   |  _ <| \\__ \\   < | |_| | |_| | (_| | |   | (_| |      ║" << std::endl;
    std::cout << "║   |_| \\_\\_|___/_|\\_\\ \\____|\\__,_|\\__,_|_|    \\__,_|      ║" << std::endl;
    std::cout << "║                CORE ML RISK MANAGEMENT SYSTEM                ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << RESET << std::endl;
}

void Dashboard::showMenu() {
    drawHeader();
    std::cout << "\n" << BOLD << " DANH MỤC ĐIỀU KHIỂN:" << RESET << std::endl;
    std::cout << "  " << CYAN << "1." << RESET << " Nạp & Chuẩn hóa dữ liệu (Use Case 1)" << std::endl;
    std::cout << "  " << CYAN << "2." << RESET << " Huấn luyện 'Bộ não' AI (Use Case 2)" << std::endl;
    std::cout << "  " << CYAN << "3." << RESET << " Thẩm định hồ sơ cấp tốc (Use Case 3)" << std::endl;
    std::cout << "  " << RED << "4." << RESET << " Thoát hệ thống" << std::endl;
    std::cout << "\n" << BOLD << "⚡ Lựa chọn của bạn: " << RESET;
}

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