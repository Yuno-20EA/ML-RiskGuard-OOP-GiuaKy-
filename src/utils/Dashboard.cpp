#include "riskguard/utils/Dashboard.hpp"
#include <iomanip>
#include <limits>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#endif

// Bộ màu Cyberpunk Terminal (ANSI Escape Codes)
#define RESET       "\033[0m"
#define BOLD        "\033[1m"
#define MAGENTA     "\033[1;35m" // Magenta sáng
#define CYAN        "\033[1;36m" // Cyan sáng
#define GREEN       "\033[1;32m" // Xanh lá cây sáng (Success)
#define RED         "\033[1;31m" // Đỏ sáng (Error/Rejected)
#define YELLOW      "\033[33m"   // Vàng (Info/Progress)
#define GRAY        "\033[90m"   // Xám (Viền/Decor)

namespace riskguard {

Dashboard::Dashboard() {
#ifdef _WIN32
    // Bật xử lý ANSI Escape Codes cho Console Windows cũ
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut != INVALID_HANDLE_VALUE) {
        DWORD dwMode = 0;
        if (GetConsoleMode(hOut, &dwMode)) {
            dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            SetConsoleMode(hOut, dwMode);
        }
    }
    // Bật Output bảng mã UTF-8 để vẽ box Unicode không bị lỗi
    SetConsoleOutputCP(CP_UTF8);
#endif
}

void Dashboard::drawHeader() const {
    std::cout << MAGENTA << BOLD;
    std::cout << " _______________________________________________________________" << std::endl;
    std::cout << " |                                                               |" << std::endl;
    std::cout << " |  ____  ___  ____  _  __  ____ _   _  ____  ____  ____         |" << std::endl;
    std::cout << " |  | _ \\|_ _|/ ___|| |/ / / ___| | | |/ ___|| _ \\| _ \\        |" << std::endl;
    std::cout << " |  | |_) || | \\___ \\| ' < | |  _| | | |\\___ \\| |_) | | | |       |" << std::endl;
    std::cout << " |  |  _ < | |  ___) | . \\ | |_| | |_| | ___) |  _ <| |_| |       |" << std::endl;
    std::cout << " |  |_| \\_\\___||____/|_|\\_\\ \\____|\\___/|____/|_| \\_\\____/       |" << std::endl;
    std::cout << " |                                                               |" << std::endl;
    std::cout << " |" << CYAN << "              HỆ THỐNG QUẢN TRỊ RỦI RO LÕI ML                  " << MAGENTA << "|" << std::endl;
    std::cout << " |_______________________________________________________________|" << RESET << std::endl;
}

void Dashboard::showMenu(const std::vector<std::string>& options) const {
    drawHeader();
    std::cout << "\n" << BOLD << CYAN << " ✧ MENU ĐIỀU KHIỂN HỆ THỐNG ✧" << RESET << std::endl;
    
    if (options.empty()) return;

    // Tìm độ dài chuỗi lớn nhất
    size_t max_len = 0;
    for (const auto& opt : options) {
        if (opt.length() > max_len) {
            max_len = opt.length();
        }
    }
    
    // Khoảng đệm cho UI
    size_t inner_width = max_len + 6;

    // Vẽ khung menu linh hoạt bằng Unicode
    std::cout << GRAY << " ┌";
    for(size_t k = 0; k < inner_width + 2; ++k) std::cout << "─";
    std::cout << "┐" << RESET << std::endl;
    
    for (size_t i = 0; i < options.size(); ++i) {
        std::cout << GRAY << " │ " << RESET 
                  << YELLOW << "[" << i + 1 << "] " << RESET 
                  << std::left << std::setw(static_cast<int>(max_len)) << options[i]
                  << GRAY << "   │" << RESET << std::endl;
    }
    
    std::cout << GRAY << " └";
    for(size_t k = 0; k < inner_width + 2; ++k) std::cout << "─";
    std::cout << "┘" << RESET << std::endl;
    
    std::cout << "\n" << BOLD << CYAN << "⚡ NHẬP LỆNH > " << RESET;
}

void Dashboard::showTrainingProgress(int epoch, int totalEpochs, double loss) const {
    float progress = static_cast<float>(epoch) / totalEpochs;
    int barWidth = 40;

    std::cout << "\r" << YELLOW << "HUẤN LUYỆN MẠNG NƠ-RON: " << RESET << "[";
    int pos = static_cast<int>(barWidth * progress);
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << CYAN << "■" << RESET;
        else std::cout << GRAY << "■" << RESET;
    }
    std::cout << "] " << CYAN << int(progress * 100.0) << "% " << RESET
              << "| Tổn thất: " << std::fixed << std::setprecision(4) << MAGENTA << loss << RESET << std::flush;

    if (epoch == totalEpochs) std::cout << GREEN << " [MẠNG ĐÃ ĐỒNG BỘ]" << RESET << std::endl;
}

void Dashboard::displayAssessmentCard(double riskProb, std::string_view mainReason) const {
    std::string status = (riskProb < 0.5) ? "PHÊ DUYỆT (RỦI RO THẤP)" : "TỪ CHỐI (RỦI RO CAO)";
    std::string color = (riskProb < 0.5) ? GREEN : RED;

    std::cout << "\n" << color << BOLD;
    std::cout << "┌──────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│                 ĐÁNH GIÁ RỦI RO TÍN DỤNG                 │" << std::endl;
    std::cout << "├──────────────────────────────────────────────────────────┤" << std::endl;
    std::cout << "│  TRẠNG THÁI: " << std::left << std::setw(46) << status << "│" << std::endl;
    std::cout << "│  XÁC SUẤT RỦI RO: " << std::fixed << std::setprecision(2)
              << (riskProb * 100) << "%" << std::setw(37) << " " << "│" << std::endl;
    std::cout << "├──────────────────────────────────────────────────────────┤" << std::endl;
    std::cout << "│  LÝ DO: " << std::left << std::setw(49) << std::string(mainReason) << "│" << std::endl;
    std::cout << "└──────────────────────────────────────────────────────────┘" << RESET << std::endl;
}

int Dashboard::getSafeInt(std::string_view prompt, int min_val, int max_val) const {
    int value;
    while (true) {
        std::cout << CYAN << prompt << RESET;
        if (std::cin >> value && value >= min_val && value <= max_val) {
            return value;
        }
        std::cin.clear(); // Xóa cờ lỗi
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Loại bỏ các ký tự rác trong buffer
        std::cout << RED << BOLD << "[LỖI] Dữ liệu không hợp lệ. Vui lòng nhập số nguyên từ " 
                  << min_val << " đến " << max_val << "." << RESET << std::endl;
    }
}

double Dashboard::getSafeDouble(std::string_view prompt, double min_val, double max_val) const {
    double value;
    while (true) {
        std::cout << CYAN << prompt << RESET;
        if (std::cin >> value && value >= min_val && value <= max_val) {
            return value;
        }
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << RED << BOLD << "[LỖI] Dữ liệu không hợp lệ. Vui lòng nhập số từ " 
                  << min_val << " đến " << max_val << "." << RESET << std::endl;
    }
}

} // namespace riskguard