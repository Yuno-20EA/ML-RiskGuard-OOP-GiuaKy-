#1. Màn hình hiển thị (Console Dashboard)
#include <iostream>
#include <string>
#define RESET   "\033[0m"
#define RED     "\033[31m"   
#define GREEN   "\033[32m"   
#define YELLOW  "\033[33m"  
#define CYAN    "\033[36m"      

class Dashboard {
public:
    void showMenu() {
        std::cout << CYAN << "\n=== RISKGUARD ML SYSTEM ===" << RESET << std::endl;
        std::cout << "1. Nạp & Chuẩn hóa dữ liệu (Use Case 1)" << std::endl;
        std::cout << "2. Huấn luyện bộ não AI (Use Case 2)" << std::endl;
        std::cout << "3. Thẩm định hồ sơ cấp tốc (Use Case 3)" << std::endl;
        std::cout << "4. Thoát" << std::endl;
        std::cout << "Lựa chọn của bạn: ";
    }

    void displayResult(double riskProbability) {
        std::cout << "\n--- KẾT QUẢ THẨM ĐỊNH ---" << std::endl;
        std::cout << "Xác suất rủi ro: " << (riskProbability * 100) << "%" << std::endl;

        if (riskProbability < 0.5) {
            std::cout << "Trạng thái: " << GREEN << "[DUYỆT VAY - AN TOÀN]" << RESET << std::endl;
        } else {
            std::cout << "Trạng thái: " << RED << "[TỪ CHỐI - RỦI RO]" << RESET << std::endl;
        }
    }
};