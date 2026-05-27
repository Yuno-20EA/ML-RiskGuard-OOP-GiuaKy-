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

### 2. Đọc file CSV và Chuẩn hóa dữ liệu (Min-Max Normalization)
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include "riskguard/core/Matrix.hpp" 

class DataLoader {
public:
    Matrix loadAndNormalize(const std::string& filename) {
        std::ifstream file(filename);
        
        if (!file.is_open()) {
            std::cerr << "Lỗi: Không thể mở file " << filename << std::endl;
            return Matrix(0, 0); 
        }

        std::vector<std::vector<double>> rawData;
        std::string line, val;

        while (std::getline(file, line)) { 
            if (line.empty()) continue; 
            
            std::stringstream ss(line);
            std::vector<double> row;
            while (std::getline(ss, val, ',')) {
                try {
                    row.push_back(std::stod(val));
                } catch (const std::exception& e) {
                    row.push_back(0.0); 
                }
            }
            if (!row.empty()) {
                rawData.push_back(row);
            }
        }
        file.close();

        if (rawData.empty()) {
            std::cerr << "Lỗi: File CSV không có dữ liệu!" << std::endl;
            return Matrix(0, 0);
        }

        int rows = rawData.size();
        int cols = rawData[0].size();
        Matrix result(rows, cols);

        for (int j = 0; j < cols; ++j) { 
            double minVal = rawData[0][j];
            double maxVal = rawData[0][j];
            
            for (int i = 1; i < rows; ++i) {
                minVal = std::min(minVal, rawData[i][j]);
                maxVal = std::max(maxVal, rawData[i][j]);
            }

            for (int i = 0; i < rows; ++i) {
                double normalizedVal = (maxVal == minVal) ? 0.0 : (rawData[i][j] - minVal) / (maxVal - minVal);
                result.at(i, j) = normalizedVal; 
            }
        }
        return result;
    }
};