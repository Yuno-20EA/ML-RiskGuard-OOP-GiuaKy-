#include "riskguard/utils/DataLoader.hpp"
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

namespace riskguard {

Matrix DataLoader::loadRawCSV(const std::string& filename) {
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Lỗi: Không thể mở file " << filename << std::endl;
        return Matrix(0, 0); 
    }

    std::vector<std::vector<double>> rawData;
    std::string line, val;
    bool isHeader = true; // Bỏ qua dòng tiêu đề

    while (std::getline(file, line)) { 
        if (line.empty()) continue; 
        
        if (isHeader) { isHeader = false; continue; }
        
        std::stringstream ss(line);
        std::vector<double> row;
        while (std::getline(ss, val, ',')) {
            try {
                row.push_back(std::stod(val));
            } catch (const std::exception& e) {
                row.push_back(0.0); // Giá trị dự phòng nếu dữ liệu hỏng
            }
        }
        
        if (!row.empty() && row.size() >= 3) {
            constexpr double EPSILON = 1e-7;
            double raw_income = row[0];
            double raw_debt = row[1];
            double raw_delinquency = row[2];
            
            double dti = (raw_income > EPSILON) ? (raw_debt / raw_income) : 999999.0;
            
            if (dti > 2.0 || raw_delinquency > 10.0 || raw_income <= EPSILON) {
                continue; // Lọc ngầm: Bỏ qua dòng dữ liệu rác/vượt ngưỡng
            }
            rawData.push_back(row);
        } else if (!row.empty()) {
            rawData.push_back(row);
        }
    }
    file.close();

    if (rawData.empty()) return Matrix(0, 0);

    int rows = static_cast<int>(rawData.size());
    int cols = static_cast<int>(rawData[0].size()); 
    Matrix result(rows, cols);

    // Điền dữ liệu trực tiếp vào ma trận mà không qua Min-Max Scaling
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (static_cast<size_t>(j) < rawData[i].size()) {
                result(i, j) = rawData[i][j];
            } else {
                result(i, j) = 0.0;
            }
        }
    }
    return result; 
}

} // namespace riskguard