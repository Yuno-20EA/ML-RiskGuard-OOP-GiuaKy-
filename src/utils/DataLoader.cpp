#include "riskguard/utils/DataLoader.hpp"
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <limits> // Cần để lấy giá trị cực đại/cực tiểu làm mốc

/**
 * @brief Đọc file CSV và chuẩn hóa dữ liệu về khoảng [0, 1].
 * Lưu ý: Hàm này tự động bỏ qua dòng đầu tiên (header) và 
 * có cơ chế chống crash khi dữ liệu không đồng nhất.
 */
namespace riskguard {

Matrix DataLoader::loadAndNormalize(const std::string& filename) {
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Lỗi: Không thể mở file " << filename << std::endl;
        return Matrix(0, 0); 
    }

    std::vector<std::vector<double>> rawData;
    std::string line, val;
    bool isHeader = true; // Cờ đánh dấu để bỏ qua dòng tiêu đề

    while (std::getline(file, line)) { 
        if (line.empty()) continue; 
        
        // Bỏ qua dòng đầu tiên vì thường là tên cột (string không thể convert sang double)
        if (isHeader) { isHeader = false; continue; }
        
        std::stringstream ss(line);
        std::vector<double> row;
        while (std::getline(ss, val, ',')) {
            try {
                row.push_back(std::stod(val));
            } catch (const std::exception& e) {
                row.push_back(0.0); // Giá trị mặc định nếu dữ liệu hỏng
            }
        }
        if (!row.empty()) rawData.push_back(row);
    }
    file.close();

    if (rawData.empty()) return Matrix(0, 0);

    int rows = rawData.size();
    // Giả định số cột dựa trên dòng đầu tiên có dữ liệu
    int cols = rawData[0].size(); 
    Matrix result(rows, cols);

    // Chuẩn hóa theo từng cột (Feature Scaling)
    for (int j = 0; j < cols; ++j) { 
        // Dùng giới hạn của double để tìm min/max chính xác
        double minVal = std::numeric_limits<double>::max();
        double maxVal = std::numeric_limits<double>::lowest();
        
        // Kiểm tra an toàn: Duyệt qua các hàng và kiểm tra size để tránh out-of-bounds
        for (int i = 0; i < rows; ++i) {
            if (static_cast<size_t>(j) < rawData[i].size()) {
                minVal = std::min(minVal, rawData[i][j]);
                maxVal = std::max(maxVal, rawData[i][j]);
            }
        }

        // Thực hiện chuẩn hóa Min-Max: (x - min) / (max - min)
        for (int i = 0; i < rows; ++i) {
            if (static_cast<size_t>(j) < rawData[i].size()) {
                // Kiểm tra (maxVal == minVal) để tránh lỗi chia 0
                double normalizedVal = (maxVal == minVal) ? 0.0 : (rawData[i][j] - minVal) / (maxVal - minVal);
                result(i, j) = normalizedVal; 
            }
        }
    }
    return result; 
}

} // namespace riskguard