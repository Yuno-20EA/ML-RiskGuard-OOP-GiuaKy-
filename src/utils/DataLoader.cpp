#include "riskguard/utils/DataLoader.hpp"
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>

Matrix DataLoader::loadAndNormalize(const std::string& filename) {
    std::ifstream file(filename);
    
    // Kiểm tra xem file có tồn tại và mở được không
    if (!file.is_open()) {
        std::cerr << "Lỗi: Không thể mở file " << filename << std::endl;
        return Matrix(0, 0); 
    }

    std::vector<std::vector<double>> rawData;
    std::string line, val;

    // Đọc từng dòng của file CSV
    while (std::getline(file, line)) { 
        if (line.empty()) continue; 
        
        std::stringstream ss(line);
        std::vector<double> row;
        // Tách các giá trị dựa trên dấu phẩy
        while (std::getline(ss, val, ',')) {
            try {
                row.push_back(std::stod(val)); // Chuyển đổi chuỗi sang số thực
            } catch (const std::exception& e) {
                row.push_back(0.0); // Nếu lỗi định dạng, gán giá trị mặc định là 0
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

    // Thực hiện chuẩn hóa theo từng cột (Feature Scaling)
    for (int j = 0; j < cols; ++j) { 
        // Tìm giá trị nhỏ nhất và lớn nhất của cột để tính toán Min-Max
        double minVal = rawData[0][j];
        double maxVal = rawData[0][j];
        
        for (int i = 1; i < rows; ++i) {
            minVal = std::min(minVal, rawData[i][j]);
            maxVal = std::max(maxVal, rawData[i][j]);
        }

        // Áp dụng công thức chuẩn hóa: (x - min) / (max - min)
        for (int i = 0; i < rows; ++i) {
            double normalizedVal = (maxVal == minVal) ? 0.0 : (rawData[i][j] - minVal) / (maxVal - minVal);
            result.at(i, j) = normalizedVal; 
        }
    }
    return result; 
}