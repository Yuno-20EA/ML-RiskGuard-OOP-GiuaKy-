#include "riskguard/utils/DataLoader.hpp"
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>

Matrix DataLoader::loadAndNormalize(const std::string& filename) {
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