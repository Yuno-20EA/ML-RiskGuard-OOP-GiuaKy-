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

    std::vector<std::vector<double>> features_matrix;
    std::vector<std::vector<double>> labels_matrix;
    
    std::string line, val;
    bool isHeader = true; // Bỏ qua dòng tiêu đề

    while (std::getline(file, line)) { 
        if (line.empty()) continue; 
        
        if (isHeader) { isHeader = false; continue; }
        
        std::stringstream ss(line);
        std::vector<double> row_data;
        while (std::getline(ss, val, ',')) {
            try {
                row_data.push_back(std::stod(val));
            } catch (const std::exception& e) {
                row_data.push_back(0.0); // Giá trị dự phòng nếu dữ liệu hỏng
            }
        }
        
        // 1. BỔ SUNG NGƯỠNG CHẶN BIÊN AN TOÀN TRONG VÒNG LẶP ĐỌC CSV
        if (row_data.empty() || row_data.size() < 5) {
            continue; // Bỏ qua ngay lập tức dòng tiêu đề, dòng rác hoặc dòng trống ở cuối file
        }

        // 2. ĐỒNG BỘ LOGIC CẮT TỈA (DATA PRUNING) VÀ NẠP MA TRẬN
        constexpr double EPSILON = 1e-7;
        double raw_income = row_data[0];
        double raw_debt = row_data[1];
        double raw_delinquency = row_data[2];
        double raw_age = row_data[3];
        double label = row_data[4];

        double dti = (raw_income > EPSILON) ? (raw_debt / raw_income) : 999999.0;
        
        if (dti > 2.0 || raw_delinquency > 10.0 || raw_income <= EPSILON) {
            continue; // Loại bỏ hồ sơ nhiễu đồng bộ
        }

        // Chỉ khi vượt qua bộ lọc trên mới thực hiện push dữ liệu
        features_matrix.push_back({raw_income, raw_debt, raw_delinquency, raw_age});
        labels_matrix.push_back({label});
    }
    file.close();

    // 3. IN ĐỊNH LƯỢNG DEBUG KÍCH THƯỚC
    std::cout << "[DEBUG] DataLoader Matrix Dimensions - X: " << features_matrix.size() 
              << "x" << (features_matrix.empty() ? 0 : features_matrix[0].size()) 
              << " | Y: " << labels_matrix.size() << "x1\n";

    if (features_matrix.empty()) return Matrix(0, 0);

    int rows = static_cast<int>(features_matrix.size());
    int cols = 5; // 4 features + 1 label (giữ nguyên cấu trúc trả về gốc của hàm)
    Matrix result(rows, cols);

    // Ghép nối features và labels vào chung Matrix result
    for (int i = 0; i < rows; ++i) {
        result(i, 0) = features_matrix[i][0];
        result(i, 1) = features_matrix[i][1];
        result(i, 2) = features_matrix[i][2];
        result(i, 3) = features_matrix[i][3];
        result(i, 4) = labels_matrix[i][0];
    }
    
    return result; 
}

} // namespace riskguard