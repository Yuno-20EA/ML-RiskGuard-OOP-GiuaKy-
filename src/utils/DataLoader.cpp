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
    
    int line_count = 0;
    int skip_size = 0;
    int skip_prune = 0;
    int skip_parse = 0;

    while (std::getline(file, line)) { 
        if (line_count == 0) { 
            line_count++; 
            continue; 
        }
        
        if (line.empty()) {
            line_count++;
            continue; 
        }
        
        std::stringstream ss(line);
        std::vector<double> row_data;
        bool parse_error = false;
        
        while (std::getline(ss, val, ',')) {
            try {
                row_data.push_back(std::stod(val));
            } catch (const std::exception& e) {
                parse_error = true;
                break; // Lỗi parse, dừng xử lý dòng này ngay lập tức
            }
        }
        
        if (parse_error) {
            skip_parse++;
            line_count++;
            continue;
        }

        // Ghi Log chi tiết cho 5 dòng ĐẦU TIÊN của dữ liệu
        if (line_count <= 5) {
            std::cout << "[LOG] Dòng " << line_count << " - Kích thước row_data: " << row_data.size() << "\n";
        }
        
        if (row_data.empty() || row_data.size() < 5) {
            skip_size++;
            line_count++;
            continue;
        }

        constexpr double EPSILON = 1e-7;
        double raw_income = row_data[0];
        double raw_debt = row_data[1];
        double raw_delinquency = row_data[2];
        double raw_age = row_data[3];
        double label = row_data[4];

        double dti = (raw_income > EPSILON) ? (raw_debt / raw_income) : 999999.0;
        
        if (dti > 2.0 || raw_delinquency > 10.0 || raw_income <= EPSILON) {
            if (line_count <= 5) {
                std::cout << "[LOG] -> Bị Pruned | Income: " << raw_income 
                          << ", Debt: " << raw_debt 
                          << ", Delinquency: " << raw_delinquency 
                          << ", DTI: " << dti << "\n";
            }
            skip_prune++;
            line_count++;
            continue;
        }

        // Chỉ khi vượt qua bộ lọc trên mới thực hiện push dữ liệu
        features_matrix.push_back({raw_income, raw_debt, raw_delinquency, raw_age});
        labels_matrix.push_back({label});
        
        line_count++;
    }
    file.close();

    // 5. IN ĐỊNH LƯỢNG DEBUG KÍCH THƯỚC VÀ KIỂM TOÁN
    std::cout << "[AUDIT] Tổng số dòng đọc được: " << line_count << "\n"
              << " -> Bị loại do thiếu cột (<5): " << skip_size << "\n"
              << " -> Bị loại do lỗi parse số: " << skip_parse << "\n"
              << " -> Bị loại do bộ lọc cực trị (Pruning): " << skip_prune << "\n";

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