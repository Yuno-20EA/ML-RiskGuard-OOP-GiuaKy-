#include "riskguard/utils/DataLoader.hpp"
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdint>
#include <algorithm>

namespace riskguard {

// ── Hàm nội bộ: Chuyển đổi UTF-16 LE sang UTF-8 (Pure C++20, zero-dependency) ──
static std::string utf16le_to_utf8(const std::vector<uint8_t>& bytes, size_t offset) {
    std::string result;
    result.reserve(bytes.size() / 2); // Ước lượng kích thước đầu ra

    for (size_t i = offset; i + 1 < bytes.size(); i += 2) {
        // Ghép 2 byte thành một code unit UTF-16 LE
        uint16_t code_unit = static_cast<uint16_t>(bytes[i]) 
                           | (static_cast<uint16_t>(bytes[i + 1]) << 8);

        if (code_unit < 0x80) {
            // ASCII thuần → 1 byte UTF-8
            result.push_back(static_cast<char>(code_unit));
        } else if (code_unit < 0x800) {
            // 2 byte UTF-8
            result.push_back(static_cast<char>(0xC0 | (code_unit >> 6)));
            result.push_back(static_cast<char>(0x80 | (code_unit & 0x3F)));
        } else {
            // 3 byte UTF-8 (bao phủ phần lớn BMP)
            result.push_back(static_cast<char>(0xE0 | (code_unit >> 12)));
            result.push_back(static_cast<char>(0x80 | ((code_unit >> 6) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | (code_unit & 0x3F)));
        }
    }
    return result;
}

// ── Hàm nội bộ: Đọc toàn bộ file thành chuỗi UTF-8 (tự động phát hiện encoding) ──
static std::string read_file_as_utf8(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return {};

    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    file.read(reinterpret_cast<char*>(bytes.data()), size);
    file.close();

    if (bytes.empty()) return {};

    // Phát hiện BOM (Byte Order Mark)
    // UTF-16 LE BOM: FF FE
    if (bytes.size() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xFE) {
        return utf16le_to_utf8(bytes, 2); // Bỏ 2 byte BOM
    }

    // UTF-8 BOM: EF BB BF
    size_t offset = 0;
    if (bytes.size() >= 3 && bytes[0] == 0xEF && bytes[1] == 0xBB && bytes[2] == 0xBF) {
        offset = 3; // Bỏ 3 byte BOM
    }

    // Trả về nguyên bản (UTF-8 hoặc ASCII thuần)
    return std::string(bytes.begin() + offset, bytes.end());
}

Matrix DataLoader::loadRawCSV(const std::string& filename) {
    std::string content = read_file_as_utf8(filename);

    if (content.empty()) {
        std::cerr << "[LỖI] Không thể mở hoặc file rỗng: " << filename << std::endl;
        return Matrix(0, 0);
    }

    std::vector<std::vector<double>> features_matrix;
    std::vector<std::vector<double>> labels_matrix;

    std::istringstream stream(content);
    std::string line, val;

    int line_count = 0;
    int skip_size = 0;
    int skip_prune = 0;
    int skip_parse = 0;

    while (std::getline(stream, line)) {
        // Bỏ qua dòng tiêu đề (dòng 0)
        if (line_count == 0) {
            line_count++;
            continue;
        }

        // Bỏ ký tự \r cuối dòng nếu có (CRLF → LF)
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
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
                break;
            }
        }

        if (parse_error) {
            skip_parse++;
            line_count++;
            continue;
        }

        // Ghi Log chi tiết cho 5 dòng ĐẦU TIÊN của dữ liệu
        if (line_count <= 5) {
            std::cout << "[LOG] Dòng " << line_count
                      << " - Kích thước row_data: " << row_data.size();
            if (row_data.size() >= 5) {
                std::cout << " | Income=" << row_data[0]
                          << " Debt=" << row_data[1]
                          << " Del=" << row_data[2]
                          << " Age=" << row_data[3]
                          << " Label=" << row_data[4];
            }
            std::cout << "\n";
        }

        // Chặn biên an toàn: mỗi dòng hợp lệ phải có đúng 5 cột
        if (row_data.empty() || row_data.size() < 5) {
            skip_size++;
            line_count++;
            continue;
        }

        // Data Pruning: Loại bỏ cực trị dị biệt
        constexpr double EPSILON = 1e-7;
        double raw_income      = row_data[0];
        double raw_debt        = row_data[1];
        double raw_delinquency = row_data[2];
        double raw_age         = row_data[3];
        double label           = row_data[4];

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

        // Chỉ khi vượt qua bộ lọc mới nạp vào ma trận
        features_matrix.push_back({raw_income, raw_debt, raw_delinquency, raw_age});
        labels_matrix.push_back({label});

        line_count++;
    }

    // Báo cáo kiểm toán
    std::cout << "[AUDIT] Tổng số dòng đọc được: " << line_count << "\n"
              << " -> Bị loại do thiếu cột (<5): " << skip_size << "\n"
              << " -> Bị loại do lỗi parse số: " << skip_parse << "\n"
              << " -> Bị loại do bộ lọc cực trị (Pruning): " << skip_prune << "\n";

    std::cout << "[DEBUG] DataLoader Matrix Dimensions - X: " << features_matrix.size()
              << "x" << (features_matrix.empty() ? 0 : features_matrix[0].size())
              << " | Y: " << labels_matrix.size() << "x1\n";

    if (features_matrix.empty()) return Matrix(0, 0);

    int rows = static_cast<int>(features_matrix.size());
    int cols = 5; // 4 features + 1 label
    Matrix result(rows, cols);

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