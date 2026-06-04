#pragma once
#include "riskguard/core/Matrix.hpp"
#include <string>
#include <vector> 

namespace riskguard {

class DataLoader {
public:
    /**
     * @brief Đọc dữ liệu từ file CSV thô (không chuẩn hóa).
     * @param filename Đường dẫn đến file CSV cần đọc.
     * @return Một đối tượng Matrix chứa dữ liệu thô.
     */
    Matrix loadRawCSV(const std::string& filename);
};

} // namespace riskguard