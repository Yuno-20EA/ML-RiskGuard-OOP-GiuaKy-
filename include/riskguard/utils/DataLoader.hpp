#pragma once
#include "riskguard/core/Matrix.hpp"
#include <string>
#include <vector> // Thêm vào để đảm bảo bạn có thể sử dụng vector nếu cần trong .cpp

namespace riskguard {

class DataLoader {
public:
    /**
     * @brief Đọc dữ liệu từ file CSV và thực hiện chuẩn hóa Min-Max.
     * @param filename Đường dẫn đến file CSV cần đọc.
     * @return Một đối tượng Matrix chứa dữ liệu đã chuẩn hóa.
     */
    Matrix loadAndNormalize(const std::string& filename);
};

} // namespace riskguard