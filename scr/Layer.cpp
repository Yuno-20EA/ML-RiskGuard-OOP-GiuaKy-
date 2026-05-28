#include "../include/core/layer.hpp"

// Khởi tạo và hủy mặc định
Layer::~Layer() = default;

// Mặc định các tầng không có tham số (như ReLU, Sigmoid) sẽ trả về mảng rỗng
// Các tầng tuyến tính (LinearLayer) ghi đè hàm này để trả về weights và biases
std::vector<Matrix*> Layer::get_parameters() { 
    return {}; 
}

// Mặc định các tầng không có tham số sẽ không cần cập nhật gì cả
void Layer::update_parameters(double lr) {
    // Không làm gì cả
}
