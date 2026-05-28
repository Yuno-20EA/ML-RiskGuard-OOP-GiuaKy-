#include "../layers/Linelayer.hpp"

// Khởi tạo trọng số ngẫu nhiên (tránh bị kẹt mạng) và bias bằng 0 lúc mới bắt đầu
LinearLayer::LinearLayer(int in_features, int out_features) 
    : weights(Matrix::random(in_features, out_features)), 
      biases(Matrix::zeros(1, out_features)),
      d_weights(in_features, out_features),
      d_biases(1, out_features) {}

// Lan truyền tiến (Forward): Tính toán toán học từ đầu vào
Matrix LinearLayer::forward(const Matrix& input) {
    last_input = input; // Ghi nhớ lại đầu vào để lát sau dùng học ngược
    
    // Nhân ma trận: Giá trị chỉ số KH * Mức độ quan trọng (Trọng số)
    Matrix result = input.multiply(weights);
    
    // Cộng thêm Bias vào mỗi hàng (Mỗi hàng đại diện cho 1 khách hàng)
    for(int i=0; i<result.rows; ++i) {
        for(int j=0; j<result.cols; ++j) {
            result.at(i, j) += biases.at(0, j);
        }
    }
    return result;
}

// Lan truyền ngược (Backward): Tìm ra lớp này đóng góp bao nhiêu phần trăm vào cái Sai số
Matrix LinearLayer::backward(const Matrix& grad) {
    // Gradient của Weights = (Đầu vào chuyển vị) nhân với Gradient tổng
    d_weights = last_input.transpose().multiply(grad);
    
    // Gradient của Biases = Tổng của Gradient tổng theo các cột
    d_biases = Matrix::zeros(1, grad.cols);
    for(int i=0; i<grad.rows; ++i) {
        for(int j=0; j<grad.cols; ++j) {
            d_biases.at(0, j) += grad.at(i, j);
        }
    }
    
    // Trả ngược lại sai số cho Input, để truyền lùi về lớp phía trước nó (nếu có)
    return grad.multiply(weights.transpose());
}

// Lấy "Bộ não" (Weights, Biases) ra để lưu
std::vector<Matrix*> LinearLayer::get_parameters() {
    return {&weights, &biases};
}

// Cập nhật trí nhớ: AI sẽ tự trừ đi đạo hàm nhân với Tốc độ học (Learning Rate) 
// Để tiến sát tới kết quả chuẩn xác nhất (Giảm sai số)
void LinearLayer::update_parameters(double lr) {
    for(size_t i=0; i<weights.data_.size(); ++i) {
        weights.data_[i] -= lr * d_weights.data_[i];
    }
    for(size_t i=0; i<biases.data_.size(); ++i) {
        biases.data_[i] -= lr * d_biases.data_[i];
    }
}
