#pragma once
#include "../include/core/layer.hpp"
#include <memory>
#include <vector>

// Bộ não Trung Tâm (Neural Network) 
// Nơi lắp ráp các Tầng (Layers) lại với nhau để tạo thành một luồng Thẩm định tín dụng khép kín
class NeuralNetwork {
public:
    // Pipeline: Một "đường ống" chứa các tầng tính toán tuần tự (Lớp tuyến tính -> Lớp Kích hoạt -> ...)
    std::vector<std::unique_ptr<Layer>> pipeline;
    
    // Tốc độ học (Learning Rate): AI sẽ "sửa sai" chậm rãi an toàn, hay sửa nhanh nhưng dễ vượt quá mức
    double learning_rate;

    NeuralNetwork(double lr = 0.01) : learning_rate(lr) {}

    // Lắp ghép thêm một Tầng mới vào kiến trúc
    void add_layer(std::unique_ptr<Layer> layer);
    
    // Đẩy dữ liệu qua đường ống để ra tỷ lệ rủi ro (Lan truyền tiến)
    Matrix forward(const Matrix& input);
    
    // Trả lỗi về hệ thống để AI tự sửa sai (Lan truyền lùi)
    void backward(const Matrix& loss_grad);
    
    // Use Case 2: Vòng lặp Huấn luyện (Đưa cho AI bộ dữ liệu Lịch sử để nó tự rút kinh nghiệm)
    void train(const Matrix& X, const Matrix& y, int epochs);
    
    // Use Case 3: Thẩm định hồ sơ tức thì (Nhập khách hàng mới -> Báo phần trăm rủi ro)
    Matrix predict(const Matrix& X);
};
