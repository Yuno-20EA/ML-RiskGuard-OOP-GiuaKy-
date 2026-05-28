#include "../layers/NeuralNetwork.hpp"
#include "../layers/LossFunction.hpp"
#include <iostream>
#include <iomanip>

void NeuralNetwork::add_layer(std::unique_ptr<Layer> layer) {
    pipeline.push_back(std::move(layer));
}

Matrix NeuralNetwork::forward(const Matrix& input) {
    Matrix current = input;
    // Chạy hồ sơ qua từng tầng (Lớp ẩn tính toán đặc trưng, Lớp Sigmoid chốt tỷ lệ)
    for (auto& layer : pipeline) {
        current = layer->forward(current);
    }
    return current; // Trả về phần trăm rủi ro (0.0 -> 1.0)
}

void NeuralNetwork::backward(const Matrix& loss_grad) {
    Matrix grad = loss_grad;
    // Tự truyền ngược lỗi (Backpropagation) để sửa lại góc nhìn của mình
    for (auto it = pipeline.rbegin(); it != pipeline.rend(); ++it) {
        grad = (*it)->backward(grad);
        (*it)->update_parameters(learning_rate);
    }
}

void NeuralNetwork::train(const Matrix& X, const Matrix& y, int epochs) {
    // Sử dụng BCELoss cho phân loại nhị phân (Rủi ro bùng nợ 0% - 100%)
    BCELoss loss_fn;
    
    std::cout << "--- USE CASE 2: HUẤN LUYỆN BỘ NÃO AI ---\n";
    std::cout << "Bắt đầu vòng lặp tự học từ dữ liệu lịch sử...\n";
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // 1. Máy tự đoán nhãn (Xác suất rủi ro dự kiến)
        Matrix pred = forward(X);
        
        // 2. So sánh với thực tế để tính sai số
        double loss = loss_fn.compute(pred, y);
        
        // Tính thêm độ chính xác (Accuracy) cho dễ hình dung
        double correct = 0.0;
        for(int i = 0; i < pred.rows; ++i) {
            // Nếu rủi ro >= 50% thì hệ thống đánh giá là có nguy cơ (1), ngược lại là an toàn (0)
            double p = pred.data_[i];
            double t = y.data_[i];
            if ((p >= 0.5 && t == 1.0) || (p < 0.5 && t == 0.0)) {
                correct += 1.0;
            }
        }
        double accuracy = (correct / pred.rows) * 100.0;
        
        // 3. Tự truyền ngược lỗi về trước để điều chỉnh trọng số (sửa sai)
        Matrix grad = loss_fn.gradient(pred, y);
        backward(grad);
        
        // 4. Màn hình liên tục in ra tiến độ: Sai số giảm, độ chính xác tăng
        if (epoch == 0 || (epoch + 1) % 100 == 0) {
            std::cout << "Vòng học [" << std::setw(4) << epoch + 1 << "/" << epochs << "] "
                      << ">> Sai số (Loss): " << std::fixed << std::setprecision(4) << loss << " "
                      << ">> Độ chính xác: " << std::setprecision(2) << accuracy << "%\n";
        }
    }
    std::cout << "--- HOÀN TẤT HUẤN LUYỆN ---\n\n";
}

Matrix NeuralNetwork::predict(const Matrix& X) {
    // USE CASE 3: Thẩm định hồ sơ cấp tốc
    // Hệ thống chạy qua các tầng tính toán đã được học và trả về ngay lập tức
    return forward(X);
}
