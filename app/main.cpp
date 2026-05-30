#include "riskguard/NeuralNetwork.hpp"
#include "riskguard/utils/DataLoader.hpp"
#include "riskguard/utils/Dashboard.hpp"
#include "riskguard/layers/LinearLayer.hpp"
#include "riskguard/layers/SigmoidLayer.hpp"
#include <iostream>
#include <memory>
#include <thread>
#include <chrono>

using namespace riskguard;

void simulate_delay(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

int main() {
    try {
        Dashboard::drawHeader();
        simulate_delay(1000);

        // 1. Khởi tạo DataLoader và tải dữ liệu
        std::cout << "\n[BƯỚC 1] Nạp & Chuẩn hóa dữ liệu từ data/dataset.csv..." << std::endl;
        simulate_delay(1000);
        DataLoader loader;
        Matrix data = loader.loadAndNormalize("../data/dataset.csv");

        int num_samples = data.get_rows();
        if (num_samples == 0) {
            std::cerr << "[LỖI] Khong the doc du lieu hoac file rong." << std::endl;
            return 1;
        }
        std::cout << "  -> Đã tải thành công " << num_samples << " bản ghi." << std::endl;
        simulate_delay(1500);

        // 2. Chia dữ liệu thành X (4 features) và Y (1 label)
        std::cout << "\n[BƯỚC 2] Tiền xử lý & Phân chia Features/Labels..." << std::endl;
        simulate_delay(1000);
        Matrix X(num_samples, 4);
        Matrix Y(num_samples, 1);
        for (int i = 0; i < num_samples; ++i) {
            X(i, 0) = data(i, 0); // Income
            X(i, 1) = data(i, 1); // Debt
            X(i, 2) = data(i, 2); // Delinquency
            X(i, 3) = data(i, 3); // Age
            Y(i, 0) = data(i, 4); // Default
        }
        std::cout << "  -> Đã tách 4 features (Income, Debt, Delinq, Age) và 1 label (Default)." << std::endl;
        simulate_delay(1500);

        // 3. Khởi tạo mạng Nơ-ron cấu hình tuyến tính
        std::cout << "\n[BƯỚC 3] Khởi tạo kiến trúc Neural Network (4 -> 8 -> 1)..." << std::endl;
        simulate_delay(1000);
        NeuralNetwork net;
        net.add_layer(std::make_unique<LinearLayer>(4, 8));
        net.add_layer(std::make_unique<SigmoidLayer>());
        net.add_layer(std::make_unique<LinearLayer>(8, 1));
        net.add_layer(std::make_unique<SigmoidLayer>());
        std::cout << "  -> Khởi tạo trọng số hoàn tất." << std::endl;
        simulate_delay(1500);

        // 4. Huấn luyện mô hình (Training loop)
        int epochs = 10;
        double learning_rate = 0.01;
        std::cout << "\n[BƯỚC 4] Bắt đầu quá trình huấn luyện AI (Gradient Descent)..." << std::endl;
        std::cout << "  -> Epochs: " << epochs << " | Learning Rate: " << learning_rate << "\n" << std::endl;
        simulate_delay(1000);

        for (int epoch = 1; epoch <= epochs; ++epoch) {
            Matrix pred = net.forward(X);
            double loss = net.calculateBCELoss(pred, Y);
            Matrix grad = net.calculateBCEGradient(pred, Y);
            net.backward(grad);
            net.update_parameters(learning_rate);
            
            Dashboard::showTrainingProgress(epoch, epochs, loss);
            simulate_delay(800); // Tạm dừng để Dev nhìn rõ từng epoch
        }
        
        simulate_delay(1500);

        // 5. Thẩm định mô phỏng một khách hàng
        std::cout << "\n\n[BƯỚC 5] Thử nghiệm thực tế: Đánh giá hồ sơ khách hàng mới..." << std::endl;
        simulate_delay(1000);
        
        // Khách hàng giả định có thu nhập thấp, nợ cao, từng trễ hạn nợ
        Matrix test_customer(1, 4);
        test_customer(0, 0) = 0.2; // Thu nhập thấp
        test_customer(0, 1) = 0.8; // Nợ cao
        test_customer(0, 2) = 1.0; // Đã từng trễ hạn
        test_customer(0, 3) = 0.3; // Tuổi trẻ

        Matrix result = net.forward(test_customer);
        double risk_prob = result(0, 0);
        
        Dashboard::displayAssessmentCard(risk_prob, "Thu nhập thấp, dư nợ cao, lịch sử trả nợ xấu");

        std::cout << "\n[HOÀN TẤT] Pipeline đã chạy xong!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[FATAL EXCEPTION] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
