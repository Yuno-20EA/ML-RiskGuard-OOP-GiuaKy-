#include "../layers/ModelManager.hpp"
#include <fstream>
#include <stdexcept>
#include <iostream>

void ModelManager::save(const NeuralNetwork& net, const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) throw std::runtime_error("Không thể mở file để lưu Bộ não AI");
    
    std::cout << "[Hệ thống] Đang xuất các kinh nghiệm thẩm định ra file: " << path << "...\n";
    
    for (const auto& layer : net.pipeline) {
        auto params = layer->get_parameters();
        if (params.empty()) continue;
        
        ofs << layer->get_type_name() << "\n";
        for (Matrix* p : params) {
            ofs << p->rows << " " << p->cols << "\n";
            for (double val : p->data_) {
                ofs << val << " ";
            }
            ofs << "\n";
        }
    }
    ofs.close();
    std::cout << "[Hệ thống] Xuất file thành công. Có thể dùng lại mà không cần huấn luyện!\n";
}

void ModelManager::load(NeuralNetwork& net, const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) throw std::runtime_error("Không thể đọc file chứa Bộ não AI");

    std::cout << "[Hệ thống] Đang nạp kinh nghiệm thẩm định từ file: " << path << "...\n";

    for (const auto& layer : net.pipeline) {
        auto params = layer->get_parameters();
        if (params.empty()) continue;

        std::string name;
        ifs >> name;
        if (name != layer->get_type_name()) {
            throw std::runtime_error("Cấu trúc mạng không khớp với dữ liệu đã lưu!");
        }

        for (Matrix* p : params) {
            int r, c;
            ifs >> r >> c;
            if (r != p->rows || c != p->cols) {
                throw std::runtime_error("Kích thước ma trận trọng số (weights/biases) không khớp!");
            }
            for (size_t i = 0; i < p->data_.size(); ++i) {
                ifs >> p->data_[i];
            }
        }
    }
    ifs.close();
    std::cout << "[Hệ thống] Nạp thành công. Sẵn sàng cho Use Case 3: Thẩm định hồ sơ cấp tốc!\n";
}
