#pragma once
#include "NeuralNetwork.hpp"
#include <string>

// Quản lý Trí nhớ AI (Model Manager)
// Chịu trách nhiệm Lưu (Save) hoặc Tải (Load) các "Kinh nghiệm thẩm định" ra File văn bản
class ModelManager {
public:
    // Trích xuất trọng số thành file text để phân phối dùng cho các máy hoặc chi nhánh khác
    static void save(const NeuralNetwork& net, const std::string& path);
    
    // Tải trọng số từ file text, AI ngay lập tức "bừng tỉnh" và có thể Thẩm định ngay (Use case 3)
    // mà không tốn thời gian phải Training lại từ đầu (Use Case 2)!
    static void load(NeuralNetwork& net, const std::string& path);
};
