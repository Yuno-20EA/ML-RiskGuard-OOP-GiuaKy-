#pragma once
#include <iostream>
#include <string>
#include <vector> // Thêm để dự phòng nếu sau này cần truyền tham số dạng vector

class Dashboard {
public:
    // Dùng static cho các hàm không truy cập biến thành viên giúp tiết kiệm bộ nhớ
    static void drawHeader();
    static void showMenu();
    static void showTrainingProgress(int epoch, int totalEpochs, double loss);
    static void displayAssessmentCard(double riskProb, const std::string& mainReason);
};