#pragma once
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace riskguard {

class Dashboard {
public:
    // Khởi tạo Dashboard (Tự động kích hoạt ANSI và UTF-8 trên Windows)
    Dashboard();

    void drawHeader() const;
    
    // Hệ thống Menu Động (Dynamic Menu System)
    void showMenu(const std::vector<std::string>& options) const;
    
    void showTrainingProgress(int epoch, int totalEpochs, double loss, double accuracy) const;
    void displayAssessmentCard(double riskProb, std::string_view mainReason) const;

    // Bộ giáp phòng vệ nhập liệu (Safe Input Wrappers)
    int getSafeInt(std::string_view prompt, int min_val, int max_val) const;
    double getSafeDouble(std::string_view prompt, double min_val, double max_val) const;
};

} // namespace riskguard