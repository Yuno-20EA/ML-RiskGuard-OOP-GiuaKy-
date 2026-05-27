#pragma once
#include <iostream>
#include <string>

class Dashboard {
public:
    static void drawHeader();
    void showMenu();
    void showTrainingProgress(int epoch, int totalEpochs, double loss);
    void displayAssessmentCard(double riskProb, const std::string& mainReason);
};