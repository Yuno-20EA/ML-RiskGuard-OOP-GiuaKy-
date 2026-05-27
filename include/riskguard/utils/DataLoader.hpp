#pragma once
#include "riskguard/core/Matrix.hpp"
#include <string>

class DataLoader {
public:
    Matrix loadAndNormalize(const std::string& filename);
};