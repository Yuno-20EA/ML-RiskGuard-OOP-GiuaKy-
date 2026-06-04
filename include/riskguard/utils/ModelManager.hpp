#pragma once
#include <string>
#include "riskguard/network/NeuralNetwork.hpp"
#include "riskguard/utils/DataPipeline.hpp"

namespace riskguard {

class ModelManager {
public:
    static bool saveModel(const std::string& filepath, const NeuralNetwork& model, const DataPipeline& pipeline);
    static bool loadModel(const std::string& filepath, NeuralNetwork& model, DataPipeline& pipeline);
};

} // namespace riskguard
