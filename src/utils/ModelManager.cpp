#include "riskguard/utils/ModelManager.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

namespace riskguard {

// Một hàm helper nhỏ để escape/format JSON
bool ModelManager::saveModel(const std::string& filepath, const NeuralNetwork& model, const DataPipeline& pipeline) {
    std::ofstream out(filepath);
    if (!out.is_open()) return false;

    out << "{\n";
    out << "  \"version\": \"2.1.0\",\n";
    out << "  \"arch\": \"4-8-1\",\n";
    out << "  \"pipeline\": {\n";
    out << "    \"income_mean\": " << pipeline.get_income_mean() << ",\n";
    out << "    \"income_std_dev\": " << pipeline.get_income_std_dev() << ",\n";
    out << "    \"debt_mean\": " << pipeline.get_debt_mean() << ",\n";
    out << "    \"debt_std_dev\": " << pipeline.get_debt_std_dev() << ",\n";
    out << "    \"delinquency_mean\": " << pipeline.get_delinquency_mean() << ",\n";
    out << "    \"delinquency_std_dev\": " << pipeline.get_delinquency_std_dev() << ",\n";
    out << "    \"age_mean\": " << pipeline.get_age_mean() << ",\n";
    out << "    \"age_std_dev\": " << pipeline.get_age_std_dev() << "\n";
    out << "  },\n";

    out << "  \"layers\": [\n";
    const auto& layers = model.get_layers();
    for (size_t i = 0; i < layers.size(); ++i) {
        out << "    {\n";
        out << "      \"type\": \"" << layers[i]->get_type_name() << "\"";
        
        auto params = layers[i]->get_parameters();
        if (!params.empty()) {
            out << ",\n      \"parameters\": [\n";
            for (size_t j = 0; j < params.size(); ++j) {
                out << "        {\n";
                out << "          \"rows\": " << params[j]->get_rows() << ",\n";
                out << "          \"cols\": " << params[j]->get_cols() << ",\n";
                out << "          \"data\": [";
                const auto& data = params[j]->get_data();
                for (size_t k = 0; k < data.size(); ++k) {
                    out << data[k];
                    if (k < data.size() - 1) out << ", ";
                }
                out << "]\n        }";
                if (j < params.size() - 1) out << ",";
                out << "\n";
            }
            out << "      ]\n";
        } else {
            out << "\n";
        }
        out << "    }";
        if (i < layers.size() - 1) out << ",";
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";

    out.close();
    return true;
}

// Hàm nạp cực kỳ đơn giản (Zero-dependency parser)
// Vì định dạng ta tự sinh ra rất cố định, ta có thể parse tuần tự bằng string search
// Để an toàn và nhanh gọn, ta sẽ bỏ qua cấu trúc JSON lồng nhau phức tạp mà quét tìm thẳng giá trị
bool ModelManager::loadModel(const std::string& filepath, NeuralNetwork& model, DataPipeline& pipeline) {
    std::ifstream in(filepath);
    if (!in.is_open()) return false;

    std::stringstream buffer;
    buffer << in.rdbuf();
    std::string content = buffer.str();

    // ── Kiểm tra version & arch compatibility ──────────────────────────────
    // Nếu arch không khớp → model.json lỗi thời, cần huấn luyện lại
    auto extract_string = [&](const std::string& key) -> std::string {
        size_t pos = content.find('"' + key + "\":");
        if (pos == std::string::npos) return {};
        size_t q1 = content.find('"', pos + key.size() + 3);
        if (q1 == std::string::npos) return {};
        size_t q2 = content.find('"', q1 + 1);
        if (q2 == std::string::npos) return {};
        return content.substr(q1 + 1, q2 - q1 - 1);
    };

    std::string saved_arch = extract_string("arch");
    if (!saved_arch.empty() && saved_arch != "4-8-1") {
        // Kiến trúc không khớp → từ chối nạp, buộc huấn luyện lại
        return false;
    }

    auto extract_double = [&](const std::string& key) -> double {
        size_t pos = content.find("\"" + key + "\":");
        if (pos != std::string::npos) {
            size_t start = pos + key.length() + 3;
            return std::stod(content.substr(start));
        }
        return 0.0;
    };

    // Load pipeline stats
    pipeline.set_income_params(extract_double("income_mean"), extract_double("income_std_dev"));
    pipeline.set_debt_params(extract_double("debt_mean"), extract_double("debt_std_dev"));
    pipeline.set_delinquency_params(extract_double("delinquency_mean"), extract_double("delinquency_std_dev"));
    pipeline.set_age_params(extract_double("age_mean"), extract_double("age_std_dev"));

    // Load layers (Cần tìm tuần tự các khối "data": [ ])
    size_t search_pos = 0;
    const auto& layers = model.get_layers();
    for (const auto& layer : layers) {
        auto params = layer->get_parameters();
        for (auto* matrix : params) {
            size_t data_pos = content.find("\"data\": [", search_pos);
            if (data_pos == std::string::npos) return false;
            
            data_pos += 9; // Bỏ qua "\"data\": ["
            size_t end_pos = content.find("]", data_pos);
            if (end_pos == std::string::npos) return false;
            
            std::string data_str = content.substr(data_pos, end_pos - data_pos);
            std::stringstream ss(data_str);
            std::string val_str;
            
            auto& matrix_data = matrix->get_data();
            size_t idx = 0;
            while (std::getline(ss, val_str, ',')) {
                if (idx < matrix_data.size()) {
                    matrix_data[idx] = std::stod(val_str);
                    idx++;
                }
            }
            search_pos = end_pos;
        }
    }

    return true;
}

} // namespace riskguard
