#include "test_runner.hpp"
#include "logger.hpp"
#include "../include/riskguard/utils/DataLoader.hpp" 
#include <stdexcept>
#include <fstream>
#include <cstdio>

using namespace riskguard;

class FileNotFoundTest : public TestCase {
public:
    FileNotFoundTest() : TestCase("DataProcessingTest", "test_file_not_found") {}
    void run_logic() override {
        std::string fake_filename = "no_exist.csv";
        DataLoader loader;
        bool handled = false;

        Logger::profile("Kiểm tra xử lý ngoại lệ khi thiếu File CSV", [&]() {
            Matrix m = loader.loadAndNormalize(fake_filename);
            if (m.get_rows() == 0 && m.get_cols() == 0) {
                handled = true; 
            }
        });

        if (!handled) {
            throw std::runtime_error("Lỗi: Hệ thống không trả về Matrix rỗng khi thiếu file!");
        }
    }
};

class MinMaxScalerTest : public TestCase {
public:
    MinMaxScalerTest() : TestCase("DataProcessingTest", "test_min_max_scaler") {}
    void run_logic() override {
        // Create a dummy CSV for testing
        std::string temp_csv = "temp_test.csv";
        std::ofstream out(temp_csv);
        out << "Header\n0.0\n50.0\n100.0\n";
        out.close();

        DataLoader loader;
        Matrix scaled_data = Logger::profile("Chuẩn hóa dữ liệu Min-Max [0, 1]", [&]() {
            return loader.loadAndNormalize(temp_csv);
        });

        std::remove(temp_csv.c_str());

        OOP_ASSERT_EQ(scaled_data.get_rows(), 3);
        OOP_ASSERT_EQ(scaled_data.get_cols(), 1);
        OOP_ASSERT_NEAR(scaled_data(0,0), 0.0, 1e-5);
        OOP_ASSERT_NEAR(scaled_data(1,0), 0.5, 1e-5);
        OOP_ASSERT_NEAR(scaled_data(2,0), 1.0, 1e-5);
    }
};
