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
            Matrix m = loader.loadRawCSV(fake_filename);
            if (m.get_rows() == 0 && m.get_cols() == 0) {
                handled = true; 
            }
        });

        if (!handled) {
            throw std::runtime_error("Lỗi: Hệ thống không trả về Matrix rỗng khi thiếu file!");
        }
    }
};

class RawCSVLoaderTest : public TestCase {
public:
    RawCSVLoaderTest() : TestCase("DataProcessingTest", "test_raw_csv_loader") {}
    void run_logic() override {
        // Create a dummy CSV for testing
        std::string temp_csv = "temp_test.csv";
        std::ofstream out(temp_csv);
        out << "Header\n0.0, 1.1\n50.0, 2.2\n100.0, 3.3\n";
        out.close();

        DataLoader loader;
        Matrix raw_data = Logger::profile("Nạp dữ liệu thô từ CSV", [&]() {
            return loader.loadRawCSV(temp_csv);
        });

        std::remove(temp_csv.c_str());

        OOP_ASSERT_EQ(raw_data.get_rows(), 3);
        OOP_ASSERT_EQ(raw_data.get_cols(), 2);
        OOP_ASSERT_NEAR(raw_data(0,0), 0.0, 1e-5);
        OOP_ASSERT_NEAR(raw_data(1,0), 50.0, 1e-5);
        OOP_ASSERT_NEAR(raw_data(2,0), 100.0, 1e-5);
        OOP_ASSERT_NEAR(raw_data(2,1), 3.3, 1e-5);
    }
};
