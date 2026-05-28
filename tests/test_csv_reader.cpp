#include "../include/test_runner.hpp"
#include "../include/logger.hpp"
#include "../include/data_loader.hpp" // File của Thu Trang

class FileNotFoundTest : public TestCase {
public:
    FileNotFoundTest() : TestCase("DataProcessingTest", "test_file_not_found") {}
    void run_logic() override {
        std::string fake_filename = "no_exist.csv";
        DataLoader loader;
        bool is_exception_caught = false;

        Logger::profile("Kiểm tra xử lý ngoại lệ khi thiếu File CSV", [&]() {
            try {
                loader.read_csv(fake_filename);
            } catch (const std::runtime_error& e) {
                is_exception_caught = true; // Bắt ngoại lệ thành công, hệ thống bảo mật tốt
            }
        });

        if (!is_exception_caught) {
            throw std::runtime_error("Lỗi: Hệ thống không ném ra ngoại lệ chuẩn, chương trình có nguy cơ bị sập!");
        }
    }
};

class MinMaxScalerTest : public TestCase {
public:
    MinMaxScalerTest() : TestCase("DataProcessingTest", "test_min_max_scaler") {}
    void run_logic() override {
        std::vector<double> raw_data = {0.0, 50.0, 100.0};
        DataLoader loader;

        std::vector<double> scaled_data = Logger::profile("Chuẩn hóa dữ liệu Min-Max [0, 1]", [&]() {
            return loader.min_max_scale(raw_data);
        });

        OOP_ASSERT_EQ(scaled_data.size(), 3);
        OOP_ASSERT_NEAR(scaled_data[0], 0.0, 1e-5);
        OOP_ASSERT_NEAR(scaled_data[1], 0.5, 1e-5);
        OOP_ASSERT_NEAR(scaled_data[2], 1.0, 1e-5);
    }
};
