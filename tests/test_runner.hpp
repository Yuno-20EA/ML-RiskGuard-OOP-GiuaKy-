#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cmath>
#include <stdexcept>

// =========================================================================
// [OOP - Tính Trừu tượng]: Lớp cơ sở trừu tượng cho mọi Test Case
// =========================================================================
class TestCase {
protected:
    std::string suite_name;
    std::string test_name;

public:
    TestCase(std::string suite, std::string name) 
        : suite_name(suite), test_name(name) {}
    
    virtual ~TestCase() = default;

    // Hàm thuần ảo: Bắt buộc các module test phụ phải ghi đè (override)
    virtual void run_logic() = 0; 

    // Getter ứng dụng tính Đóng gói (Encapsulation)
    std::string get_full_name() const { 
        return suite_name + "." + test_name; 
    }
};

// =========================================================================
// [OOP - Đóng gói & Quản lý]: Bộ điều phối Test Runner trung tâm
// =========================================================================
class TestRunner {
private:
    // Quản lý đa hình bằng Smart Pointer để quản lý bộ nhớ tự động
    std::vector<std::unique_ptr<TestCase>> test_registry;
    int passed_count = 0;
    int failed_count = 0;

    TestRunner() = default; // Thiết kế dạng Singleton Pattern

public:
    static TestRunner& get_instance() {
        static TestRunner instance;
        return instance;
    }

    // Đăng ký một đối tượng Test Case phụ vào hệ thống chính
    void register_test(std::unique_ptr<TestCase> test) {
        test_registry.push_back(std::move(test));
    }

    // Duyệt và kích hoạt toàn bộ các test case phụ
    int run_all() {
        std::cout << "==================================================\n";
        std::cout << "       STARTING AUTOMATED OOP TEST RUNNER         \n";
        std::cout << "==================================================\n";

        for (const auto& test : test_registry) {
            std::cout << "[ RUN      ] " << test->get_full_name() << "\n";
            try {
                // [OOP - Tính Đa hình]: Gọi hàm xử lý thực tế của lớp con tại runtime
                test->run_logic(); 
                std::cout << "[       OK ] " << test->get_full_name() << "\n";
                passed_count++;
            } catch (const std::exception& e) {
                std::cerr << "[  FAILED  ] " << test->get_full_name() << "\n";
                std::cerr << "             Reason: " << e.what() << "\n";
                failed_count++;
            }
        }

        std::cout << "==================================================\n";
        std::cout << "TEST SUMMARY: Passed: " << passed_count << " | Failed: " << failed_count << "\n";
        std::cout << "==================================================\n";

        return failed_count == 0 ? 0 : 1;
    }
};

// Các bộ macro Assert dùng chung để kiểm tra logic
#define OOP_ASSERT_EQ(actual, expected) \
    if ((actual) != (expected)) { \
        throw std::runtime_error("Line " + std::to_string(__LINE__) + ": Expected [" + std::to_string(expected) + "], Got [" + std::to_string(actual) + "]"); \
    }

#define OOP_ASSERT_NEAR(actual, expected, abs_error) \
    if (std::abs((actual) - (expected)) > (abs_error)) { \
        throw std::runtime_error("Line " + std::to_string(__LINE__) + ": Expected near [" + std::to_string(expected) + "], Got [" + std::to_string(actual) + "]"); \
    }
