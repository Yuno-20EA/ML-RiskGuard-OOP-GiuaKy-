#include "test_runner.hpp"
#include "logger.hpp"

// Bao gồm các file test phụ
#include "../tests/test_matrix.cpp"
#include "../tests/test_layers.cpp"
#include "../tests/test_csv_reader.cpp"

int main() {
    Logger::log_info("Hệ thống kiểm thử tự động bắt đầu khởi tạo...");

    // Lấy instance duy nhất của bộ xương chính TestRunner
    TestRunner& runner = TestRunner::get_instance();

    // 1. Đăng ký các test case phụ thuộc nhóm Ma Trận
    runner.register_test(std::make_unique<MatrixMultiplicationTest>());
    runner.register_test(std::make_unique<MatrixBroadcastingTest>());
    runner.register_test(std::make_unique<MatrixTransposeTest>());

    // 2. Đăng ký các test case phụ thuộc nhóm Logic AI
    runner.register_test(std::make_unique<SigmoidActivationTest>());
    runner.register_test(std::make_unique<LinearForwardTest>());
    runner.register_test(std::make_unique<BCELossGradientTest>());

    // 3. Đăng ký các test case phụ thuộc nhóm Xử lý dữ liệu
    runner.register_test(std::make_unique<FileNotFoundTest>());
    runner.register_test(std::make_unique<MinMaxScalerTest>());

    // KÍCH HOẠT TÍNH ĐA HÌNH: Chạy toàn bộ danh sách test đã đăng ký
    return runner.run_all();
}
