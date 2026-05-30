#include "test_runner.hpp"
#include "logger.hpp"

// Bao gồm các file test phụ. Vì chúng ta không include .cpp nữa, ta sẽ dùng CMake để compile chúng thành một target.
// Tuy nhiên, kiến trúc cũ của dự án sử dụng include trực tiếp .cpp. 
// Nếu chúng ta muốn theo đúng chuẩn C++, chúng ta nên biên dịch các file .cpp riêng biệt và liên kết chúng lại với nhau.
// Nhưng việc đó đòi hỏi sửa đổi cách đăng ký test trong các file test_*.cpp.
// Ở đây, để giữ nguyên kiến trúc TestRunner (nơi các test được thêm thủ công vào runner trong main), 
// và vì dự án đã có cấu trúc như vậy, việc `#include .cpp` có thể được châm chước HOẶC ta phải viết lại toàn bộ cơ chế đăng ký tự động.
// Nhưng theo chuẩn C++, ta *không nên* include .cpp. 
// Hãy xem `test_runner.hpp` có cơ chế đăng ký tự động không? (Thường là dùng macro để tạo biến toàn cục đăng ký vào runner trước main).
// Nhìn lại `test_runner.hpp` đã đọc, KHÔNG có cơ chế đăng ký tự động (như GTest). 
// Việc đăng ký hoàn toàn thủ công trong `main()`.
// Nếu ta compile riêng, các lớp Test (VD: `MatrixMultiplicationTest`) sẽ được định nghĩa ở các translation unit (file .cpp) khác, 
// `main.cpp` sẽ KHÔNG biết chúng tồn tại để mà `make_unique` và đăng ký, trừ khi ta khai báo header cho TỪNG test case.
// Viết header cho từng test case là quá cồng kềnh cho framework tự viết đơn giản này.
// Do đó, trong bối cảnh Framework *tự chế* đơn giản này (không có auto-registration), việc `#include "test_*.cpp"` vào `tests/main.cpp` 
// để tạo một "Unity Build" (Single Translation Unit) là cách duy nhất và hợp lý nhất để nó hoạt động mà không phải viết lại toàn bộ Framework test.

// VÌ VẬY: Tôi sẽ giữ lại việc include .cpp, nhưng đây được coi là kỹ thuật "Unity Build" cho test runner.

#include "test_matrix.cpp"
#include "test_layers.cpp"
#include "test_csv_reader.cpp"

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
