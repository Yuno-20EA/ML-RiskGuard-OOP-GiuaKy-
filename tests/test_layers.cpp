#include "../include/test_runner.hpp"
#include "../include/logger.hpp"
#include "../include/layers.hpp" // File của Đức Minh
#include "../include/loss.hpp"   // File sửa lỗi chia cho 0 của Hải

class SigmoidActivationTest : public TestCase {
public:
    SigmoidActivationTest() : TestCase("AILogicTest", "test_sigmoid_activation") {}
    void run_logic() override {
        std::vector<double> input = {0.0, 100.0, -100.0};
        
        std::vector<double> output = Logger::profile("Hàm kích hoạt Sigmoid (Clamping)", [&]() {
            return Activation::sigmoid(input);
        });

        OOP_ASSERT_EQ(output.size(), 3);
        OOP_ASSERT_NEAR(output[0], 0.5, 1e-5);
        OOP_ASSERT_NEAR(output[1], 1.0, 1e-5); // Kiểm tra chống lỗi tràn số e^100
        OOP_ASSERT_NEAR(output[2], 0.0, 1e-5);
    }
};

class LinearForwardTest : public TestCase {
public:
    LinearForwardTest() : TestCase("AILogicTest", "test_linear_forward") {}
    void run_logic() override {
        std::vector<double> input_X = {2.0, 3.0};
        LinearLayer layer(2, 1);
        layer.set_weights({{4.0}, {5.0}});
        layer.set_bias({1.0});

        std::vector<double> output_Y = Logger::profile("Linear Forward Pass (X*W + B)", [&]() {
            return layer.forward(input_X);
        });

        OOP_ASSERT_EQ(output_Y.size(), 1);
        OOP_ASSERT_NEAR(output_Y[0], 24.0, 1e-5); // Đối chiếu kết quả tính tay
    }
};

class BCELossGradientTest : public TestCase {
public:
    BCELossGradientTest() : TestCase("AILogicTest", "test_bce_loss_gradient") {}
    void run_logic() override {
        double pred = 0.999999; // Giá trị sát biên 1
        double target = 1.0;

        double gradient = Logger::profile("Tính Đạo hàm BCE Loss chống lỗi NaN", [&]() {
            return BCELoss::backward(pred, target);
        });

        // Khóa chết lỗi chặn không cho hệ thống sinh ra số NaN (chia cho 0)
        if (std::isnan(gradient) || std::isinf(gradient)) {
            throw std::runtime_error("Lỗi nghiêm trọng: Đạo hàm trả về giá trị không hợp lệ (NaN/Infinity)!");
        }
        OOP_ASSERT_NEAR(gradient, -1.000001, 1e-4);
    }
};
