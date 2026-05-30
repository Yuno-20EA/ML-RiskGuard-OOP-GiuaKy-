#include "../include/test_runner.hpp"
#include "../include/logger.hpp"
#include "../include/riskguard/layers/LinearLayer.hpp"
#include "../include/riskguard/layers/SigmoidLayer.hpp"
#include "../include/riskguard/NeuralNetwork.hpp"
#include <cmath>
#include <stdexcept>

using namespace riskguard;

class SigmoidActivationTest : public TestCase {
public:
    SigmoidActivationTest() : TestCase("AILogicTest", "test_sigmoid_activation") {}
    void run_logic() override {
        Matrix input(3, 1);
        input(0,0) = 0.0; input(1,0) = 100.0; input(2,0) = -100.0;
        
        SigmoidLayer layer;
        Matrix output = Logger::profile("Hàm kích hoạt Sigmoid (Clamping)", [&]() {
            return layer.forward(input);
        });

        OOP_ASSERT_EQ(output.get_rows(), 3);
        OOP_ASSERT_EQ(output.get_cols(), 1);
        OOP_ASSERT_NEAR(output(0,0), 0.5, 1e-5);
        OOP_ASSERT_NEAR(output(1,0), 1.0, 1e-5); 
        OOP_ASSERT_NEAR(output(2,0), 0.0, 1e-5);
    }
};

class LinearForwardTest : public TestCase {
public:
    LinearForwardTest() : TestCase("AILogicTest", "test_linear_forward") {}
    void run_logic() override {
        Matrix input_X(1, 2);
        input_X(0,0) = 2.0; input_X(0,1) = 3.0;
        
        LinearLayer layer(2, 1);
        auto params = layer.get_parameters();
        if (params.size() >= 2) {
            params[0]->operator()(0,0) = 4.0;
            params[0]->operator()(1,0) = 5.0;
            params[1]->operator()(0,0) = 1.0;
        }

        Matrix output_Y = Logger::profile("Linear Forward Pass (X*W + B)", [&]() {
            return layer.forward(input_X);
        });

        OOP_ASSERT_EQ(output_Y.get_rows(), 1);
        OOP_ASSERT_EQ(output_Y.get_cols(), 1);
        OOP_ASSERT_NEAR(output_Y(0,0), 24.0, 1e-5);
    }
};

class BCELossGradientTest : public TestCase {
public:
    BCELossGradientTest() : TestCase("AILogicTest", "test_bce_loss_gradient") {}
    void run_logic() override {
        Matrix pred(1, 1); pred(0,0) = 0.999999; 
        Matrix target(1, 1); target(0,0) = 1.0;

        NeuralNetwork net;
        Matrix gradient = Logger::profile("Tính Đạo hàm BCE Loss chống lỗi NaN", [&]() {
            return net.calculateBCEGradient(pred, target);
        });

        if (std::isnan(gradient(0,0)) || std::isinf(gradient(0,0))) {
            throw std::runtime_error("Lỗi nghiêm trọng: Đạo hàm trả về giá trị không hợp lệ (NaN/Infinity)!");
        }
    }
};
