#include "test_runner.hpp"
#include "logger.hpp"
#include "../include/riskguard/core/Matrix.hpp" // File của Hải (Trưởng nhóm)

using namespace riskguard;

// [OOP - Tính Kế thừa]: Kế thừa lớp TestCase chính
class MatrixMultiplicationTest : public TestCase {
public:
    MatrixMultiplicationTest() : TestCase("MatrixTest", "test_matrix_multiplication") {}
    void run_logic() override {
        Matrix A(2, 3);
        A(0,0)=1.0; A(0,1)=2.0; A(0,2)=3.0;
        A(1,0)=4.0; A(1,1)=5.0; A(1,2)=6.0;

        Matrix B(3, 2);
        B(0,0)=7.0; B(0,1)=8.0;
        B(1,0)=9.0; B(1,1)=1.0;
        B(2,0)=2.0; B(2,1)=3.0;

        Matrix C = Logger::profile("Phép nhân Ma trận A(2x3) * B(3x2)", [&]() { return A.multiply(B); });

        OOP_ASSERT_EQ(C.get_rows(), 2);
        OOP_ASSERT_EQ(C.get_cols(), 2);
        OOP_ASSERT_NEAR(C(0,0), 31.0, 1e-5);
        OOP_ASSERT_NEAR(C(1,1), 55.0, 1e-5);
    }
};

class MatrixBroadcastingTest : public TestCase {
public:
    MatrixBroadcastingTest() : TestCase("MatrixTest", "test_matrix_broadcasting") {}
    void run_logic() override {
        Matrix A(2, 2);
        A(0,0)=1.0; A(0,1)=2.0;
        A(1,0)=3.0; A(1,1)=4.0;

        Matrix Bias(1, 2);
        Bias(0,0)=10.0; Bias(0,1)=20.0;

        Matrix result = Logger::profile("Phép toán Cộng Broadcasting Bias", [&]() {
            Matrix res = A; 
            res.addBias(Bias); 
            return res; 
        });

        OOP_ASSERT_EQ(result.get_rows(), 2);
        OOP_ASSERT_NEAR(result(0,0), 11.0, 1e-5);
        OOP_ASSERT_NEAR(result(1,1), 24.0, 1e-5);
    }
};

class MatrixTransposeTest : public TestCase {
public:
    MatrixTransposeTest() : TestCase("MatrixTest", "test_matrix_transpose") {}
    void run_logic() override {
        Matrix X(3, 1);
        X(0,0)=1.0; X(1,0)=2.0; X(2,0)=3.0;

        Matrix XT = Logger::profile("Phép toán Chuyển vị Ma trận X(3x1)", [&]() { return X.transpose(); });

        OOP_ASSERT_EQ(XT.get_rows(), 1);
        OOP_ASSERT_EQ(XT.get_cols(), 3);
        OOP_ASSERT_NEAR(XT(0,0), 1.0, 1e-5);
        OOP_ASSERT_NEAR(XT(0,2), 3.0, 1e-5);
    }
};
