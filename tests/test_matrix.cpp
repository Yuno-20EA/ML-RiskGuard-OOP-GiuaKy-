#include "../include/test_runner.hpp"
#include "../include/logger.hpp"
#include "../include/matrix.hpp" // File của Hải (Trưởng nhóm)

// [OOP - Tính Kế thừa]: Kế thừa lớp TestCase chính
class MatrixMultiplicationTest : public TestCase {
public:
    MatrixMultiplicationTest() : TestCase("MatrixTest", "test_matrix_multiplication") {}
    void run_logic() override {
        Matrix A(2, 3, {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
        Matrix B(3, 2, {{7.0, 8.0}, {9.0, 1.0}, {2.0, 3.0}});

        Matrix C = Logger::profile("Phép nhân Ma trận A(2x3) * B(3x2)", [&]() { return A * B; });

        OOP_ASSERT_EQ(C.rows, 2);
        OOP_ASSERT_EQ(C.cols, 2);
        OOP_ASSERT_NEAR(C.data[0][0], 31.0, 1e-5);
        OOP_ASSERT_NEAR(C.data[1][1], 55.0, 1e-5);
    }
};

class MatrixBroadcastingTest : public TestCase {
public:
    MatrixBroadcastingTest() : TestCase("MatrixTest", "test_matrix_broadcasting") {}
    void run_logic() override {
        Matrix A(2, 2, {{1.0, 2.0}, {3.0, 4.0}});
        Matrix Bias(1, 2, {{10.0, 20.0}});

        Matrix result = Logger::profile("Phép toán Cộng Broadcasting Bias", [&]() { return A + Bias; });

        OOP_ASSERT_EQ(result.rows, 2);
        OOP_ASSERT_NEAR(result.data[0][0], 11.0, 1e-5);
        OOP_ASSERT_NEAR(result.data[1][1], 24.0, 1e-5);
    }
};

class MatrixTransposeTest : public TestCase {
public:
    MatrixTransposeTest() : TestCase("MatrixTest", "test_matrix_transpose") {}
    void run_logic() override {
        Matrix X(3, 1, {{1.0}, {2.0}, {3.0}});

        Matrix XT = Logger::profile("Phép toán Chuyển vị Ma trận X(3x1)", [&]() { return X.transpose(); });

        OOP_ASSERT_EQ(XT.rows, 1);
        OOP_ASSERT_EQ(XT.cols, 3);
        OOP_ASSERT_NEAR(XT.data[0][0], 1.0, 1e-5);
        OOP_ASSERT_NEAR(XT.data[0][2], 3.0, 1e-5);
    }
};
