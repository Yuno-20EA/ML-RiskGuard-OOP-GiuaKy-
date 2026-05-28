#pragma once
#include "../include/core/matrix.hpp"

// Hàm Mất Mát (Loss Function): Chiếc "Thước đo" để đánh giá xem AI đoán sai tới mức nào
// Nhiệm vụ của AI là tự học sao cho giá trị Loss này càng gần 0 càng tốt
class LossFunction {
public:
    virtual ~LossFunction() = default;
    
    // Tính khoảng cách giữa Dự đoán (pred) và Thực tế (true)
    virtual double compute(const Matrix& pred, const Matrix& true_) = 0;
    
    // Tính đạo hàm: Để chỉ đường cho AI biết phải đi hướng nào thì cái Mất Mát mới giảm đi
    virtual Matrix gradient(const Matrix& pred, const Matrix& true_) = 0;
};

// Hàm MSE (Mean Squared Error): Phù hợp cho việc AI dự đoán các con số thực
// Ví dụ: Dự đoán Khoản tiền tối đa có thể cho vay (hồi quy)
class MSELoss : public LossFunction {
public:
    double compute(const Matrix& pred, const Matrix& true_) override;
    Matrix gradient(const Matrix& pred, const Matrix& true_) override;
};

// Hàm BCE (Binary Cross Entropy): CHUYÊN DỤNG cho bài toán phân loại Rủi ro nợ xấu
// Giúp ép AI cực kỳ cẩn thận khi kết luận một hồ sơ là Bùng Nợ (1) hay Trả Tốt (0)
class BCELoss : public LossFunction {
private:
    double epsilon = 1e-15; // Hệ số nhỏ xíu để hệ thống không bị crash do lỗi chia 0 hoặc log(0)
public:
    double compute(const Matrix& pred, const Matrix& true_) override;
    Matrix gradient(const Matrix& pred, const Matrix& true_) override;
};
