# RiskGuard ML Framework (v2.0)

**RiskGuard ML Framework** là một thư viện học máy (Machine Learning) toán học thuần túy, tối ưu hóa bằng **C++20**, được thiết kế chuyên biệt cho bài toán đánh giá rủi ro tín dụng (Credit Risk Assessment). Hệ thống được xây dựng theo nguyên tắc **100% offline và không phụ thuộc vào bất kỳ thư viện ngoài nào** (Zero-dependency), giúp đảm bảo tốc độ tính toán tối đa và tính bảo mật tuyệt đối cho dữ liệu tài chính.

---

## Mục lục
1. [Công nghệ sử dụng (Tech Stack)](#công-nghệ-sử-dụng-tech-stack)
2. [Cấu trúc thư mục](#cấu-trúc-thư-mục)
3. [Hướng dẫn cài đặt (Installation)](#hướng-dẫn-cài-đặt-installation)
4. [Hướng dẫn sử dụng (Usage)](#hướng-dẫn-sử-dụng-usage)
5. [Quy tắc quản lý Kho lưu trữ (Repository Rules)](#quy-tắc-quản-lý-kho-lưu-trữ-repository-rules)
6. [Quy trình Đóng góp & Duyệt Code (Contributing)](#quy-trình-đóng-góp--duyệt-code-contributing)
7. [Thành viên & Vai trò (Team Members)](#thành-viên--vai-trò-team-members)
8. [Giấy phép (License)](#giấy-phép-license)

---

## Công nghệ sử dụng (Tech Stack)

* **Ngôn ngữ lập trình**: C++20 Core.
* **Tính năng C++20 áp dụng**:
  * `std::span<const double>`: Giao tiếp dữ liệu zero-copy giữa Data Pipeline và Core Matrix.
  * `Concepts`: Ràng buộc kiểu dữ liệu ma trận (`MatrixLike`) tường minh ngay tại thời điểm biên dịch thay vì dùng SFINAE.
  * `std::format`: Định dạng chuỗi nhật ký hệ thống an toàn và tối ưu hiệu năng.
  * `[[nodiscard]]`: Ràng buộc kiểm tra kết quả tính toán trên các toán tử tuyến tính.
* **Hệ thống biên dịch**: CMake (Yêu cầu tối thiểu bản 3.20).
* **Môi trường phát triển tối ưu**: Clion / GCC 11+ / Clang 13+.

---

## Cấu trúc thư mục

```text
d:/github/ML RiskGuard/
├── CMakeLists.txt                      # Cấu hình C++20 tĩnh và các thực thi
├── README.md                           # Tài liệu hướng dẫn dự án
├── include/riskguard/
│   ├── core/                           
│   │   ├── Matrix.hpp                  # Flat storage lưu trữ tuyến tính phẳng
│   │   └── Layer.hpp                   # Lớp trừu tượng nền tảng với Concept constraint
│   ├── layers/                         # Các tầng của mạng Nơ-ron (Linear, ReLU, Sigmoid)
│   ├── loss/                           # Hàm mất mát (BCELoss, MSELoss)
│   └── utils/                          # Module đọc dữ liệu, chuẩn hóa và lưu trữ mô hình
├── src/                                # Source code thực thi chi tiết tương ứng
├── app/
│   └── main.cpp                        # Luồng Pipeline mẫu hoàn chỉnh
├── tests/
│   └── test_runner.cpp                 # Trình điều phối và kiểm thử tự động (No Third-party)
└── data/                               # Thư mục lưu trữ dữ liệu thô và trọng số mô hình
