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

Hướng dẫn cài đặt (Installation)
Do hệ thống hoàn toàn độc lập không phụ thuộc thư viện ngoài, bạn chỉ cần cấu hình trình biên dịch hỗ trợ chuẩn C++20.
# 1. Clone repository về máy cục bộ
git clone [https://github.com/your-username/ML-RiskGuard.git](https://github.com/your-username/ML-RiskGuard.git)
cd ML-RiskGuard

# 2. Tạo thư mục build
mkdir build && cd build

# 3. Chạy CMake để cấu hình hệ thống
cmake ..

# 4. Biên dịch mã nguồn
make

# 5. Khởi chạy bộ kiểm thử để đảm bảo tính ổn định toán học
./tests/test_runner

Hướng dẫn sử dụng (Usage)
Dưới đây là ví dụ minh họa cách khởi tạo pipeline đọc dữ liệu từ file CSV, chuẩn hóa phân phối Z-score, đưa vào mạng nơ-ron huấn luyện và lưu trữ cấu trúc trọng số xuống đĩa cứng:#include "riskguard/NeuralNetwork.hpp"
#include "riskguard/utils/CSVReader.hpp"
#include "riskguard/utils/DataNormalizer.hpp"
#include "riskguard/utils/ModelManager.hpp"

int main() {
    // 1. Đọc dữ liệu từ tập tin CSV
    CSVReader reader("../data/raw/credit_dataset.csv", ',', true);
    Matrix X_raw = reader.read_features({0, 1, 2, 3}); // Đọc các cột thuộc tính mẫu
    Matrix y = reader.read_labels(4);                  // Cột nhãn phân lớp

    // 2. Chuẩn hóa dữ liệu về khoảng mong muốn (0-1)
    DataNormalizer normalizer;
    Matrix X_norm = normalizer.fit_transform(X_raw);

    // 3. Khởi tạo mạng Nơ-ron cấu hình tuyến tính
    NeuralNetwork net(0.01); // Learning rate = 0.01
    net.add_layer(std::make_unique<LinearLayer>(4, 8));
    net.add_layer(std::make_unique<ReLULayer>());
    net.add_layer(std::make_unique<LinearLayer>(8, 1));
    net.add_layer(std::make_unique<SigmoidLayer>());

    // 4. Huấn luyện mô hình
    net.train(X_norm, y, 1000); // Huấn luyện qua 1000 Epochs

    // 5. Lưu trữ trạng thái mô hình
    ModelManager::save(net, "../data/weights/credit_model.bin");
    
    return 0;
}

Quy tắc quản lý Kho lưu trữ (Repository Rules)
1. Chiến lược quản lý Nhánh (Branching Strategy)
Tuyệt đối không push trực tiếp lên nhánh main hoặc master. Nhánh này phản ánh trạng thái production ổn định, chỉ chứa mã nguồn đã kiểm thử và nghiệm thu thành công.

Nhánh develop là nhánh trung tâm tích hợp. Mọi tính năng mới phải được gộp vào đây trước khi phát hành phiên bản chính thức.

Áp dụng nghiêm ngặt cơ chế Feature Branching. Tên nhánh bắt buộc tuân theo tiền tố cấu trúc:

feature/ten-tính-năng (Ví dụ: feature/csv-reader)

bugfix/tên-lỗi (Ví dụ: bugfix/fix-matrix-overflow)

hotfix/lỗi-khẩn-cấp (Sửa đổi trực tiếp các lỗi nghiêm trọng trên hệ thống đang vận hành)

2. Quy chuẩn Commit (Commit Messages)
Thông điệp commit phải ngắn gọn, rõ nghĩa và áp dụng chuẩn Conventional Commits:

Plaintext
<loại>: <mô tả ngắn bằng tiếng Anh hoặc tiếng Việt không dấu>
feat: Thêm một tính năng mới (Ví dụ: feat: add memory span optimization to matrix)

fix: Sửa chữa một lỗi logic hoặc lỗi biên dịch (Ví dụ: fix: resolve nan gradient in bce loss)

docs: Cập nhật tài liệu hoặc file README (Ví dụ: docs: update deployment guidelines)

style: Định dạng lại mã nguồn (căn lề, dấu chấm phẩy) mà không thay đổi logic chạy của code.

Quy trình Đóng góp & Duyệt Code (Contributing)
Bước 1: Tạo Pull Request (PR)
Khi hoàn thành một tính năng, thực hiện đẩy nhánh tính năng lên máy chủ từ xa và khởi tạo Pull Request hướng về nhánh develop.

Điền đầy đủ thông tin mô tả sự thay đổi và đảm bảo code vượt qua bộ kiểm thử cục bộ.

Bước 2: Duyệt mã nguồn (Code Review)
Một PR chỉ đủ điều kiện gộp (merge) vào nhánh trung tâm khi có ít nhất 1 thành viên khác thẩm định cấu trúc, tối ưu toán học và bấm phê duyệt (Approve).

Bước 3: Giải quyết xung đột đột biến (Conflict Resolution)
Nếu xảy ra xung đột mã nguồn (Conflict), chủ nhân của PR phải tự kéo (git pull) mã nguồn mới nhất từ nhánh develop về máy cục bộ, giải quyết xung đột, chạy lại bộ kiểm thử ổn định trước khi yêu cầu review lại.

Bước 4: Quy chuẩn chất lượng Code
Tất cả mã nguồn C++ trước khi push lên hệ thống phải được định dạng tự động qua cấu hình chung của dự án trên IDE (Clion/VS Code).

Tuyệt đối không đẩy các file cấu hình cục bộ cá nhân (.idea/, .vscode/, build/) lên hệ thống. Đảm bảo cấu hình và cập nhật đúng file .gitignore.

Thành viên & Vai trò (Team Members)
Dự án được phát triển và phối hợp vận hành bởi các thành viên:

Nguyễn Hoàng Hải – Nhóm trưởng

Nhiệm vụ: Thiết lập kiến trúc hệ thống, cấu hình CMake (C++20), phát triển lõi tính toán ma trận, tiền xử lý và làm sạch dữ liệu (Python), phát triển thuật toán học máy lõi (lan truyền ngược) và xử lý ổn định số học (sửa lỗi tràn số).

Anh Minh / Minh – Thành viên

Nhiệm vụ: Thiết kế và dựng sẵn hệ thống khung lớp (Class templates) cho mạng nơ-ron; phát triển module lưu trữ cấu trúc bộ não của máy (Model Persistence) ra tập tin mã hóa để tái sử dụng.

Trang – Thành viên

Nhiệm vụ: Viết module đọc dữ liệu từ tệp nguồn CSV, tự động thực hiện chuẩn hóa dữ liệu số lớn về khoảng tối ưu [0-1]; thiết kế màn hình hiển thị menu chữ điều khiển và hệ thống cảnh báo màu trực quan (Xanh/Đỏ).

Hiếu – Thành viên

Nhiệm vụ: Phát triển bộ kiểm thử tự động (Test Runner) độc lập bảo vệ logic hệ thống; thực hiện mã hóa công cụ đo tốc độ chạy, hiển thị thời gian tính toán chi tiết và quản lý nhật ký lịch sử hoạt động hệ thống.

Cả nhóm phối hợp – Integration & Deployment Team

Nhiệm vụ: Đồng bộ ráp nối tất cả các module thành phần, chạy thử nghiệm toàn diện trên 02 hồ sơ dữ liệu thực tế nhằm nghiệm thu dự án.

Nguồn dữ liệu : DATASET được lấy từ Kaggle.com
