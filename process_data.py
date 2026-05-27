import pandas as pd
import os

def process_data():
    # 1. Đọc file dữ liệu Kaggle thô
    raw_file = 'credit_risk_dataset_raw.csv'
    df = pd.read_csv(raw_file)

    # 2. Xóa bỏ các hàng có dữ liệu trống
    df = df.dropna()

    # 3. Lọc bỏ các dòng có tuổi vô lý (giả sử tuổi hợp lý là <= 200)
    df = df[df['person_age'] <= 200]

    # 4. Ánh xạ 5 cột cốt lõi
    # person_income -> Income
    # loan_amnt -> Debt
    # cb_person_default_on_file -> Delinquency (Lịch sử nợ xấu: Y/N)
    # person_age -> Age
    # loan_status -> Default (Tình trạng vỡ nợ: 0/1)
    columns_mapping = {
        'person_income': 'Income',
        'loan_amnt': 'Debt',
        'cb_person_default_on_file': 'Delinquency',
        'person_age': 'Age',
        'loan_status': 'Default'
    }
    
    # Giữ lại các cột cần thiết và đổi tên
    df = df[list(columns_mapping.keys())].rename(columns=columns_mapping)

    # 5. Chuyển đổi các cột chữ sang số thực
    # Đảm bảo xử lý đúng các giá trị 'Y'/'N' (loại bỏ khoảng trắng nếu có)
    if df['Delinquency'].dtype == 'object' or pd.api.types.is_string_dtype(df['Delinquency']):
        df['Delinquency'] = df['Delinquency'].astype(str).str.strip().str.upper()
        df['Delinquency'] = df['Delinquency'].map({'Y': 1.0, 'N': 0.0})
    
    # Xóa tiếp các hàng NA (do quá trình map sinh ra nếu có giá trị lạ)
    df = df.dropna()

    # Ép kiểu toàn bộ data frame sang số thực (float)
    df = df.astype(float)

    # 6. Xuất ra file data/dataset.csv sạch
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'dataset.csv')
    df.to_csv(output_file, index=False)
    print(f"Đã xử lý và lưu dữ liệu thành công tại: {output_file}")
    print(f"Số lượng bản ghi sau khi làm sạch: {len(df)}")

if __name__ == '__main__':
    process_data()
