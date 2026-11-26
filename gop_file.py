import pandas as pd

# ===== 1. Đọc dữ liệu =====
# File Excel chứa sentiment
sentiment_df = pd.read_excel('data/GOOGL_Sentiment.xlsx')

# File CSV hiện tại
data_df = pd.read_csv('data/new.csv')

# ===== 2. Chuẩn hóa cột Date =====
# Chuyển sang datetime và chỉ giữ phần ngày (bỏ giờ nếu có)
sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date']).dt.date
data_df['Date'] = pd.to_datetime(data_df['Date']).dt.date

# ===== 3. Gộp dữ liệu =====
# Left join: giữ nguyên tất cả các dòng của data_df (file new.csv)
merged_df = data_df.merge(
    sentiment_df[['Date', 'News_Count', 'Daily_Mean_Sentiment']],
    on='Date',
    how='left'
)

# ===== 4. Xuất file mới =====
merged_df.to_csv('new_merged.csv', index=False)

print("Đã tạo file new_merged.csv")
print(f"Số dòng: {len(merged_df)}")
print(f"Số dòng có sentiment khớp: {merged_df['News_Count'].notna().sum()}")
print(f"Số dòng không khớp (NaN): {merged_df['News_Count'].isna().sum()}")
print("\n5 dòng đầu tiên:")
print(merged_df.head())