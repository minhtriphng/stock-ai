import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# Seed cố định
np.random.seed(42)

# ==========================================
# 1. ĐỌC DỮ LIỆU FEATURES DÀNH RIÊNG CHO XGBOOST
# ==========================================
print("1. Đang tải dữ liệu từ file features_xgb.csv...")
df_xgb = pd.read_csv('data/new.csv')
xgb_features = [
    'Nikkei225',
    'DAX',
    'Nasdaq_Futures',
    'Hangseng',
    # 'Daily_Mean_Sentiment',
    # 'News_Count'
]


X = df_xgb[xgb_features]
y = df_xgb['Target_Class']

print(f"Tổng số mẫu dữ liệu: {len(df_xgb)}")
print(f"Số lượng Features sử dụng ({len(xgb_features)}): {xgb_features}")

# ==========================================
# 2. CHIA TẬP TRAIN / VAL / TEST THEO THỜI GIAN (70% - 15% - 15%)
# ==========================================
n = len(df_xgb)
train_end = int(n * 0.70)
val_end = train_end + int(n * 0.15)

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
ratio = sum(y_train == 0) / sum(y_train == 1)
print(f"\nPhân chia dữ liệu: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

# ==========================================
# 3. HUẤN LUYỆN XGBOOST CLASSIFIER
# ==========================================
print("\n2. Đang huấn luyện XGBoost Classifier (Tầng 1)...")
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    scale_pos_weight=ratio,
    n_estimators=100,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# ==========================================
# 4. LƯU MODEL 
# ==========================================
model.save_model("models/xgboost_model.json")
print("-> Đã xuất: xgboost_model.json")
# ==========================================
# 5. THIẾT LẬP NGƯỠNG TIN CẬY (TÙY CHỈNH TẠI ĐÂY)
# ==========================================
UP_THRESHOLD = 0.55    # Xác suất >= 0.55 mới kích hoạt lệnh MUA (Up)
DOWN_THRESHOLD = 0.45  # Xác suất <= 0.45 mới kích hoạt lệnh BÁN (Down)
                       # Khoảng giữa (0.45 < prob < 0.55) sẽ ĐỨNG NGOÀI

# ==========================================
# 6. LẤY XÁC SUẤT VÀ ÁP DỤNG BỘ LỌC NGƯỠNG
# ==========================================
# Lấy xác suất dự đoán cho lớp Up (1)
y_probs = model.predict_proba(X_test)[:, 1]

# Khởi tạo mảng dự đoán (-1: Đứng ngoài, 0: Down, 1: Up)
y_pred_action = np.full(len(y_probs), -1)

# Áp dụng điều kiện ngưỡng
y_pred_action[y_probs >= UP_THRESHOLD] = 1
y_pred_action[y_probs <= DOWN_THRESHOLD] = 0

# ==========================================
# 7. ĐÁNH GIÁ CHỈ SỐ BỘ LỌC TẦNG 1
# ==========================================
# Tạo mask cho các phiên được chấp nhận vào lệnh (khác -1)
traded_mask = (y_pred_action != -1)

total_sessions = len(y_test)
traded_sessions = np.sum(traded_mask)
skipped_sessions = total_sessions - traded_sessions

print("\n" + "=" * 65)
print(f"=== BÁO CÁO MÔ HÌNH XGBOOST CLASSIFIER (NGƯỠNG TIN CẬY) ===")
print("=" * 65)
print(f"Ngưỡng kích hoạt MUA (Up)   : >= {UP_THRESHOLD*100:.1f}%")
print(f"Ngưỡng kích hoạt BÁN (Down) : <= {DOWN_THRESHOLD*100:.1f}%")
print("-" * 65)
print(f"Tổng số ngày tập Test        : {total_sessions} phiên")
print(f"Số ngày KÍCH HOẠT VÀO LỆNH   : {traded_sessions} phiên ({traded_sessions/total_sessions*100:.1f}%)")
print(f"Số ngày ĐỨNG NGOÀI (Lọc nhiễu): {skipped_sessions} phiên ({skipped_sessions/total_sessions*100:.1f}%)")

# Lọc dữ liệu thực tế và dự đoán chỉ trên các phiên ĐƯỢC CẤP LỆNH
y_test_traded = y_test[traded_mask]
y_pred_traded = y_pred_action[traded_mask]

if traded_sessions > 0:
    print("\n" + "-" * 65)
    print("=== CLASSIFICATION REPORT (CÁC LỆNH ĐƯỢC CHẤP NHẬN VÀO) ===")
    print(classification_report(y_test_traded, y_pred_traded, target_names=['Gap Down (Bán)', 'Gap Up (Mua)']))

    cm = confusion_matrix(y_test_traded, y_pred_traded)
    print("-" * 65)
    print("=== CONFUSION MATRIX (MA TRẬN NHẦM LẪN) ===")
    print(f"Đoán Down Đúng (True Down) : {cm[0, 0]} phiên")
    print(f"Đoán Down Sai  (False Down): {cm[0, 1]} phiên (Thực tế là Up)")
    print(f"Đoán Up Sai    (False Up)  : {cm[1, 0]} phiên (Thực tế là Down)")
    print(f"Đoán Up Đúng   (True Up)   : {cm[1, 1]} phiên")
else:
    print("\n[CẢNH BÁO] Không có phiên nào vượt qua ngưỡng tin cậy!")

# ROC-AUC giữ nguyên trên toàn bộ tập Test để theo dõi năng lực phân tách tổng thể
auc = roc_auc_score(y_test, y_probs)
print(f"\nROC-AUC Score (Toàn bộ tập Test): {auc:.4f}")
print("=" * 65)