from pathlib import Path
import json
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ===============================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ===============================================================
BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "models" / "xgboost_model.json"
FEATURES_PATH = BASE_DIR / "models" / "features.json"
CSV_PATH = BASE_DIR / "data" / "realtime.csv" # File 10 dòng

# ===============================================================
# 2. LOAD MODEL VÀ FEATURES
# ===============================================================
if not MODEL_PATH.exists():
    raise RuntimeError(f"Không tìm thấy model tại: {MODEL_PATH}")
if not FEATURES_PATH.exists():
    raise RuntimeError(f"Không tìm thấy file features tại: {FEATURES_PATH}")
if not CSV_PATH.exists():
    raise RuntimeError(f"Không tìm thấy file data tại: {CSV_PATH}")

booster = xgb.Booster()
booster.load_model(str(MODEL_PATH))

with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    FEATURES: list[str] = json.load(f)

logger = logging.getLogger("uvicorn")
logger.info(f"✅ Đã load model từ {MODEL_PATH}")
logger.info(f"✅ Features ({len(FEATURES)}): {FEATURES}")

# Mapping mã yfinance cho 4 thị trường
TICKERS_MAP = {
    "Nikkei225": "^N225",
    "DAX": "^GDAXI",
    "Nasdaq_Futures": "NQ=F",
    "Hangseng": "^HSI"
}

# ===============================================================
# 3. SCHEMA CHO RESPONSE
# ===============================================================
class PredictResponse(BaseModel):
    probability_up: float = Field(..., description="Xác suất giá TĂNG (class 1)")
    probability_down: float = Field(..., description="Xác suất giá GIẢM (class 0)")
    signal: str = Field(..., description="Tín hiệu: UP, DOWN, hoặc HOLD (đứng ngoài)")

app = FastAPI(title="Stock AI Service")

# ===============================================================
# 4. HÀM LẤY DỮ LIỆU VÀ TÍNH FEATURES (ĐÃ SỬA)
# ===============================================================
def get_latest_features() -> dict:
    # 1. Cấu hình tải dữ liệu 7 ngày gần nhất
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    logger.info(f"Đang tải dữ liệu từ {start_date} đến {end_date}...")

    # 2. Tải riêng từng thị trường để dễ áp công thức
    try:
        # Cần lấy thêm 1 ngày để bù vào chỗ shift(-1) bị mất dòng cuối
        nikkei_raw = yf.download("^N225", start=start_date, end=end_date, multi_level_index=False, progress=False)
        hangseng_raw = yf.download("^HSI", start=start_date, end=end_date, multi_level_index=False, progress=False)
        dax_raw = yf.download("^GDAXI", start=start_date, end=end_date, multi_level_index=False, progress=False)
        nasdaq_raw = yf.download("NQ=F", start=start_date, end=end_date, multi_level_index=False, progress=False)
        
        # Kiểm tra xem có dữ liệu không
        if any(df.empty for df in [nikkei_raw, hangseng_raw, dax_raw, nasdaq_raw]):
            raise ValueError("Một trong các thị trường không có dữ liệu trả về từ yfinance.")

    except Exception as e:
        logger.error(f"Lỗi khi tải dữ liệu từ yfinance: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi tải dữ liệu thị trường: {str(e)}")

    # 3. Áp công thức chính xác theo yêu cầu của bạn
    
    # Nikkei: (Close T+1 - Open T+1) / Open T+1
    nikkei_raw["Nikkei225"] = (nikkei_raw['Close'].shift(-1) - nikkei_raw['Open'].shift(-1)) / nikkei_raw['Open'].shift(-1)
    
    # Hangseng: (Close T+1 - Open T+1) / Open T+1
    hangseng_raw["Hangseng"] = (hangseng_raw['Close'].shift(-1) - hangseng_raw['Open'].shift(-1)) / hangseng_raw['Open'].shift(-1)
    
    # DAX: (Open T+1 - Close T) / Close T
    dax_raw["DAX"] = (dax_raw['Open'].shift(-1) - dax_raw['Close']) / dax_raw['Close']
    
    # Nasdaq Futures: (Open T+1 - Close T) / Close T
    nasdaq_raw["Nasdaq_Futures"] = (nasdaq_raw['Open'].shift(-1) - nasdaq_raw['Close']) / nasdaq_raw['Close']

    # 4. Lấy dòng cuối cùng có chứa dữ liệu (bỏ qua dòng NaN do shift)
    # Dùng .dropna() để loại bỏ dòng cuối cùng bị NaN, sau đó lấy iloc[-1]
    try:
        latest_nikkei = nikkei_raw["Nikkei225"].dropna().iloc[-1]
        latest_hangseng = hangseng_raw["Hangseng"].dropna().iloc[-1]
        latest_dax = dax_raw["DAX"].dropna().iloc[-1]
        latest_nasdaq = nasdaq_raw["Nasdaq_Futures"].dropna().iloc[-1]
    except IndexError:
        raise HTTPException(status_code=500, detail="Dữ liệu trả về không đủ để tính toán (bị rỗng sau khi dropna).")

    # 5. Gom lại thành dict
    features_dict = {
        "Nikkei225": float(latest_nikkei),
        "DAX": float(latest_dax),
        "Nasdaq_Futures": float(latest_nasdaq),
        "Hangseng": float(latest_hangseng)
    }
    
    logger.info(f"✅ Features mới nhất: {features_dict}")
    return features_dict

# ===============================================================
# 5. ENDPOINT DỰ ĐOÁN
# ===============================================================
@app.get("/predict", response_model=PredictResponse)
def predict():
    try:
        # 1. Lấy data từ file realtime.csv và merge ngày hiện tại
        features_dict = get_latest_features()
        
        # 2. Sắp xếp đúng thứ tự
        row = [float(features_dict[name]) for name in FEATURES]
        
        # 3. Predict
        X = np.array([row], dtype=np.float32)
        dmat = xgb.DMatrix(X, feature_names=FEATURES)
        prob_up = float(booster.predict(dmat)[0])
        
        prob_down = 1.0 - prob_up
        
        # 4. Áp ngưỡng
        if prob_up >= 0.55:
            signal = "UP"
        elif prob_up <= 0.45:
            signal = "DOWN"
        else:
            signal = "HOLD"

        return PredictResponse(
            probability_up=round(prob_up, 4),
            probability_down=round(prob_down, 4),
            signal=signal
        )
    except Exception as e:
        logger.error(f"Lỗi dự đoán: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"service": "stock-ai", "docs": "/docs"}