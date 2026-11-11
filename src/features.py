import xgboost as xgb
import pandas as pd
import numpy as np
import yfinance as yf

df = yf.download('GOOG', start='2006-01-01', multi_level_index=False)


print(df.head())
# Tải dữ liệu
nasdaq_futures_raw = yf.download("NQ=F", start='2006-01-01', multi_level_index=False) #Nasdaq_Futures
nikkei225_raw = yf.download("^N225", start='2006-01-01', multi_level_index=False)#Nikkei225
hangseng_raw = yf.download("^HSI", start='2006-01-01', multi_level_index=False)#Hangseng
dax_raw = yf.download("^GDAXI", start='2006-01-01', multi_level_index=False)#DAX



# Tạo features
nikkei225_raw["Nikkei225"]  = (nikkei225_raw['Close'].shift(-1) - nikkei225_raw['Open'].shift(-1)) / nikkei225_raw['Open'].shift(-1)
hangseng_raw["Hangseng"]  = (hangseng_raw['Close'].shift(-1) - hangseng_raw['Open'].shift(-1)) / hangseng_raw['Open'].shift(-1)
dax_raw["DAX"]  = (dax_raw['Open'].shift(-1) - dax_raw['Close']) / dax_raw['Close']
nasdaq_futures_raw['Nasdaq_Futures'] = (nasdaq_futures_raw['Open'].shift(-1) - nasdaq_futures_raw['Close']) / nasdaq_futures_raw['Close']

#Mapping
df = df.join(nikkei225_raw[['Nikkei225']], how='left')
df = df.join(hangseng_raw[['Hangseng']], how='left')
df = df.join(dax_raw[['DAX']], how='left')
df = df.join(nasdaq_futures_raw[['Nasdaq_Futures']], how='left')


#Fill dữ liệu
cols_ffill = ['Nikkei225', 'Hangseng', 'DAX', 'Nasdaq_Futures']
df[cols_ffill] = df[cols_ffill].ffill()


# ==========================================
# Tạo target
# ==========================================
# Target: Mức giá Open phiên tiếp theo (Open_{T+1})
df['Target_Class'] = ((df['Open'].shift(-1) - df['Close']) > 0).astype(int)

print(f"Tổng số dòng dữ liệu trước khi xóa: {len(df)}")
df = df.iloc[20:]

print(f"Tổng số dòng dữ liệu hoàn chỉnh dùng chung: {len(df)}")
df.to_csv('data/new.csv')
print("-> Đã lưu thành công file 'new.csv'!")