# Lưu thư viện để sau này clone dự án

Lưu thư viện: pip freeze > requirements.txt

# Các bước chạy

## Bước 1: Cài python phù hợp phiên bản (nâng cấp)

sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv

## Bước 2: Tạo môi trường ảo

python3.12 -m venv .venv
source .venv/bin/activate

## Bước 3: Cài thư viện

pip install -r requirements.txt

## Bước 3.1: Chạy tạo features để lấy dữ liệu mới nhất

python src/features.py

## Bước 3.2: Chạy train model với dữ liệu mới tạo

python src/xgboost_train.py

## Bước 4: Chạy local

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
