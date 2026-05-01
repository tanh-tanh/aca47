# Aca47 - Phân tích bán hàng, tồn kho và dự báo nhu cầu

## Giới thiệu
Dự án này tập trung vào phân tích cho cuộc thi datathon dữ liệu bán hàng theo góc nhìn vận hành và kinh doanh, gồm:
- Theo dõi doanh thu, danh mục sản phẩm và tồn kho.
- Phân tích retention, tác động khuyến mãi và tín hiệu từ review/web traffic.
- Phân khúc khách hàng (RFM), giải thích mô hình (XAI) và dự báo cho năm 2024.

## Cấu trúc thư mục
- `raw/`: Dữ liệu nguồn dạng CSV và notebook baseline.
- `notebooks/`: Notebook EDA và forecasting.
- `OUTPUT/`: Kết quả đầu ra mô hình và biểu đồ tổng hợp.
- `dashboard_outputs/`: Ảnh dashboard đã export.

## Dữ liệu chính
Một số file dữ liệu quan trọng trong `raw/`:
- `orders.csv`, `order_items.csv`, `sales.csv`: giao dịch bán hàng.
- `products.csv`, `inventory.csv`: thông tin sản phẩm và tồn kho.
- `customers.csv`, `reviews.csv`, `web_traffic.csv`: hành vi khách hàng.
- `promotions.csv`, `returns.csv`, `shipments.csv`, `payments.csv`, `geography.csv`.

## Kết quả hiện có
- Dự báo: `notebooks/Forecasting.ipynb`.
- Dashboard EDA: `notebooks/[EDA] dashboard.ipynb`.
- Ảnh dashboard:
  - `dashboard_outputs/dashboard_1_revenue_inventory_category.png`
  - `dashboard_outputs/dashboard_2_retention_promo_xai.png`
  - `dashboard_outputs/dashboard_3_rfm_xai_forecast_2024.png`
- Kết quả mẫu và submission:
  - `OUTPUT/submission.csv`
  - `raw/sample_submission.csv`

## Hướng dẫn chạy end-to-end

### 1) Chuẩn bị môi trường
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install jupyter pandas numpy matplotlib seaborn plotly scikit-learn xgboost lightgbm
```

### 2) Kiểm tra dữ liệu đầu vào
Đảm bảo các file CSV nằm trong thư mục `raw/` (ví dụ: `orders.csv`, `sales.csv`, `products.csv`, `customers.csv`, ...).

## Cách sử dụng nhanh
1. Mở notebook EDA:
   - `notebooks/[EDA] dashboard.ipynb`
2. Chạy notebook dự báo:
   - `notebooks/Forecasting.ipynb`
3. Kiểm tra kết quả trong:
   - `OUTPUT/`
   - `dashboard_outputs/`

## Gợi ý môi trường
Khuyến nghị dùng Python 3.10+ với các thư viện phổ biến trong phân tích dữ liệu:
- `pandas`, `numpy`
- `matplotlib`, `seaborn`, `plotly`
- `scikit-learn`, `xgboost`, `lightgbm`
- `jupyter`
