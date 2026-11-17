# Đề tài 11 – Đánh giá rủi ro tín dụng khách hàng 

## 1. Mục tiêu
Xây dựng pipeline học máy phân loại dữ liệu Statlog German Credit thành:
- `1 = Good`: nhóm khách hàng được gán nhãn Good trong dữ liệu.
- `2 = Bad`: nhóm khách hàng được gán nhãn Bad trong dữ liệu.

**Cảnh báo:** Đây là bài tập học thuật với bộ dữ liệu nhỏ/cũ; không sử dụng để đánh giá người vay hoặc ra quyết định tín dụng thực tế.

## 2. Cấu trúc thư mục
```text
de_tai_11_rui_ro_tin_dung/
├── main.py                 # EDA, tiền xử lý, train, CV, đánh giá
├── app.py                  # Giao diện demo Streamlit
├── german.data             # Dữ liệu Statlog German Credit
├── statlog+german+credit+data.zip
├── requirements.txt
├── README.md
├── AI_USAGE.md
├── outputs/
│   ├── best_credit_risk_model.joblib
│   ├── model_metadata.json
│   ├── model_comparison.csv
│   ├── hyperparameter_search_details.csv
│   ├── test_predictions_best_model.csv
│   ├── test_probabilities_best_model.csv
│   └── ... biểu đồ, báo cáo EDA và phân tích lỗi
└── docs/
    ├── CHECKLIST_BAO_CAO.md
    └── HUONG_DAN_BAO_VE.md
```

## 3. Cài đặt trên Windows
Mở Terminal/CMD tại thư mục dự án:
```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Chạy pipeline học máy
```bash
python main.py
```
Chương trình sẽ:
1. Đọc dữ liệu và kiểm tra định dạng.
2. EDA: thống kê dữ liệu, missing, duplicate, phân phối nhãn, histogram, boxplot, tương quan và biểu đồ theo nhóm phân loại.
3. Chia train/test theo tỷ lệ 80/20, stratify, random_state=42.
4. So sánh hai pipeline tiền xử lý: StandardScaler và MinMaxScaler.
5. Huấn luyện KNN, Gaussian Naive Bayes và Decision Tree.
6. Dùng Stratified 5-Fold GridSearchCV trên tập train, tối ưu Macro-F1.
7. So sánh với DummyClassifier baseline.
8. Đánh giá tập test, lưu confusion matrix, báo cáo phân loại, dự đoán sai và kết quả xác suất.
9. Lưu mô hình tốt nhất cùng metadata vào `outputs/`.

**Lưu ý:** Chạy `main.py` trước khi mở giao diện để tạo file mô hình.

## 5. Mở giao diện demo
```bash
streamlit run app.py
```
Trình duyệt sẽ mở giao diện nhập các thuộc tính và xem dự đoán minh họa. Nếu không tự mở, Terminal sẽ in địa chỉ local để truy cập.

## 6. Các tệp kết quả quan trọng
| Tệp | Ý nghĩa |
|---|---|
| `outputs/eda_summary.csv` | Tóm tắt cấu trúc và chất lượng dữ liệu |
| `outputs/eda_describe.csv` | Thống kê mô tả |
| `outputs/model_comparison.csv` | Bảng so sánh 6 tổ hợp pipeline/mô hình và baseline |
| `outputs/hyperparameter_search_details.csv` | Chi tiết GridSearchCV |
| `outputs/06_best_model_confusion_matrix.png` | Ma trận nhầm lẫn của mô hình chọn |
| `outputs/test_predictions_best_model.csv` | Nhãn thật, nhãn dự đoán và đúng/sai |
| `outputs/misclassified_cases.csv` | Các mẫu dự đoán sai |
| `outputs/test_probabilities_best_model.csv` | Xác suất dự đoán lớp Bad |
| `outputs/best_credit_risk_model.joblib` | Mô hình đã huấn luyện để dùng lại |
| `outputs/model_metadata.json` | Thông tin cấu hình và đặc trưng |

## 7. Cách diễn giải kết quả
Không kết luận mô hình tốt chỉ vì Accuracy. Cần đọc đồng thời:
- **Recall_Bad:** trong các mẫu Bad thật, mô hình phát hiện được bao nhiêu.
- **Precision_Bad:** trong các mẫu được dự đoán Bad, bao nhiêu mẫu thực sự Bad.
- **F1_Bad / Macro-F1:** cân bằng Precision và Recall; Macro-F1 xem xét cả hai lớp.
- **Balanced Accuracy:** trung bình Recall của hai lớp.
- **ROC-AUC lớp Bad:** khả năng xếp hạng mẫu Bad cao hơn mẫu Good.
- **Confusion Matrix:** chỉ rõ hai kiểu nhầm lẫn.

Kết quả hiện có trên lần chạy đã kiểm tra: mô hình được chọn theo CV Macro-F1 là Pipeline B (MinMaxScaler) + Decision Tree; Accuracy test 0.65, Recall_Bad 0.50, Macro-F1 test khoảng 0.601. Điều này cho thấy mô hình còn bỏ sót một phần mẫu Bad; không nên mô tả là mô hình có hiệu năng cao. Kết quả có thể thay đổi theo môi trường/thư viện.

## 8. Hạn chế cần nêu trong báo cáo
- Dữ liệu có 1.000 mẫu, không đại diện cho thị trường tín dụng hiện tại.
- Kết quả phụ thuộc cách chia dữ liệu và cấu hình.
- Chi phí sai phân loại 1/5 trong code chỉ là giả định minh họa, không phải chi phí thực tế.
- Chưa có kiểm định độc lập trên bộ dữ liệu ngoài.
- Chưa thực hiện đánh giá fairness/thiên lệch theo nhóm; không nên sử dụng để ra quyết định thực tế.
- Xác suất từ `predict_proba` chưa được hiệu chuẩn; không diễn giải như xác suất vỡ nợ thực tế.

## 9. Trước khi nộp
Đọc `docs/CHECKLIST_BAO_CAO.md` và `docs/HUONG_DAN_BAO_VE.md`. Nhóm phải tự chạy lại, xác nhận các kết quả, cập nhật AI_USAGE.md đúng thực tế, ghi nguồn dữ liệu và phân công/commit của từng thành viên.
