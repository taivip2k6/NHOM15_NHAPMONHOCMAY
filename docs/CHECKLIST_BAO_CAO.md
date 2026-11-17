# Checklist hoàn thiện báo cáo theo hướng dẫn FIT4013

## 1. Đặt vấn đề và phân tích yêu cầu
- [ ] Nêu bối cảnh đánh giá rủi ro tín dụng và mục tiêu phân loại.
- [ ] Nêu nguồn Statlog German Credit, 1.000 mẫu, 20 đặc trưng, nhãn Good/Bad.
- [ ] Chèn bảng cấu trúc, kiểu dữ liệu, phân phối nhãn, missing/duplicate.
- [ ] Chèn biểu đồ EDA và giải thích điều quan sát được.

## 2. Thiết kế giải pháp và quy trình học máy
- [ ] Vẽ pipeline từ dữ liệu thô đến dự đoán.
- [ ] Giải thích median imputation, most-frequent imputation, One-Hot Encoding.
- [ ] So sánh StandardScaler (Pipeline A) với MinMaxScaler (Pipeline B).
- [ ] Giải thích train/test 80/20, stratify, random_state=42.
- [ ] Giải thích vì sao preprocessing nằm trong sklearn Pipeline để hạn chế data leakage.
- [ ] Mô tả KNN, GaussianNB, Decision Tree và siêu tham số tìm kiếm.

## 3. Phân tích mã nguồn và chức năng
- [ ] Mô tả `main.py`, `app.py`, `outputs/`.
- [ ] Chỉ trích các đoạn code quan trọng; không chép toàn bộ source.
- [ ] Giải thích cách lưu và nạp mô hình.

## 4. Thử nghiệm và kết quả
- [ ] Trình bày baseline DummyClassifier.
- [ ] Chèn bảng `model_comparison.csv`.
- [ ] Chèn confusion matrix, biểu đồ metric và giải thích.
- [ ] So sánh cả hai pipeline và ba thuật toán.
- [ ] Nêu rõ mô hình chọn theo CV Macro-F1 trên train, không chọn theo test.

## 5. Thảo luận và so sánh
- [ ] Phân tích Precision_Bad, Recall_Bad, F1_Bad, Macro-F1, Balanced Accuracy và ROC-AUC.
- [ ] Giải thích hai loại lỗi Good→Bad và Bad→Good.
- [ ] Chọn một vài dòng trong `misclassified_cases.csv` để minh họa.
- [ ] Giải thích hạn chế dữ liệu và giả định chi phí sai phân loại.
- [ ] Không tuyên bố mô hình dùng được trong nghiệp vụ tín dụng thực tế.

## 6. Tổng kết và hướng phát triển
- [ ] Tóm tắt kết quả thực nghiệm thực tế sau khi nhóm chạy lại.
- [ ] Đề xuất thu thập dữ liệu mới, kiểm định ngoài mẫu, hiệu chuẩn xác suất và đánh giá fairness.
- [ ] Nêu giới hạn nghiên cứu.

## 7. Sản phẩm nộp và liêm chính
- [ ] PDF LaTeX theo template học phần.
- [ ] GitHub có README, source, requirements, hướng dẫn chạy.
- [ ] Có log/commit thể hiện đóng góp từng thành viên.
- [ ] Cập nhật AI_USAGE.md trung thực.
- [ ] Nhóm tự chạy lại và giải thích được mọi phần khi bảo vệ.
