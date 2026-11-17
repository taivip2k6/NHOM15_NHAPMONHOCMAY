# Gợi ý ôn bảo vệ Đề tài 11

1. **Bài toán là gì?** Phân loại dữ liệu German Credit thành hai nhãn Good (1) và Bad (2).
2. **Vì sao phải EDA?** Để hiểu cấu trúc, phân phối nhãn, giá trị thiếu, trùng lặp, ngoại lệ và quan hệ giữa thuộc tính.
3. **Vì sao chia train/test có stratify?** Giữ tỷ lệ lớp tương đối ổn định giữa hai tập.
4. **Data leakage được hạn chế thế nào?** Imputer, scaler và encoder được đặt trong Pipeline; GridSearchCV fit chúng bên trong từng fold chỉ từ phần train tương ứng.
5. **Vì sao có hai pipeline?** Để kiểm tra tác động của cách chuẩn hóa số: StandardScaler và MinMaxScaler.
6. **Vì sao dùng Macro-F1?** Vì bài toán có hai lớp và cần đánh giá cân bằng thay vì chỉ nhìn Accuracy.
7. **Baseline có ý nghĩa gì?** Cung cấp mốc đơn giản để xác định mô hình học máy có cải thiện so với dự đoán lớp phổ biến hay không.
8. **Recall_Bad thấp có ý nghĩa gì?** Mô hình bỏ sót một số mẫu thuộc lớp Bad; cần phân tích confusion matrix và không chỉ dựa vào Accuracy.
9. **Vì sao không chọn mô hình theo test?** Để giữ tập test làm đánh giá cuối cùng, tránh tối ưu quá mức theo dữ liệu kiểm thử.
10. **Giới hạn lớn nhất?** Dataset nhỏ/cũ, kết quả phụ thuộc split; giả định cost là minh họa và mô hình không dùng cho quyết định tín dụng thực tế.

Hãy dùng kết quả thực tế vừa chạy trên máy của nhóm; không học thuộc các con số cũ nếu kết quả thay đổi.
