# -*- coding: utf-8 -*-
"""Dashboard Streamlit nâng cấp cho Đề tài 11 - Đánh giá rủi ro tín dụng.
   Chỉ phục vụ mục đích học tập, không dùng để ra quyết định tín dụng thực tế.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "outputs"
DATA_FILE = BASE_DIR / "german.data"
MODEL_FILE = OUT / "best_credit_risk_model.joblib"
META_FILE = OUT / "model_metadata.json"

COLUMNS = [
    "checking_account", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_account", "employment_since",
    "installment_rate", "personal_status_sex", "other_debtors",
    "residence_since", "property", "age_years",
    "other_installment_plans", "housing", "existing_credits", "job",
    "liable_people", "telephone", "foreign_worker", "risk"
]
LABELS = {
    "checking_account": "Tình trạng tài khoản thanh toán",
    "duration_months": "Thời hạn vay (tháng)",
    "credit_history": "Lịch sử tín dụng",
    "purpose": "Mục đích vay",
    "credit_amount": "Số tiền vay (DM)",
    "savings_account": "Tiền tiết kiệm / trái phiếu",
    "employment_since": "Thời gian làm việc hiện tại",
    "installment_rate": "Tỷ lệ trả góp (% thu nhập)",
    "personal_status_sex": "Tình trạng cá nhân",
    "other_debtors": "Người bảo lãnh / đồng nghĩa vụ",
    "residence_since": "Thời gian cư trú hiện tại (năm)",
    "property": "Tài sản sở hữu",
    "age_years": "Tuổi (năm)",
    "other_installment_plans": "Kế hoạch trả góp khác",
    "housing": "Tình trạng nhà ở",
    "existing_credits": "Số khoản tín dụng hiện có",
    "job": "Nghề nghiệp",
    "liable_people": "Số người phụ thuộc",
    "telephone": "Có điện thoại đăng ký",
    "foreign_worker": "Lao động nước ngoài",
}
DESCRIPTIONS = {
    "checking_account": "Mã trạng thái tài khoản theo bộ dữ liệu German Credit; xem chú giải mã trong bộ dữ liệu gốc.",
    "duration_months": "Số tháng dự kiến của khoản vay.",
    "credit_history": "Mã lịch sử tín dụng được cung cấp trong dữ liệu.",
    "purpose": "Mã mục đích sử dụng khoản vay.",
    "credit_amount": "Giá trị khoản vay trong đơn vị tiền tệ của bộ dữ liệu.",
    "savings_account": "Mã nhóm tiền tiết kiệm / trái phiếu.",
    "employment_since": "Mã thời gian làm việc tại nơi hiện tại.",
    "installment_rate": "Tỷ lệ khoản trả góp so với thu nhập khả dụng.",
    "personal_status_sex": "Mã nhóm tình trạng cá nhân trong dữ liệu.",
    "other_debtors": "Thông tin người bảo lãnh hoặc đồng nghĩa vụ.",
    "residence_since": "Số năm sống tại nơi cư trú hiện tại.",
    "property": "Mã loại tài sản.",
    "age_years": "Tuổi của người trong hồ sơ dữ liệu.",
    "other_installment_plans": "Thông tin các kế hoạch trả góp khác.",
    "housing": "Mã tình trạng nhà ở.",
    "existing_credits": "Số khoản tín dụng đang có tại ngân hàng.",
    "job": "Mã nhóm nghề nghiệp.",
    "liable_people": "Số người mà người vay có trách nhiệm chu cấp.",
    "telephone": "Tình trạng điện thoại theo mã dữ liệu.",
    "foreign_worker": "Thông tin lao động nước ngoài theo mã dữ liệu.",
}

# Từ điển Việt hóa tên mô hình, pipeline và cột kết quả khi hiển thị trên giao diện.
MODEL_NAMES = {
    "KNeighborsClassifier": "K- láng giềng gần nhất (KNN)",
    "GaussianNB": "Naive Bayes Gaussian",
    "DecisionTreeClassifier": "Cây quyết định (Decision Tree)",
    "MostFrequent": "Baseline - dự đoán lớp phổ biến nhất",
}
PIPELINE_NAMES = {
    "Pipeline_A_StandardScaler": "Luồng A - Chuẩn hóa StandardScaler",
    "Pipeline_B_MinMaxScaler": "Luồng B - Chuẩn hóa MinMaxScaler",
    "Baseline": "Mô hình cơ sở (Baseline)",
}
COLUMN_NAMES = {
    "Pipeline": "Quy trình tiền xử lý", "Model": "Mô hình", "Best_Params": "Tham số tốt nhất",
    "CV_Macro_F1_mean": "CV Macro-F1 Mean", "CV_Macro_F1_std": "CV Macro-F1 Std",
    "Accuracy_test": "Accuracy", "Precision_Bad": "Precision - nhóm rủi ro cao",
    "Recall_Bad": "Recall - nhóm rủi ro cao", "F1_Bad": "F1-score - nhóm rủi ro cao",
    "Macro_F1_test": "Macro-F1 (Test)", "Balanced_Accuracy_test": "Balanced Accuracy",
    "ROC_AUC_Bad": "ROC-AUC - nhóm rủi ro cao", "Cost_total": "Tổng chi phí lỗi (minh họa)",
    "actual_risk": "Nhãn thực tế", "predicted_risk": "Nhãn dự đoán",
    "correct": "Dự đoán đúng", "row_index": "Mã dòng dữ liệu",
}
def model_name_vi(value):
    value = str(value)
    return MODEL_NAMES.get(value, value.replace("KNeighborsClassifier", "KNN").replace("DecisionTreeClassifier", "Cây quyết định"))
def pipeline_name_vi(value):
    value = str(value)
    return PIPELINE_NAMES.get(value, value)
def translate_result_table(frame):
    view = frame.copy()
    if "Model" in view.columns:
        view["Model"] = view["Model"].map(model_name_vi)
    if "Pipeline" in view.columns:
        view["Pipeline"] = view["Pipeline"].map(pipeline_name_vi)
    return view.rename(columns=COLUMN_NAMES)


st.set_page_config(page_title="CreditRisk | Đề tài 11", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root { --ink:#17243a; --muted:#64748b; --line:#e5eaf2; --blue:#2457a7; }
.block-container {padding-top:1.6rem; padding-bottom:2.5rem; max-width:1440px;}
[data-testid="stSidebar"] {background:#f5f7fb; border-right:1px solid #e7ebf2;}
.hero {padding:1.35rem 1.55rem; border-radius:18px; background:linear-gradient(115deg,#102b50,#2457a7); color:white; margin-bottom:1rem;}
.hero h1 {font-size:2rem; margin:0 0 .35rem 0; color:white;}
.hero p {margin:0; color:#dce8fb; font-size:.98rem;}
.eyebrow {text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; font-weight:700; color:#b9d3fb; margin-bottom:.45rem;}
.section-note {color:#64748b; font-size:.9rem; margin-top:-.5rem;}
.panel {border:1px solid #e6ebf2; border-radius:14px; padding:1rem 1.1rem; background:white;}
.small-muted {font-size:.82rem; color:#64748b;}
div[data-testid="stMetric"] {background:white; border:1px solid #e6ebf2; padding:13px 16px; border-radius:13px;}
div.stButton>button, div.stFormSubmitButton>button {border-radius:10px; font-weight:650; min-height:2.7rem;}
[data-testid="stTabs"] button {font-weight:650;}
hr {border-color:#e8edf4;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

@st.cache_data
def load_reference():
    return pd.read_csv(DATA_FILE, sep=r"\s+", header=None, names=COLUMNS)

@st.cache_data
def read_csv_if_exists(name):
    path = OUT / name
    return pd.read_csv(path) if path.exists() else None

@st.cache_data
def read_text_if_exists(name):
    path = OUT / name
    return path.read_text(encoding="utf-8") if path.exists() else None

if not MODEL_FILE.exists() or not META_FILE.exists():
    st.error("Chưa có mô hình đã huấn luyện. Hãy mở Terminal tại thư mục dự án và chạy `python main.py` trước.")
    st.code("python main.py\nstreamlit run app.py", language="powershell")
    st.stop()
if not DATA_FILE.exists():
    st.error("Không tìm thấy german.data. Hãy đặt file dữ liệu cùng thư mục app.py.")
    st.stop()

model = load_model()
reference = load_reference()
meta = json.loads(META_FILE.read_text(encoding="utf-8"))
comparison = read_csv_if_exists("model_comparison.csv")
eda = read_csv_if_exists("eda_summary.csv")
error_summary = read_csv_if_exists("error_analysis_summary.csv")

# Sidebar navigation
st.sidebar.markdown("## 📊 CreditRisk")
st.sidebar.caption("ĐỀ TÀI 11 · HỌC PHẦN NHẬP MÔN HỌC MÁY")
page = st.sidebar.radio("ĐIỀU HƯỚNG", ["Tổng quan", "Dự đoán hồ sơ mẫu", "Khám phá dữ liệu", "Đánh giá mô hình", "Phân tích lỗi"], label_visibility="visible")
st.sidebar.divider()
st.sidebar.markdown("**Thông tin mô hình đang nạp**")
st.sidebar.write(f"- Thuật toán: **{model_name_vi(meta.get('best_model', '—'))}**")
st.sidebar.write(f"- Quy trình: **{pipeline_name_vi(meta.get('best_pipeline', '—'))}**")
st.sidebar.write(f"- Số đặc trưng: **{len(meta.get('feature_columns', []))}**")
st.sidebar.caption("Bộ dữ liệu German Credit · 1.000 hồ sơ")
st.sidebar.divider()
st.sidebar.warning("Chỉ phục vụ học tập. Không sử dụng để quyết định cấp hoặc từ chối tín dụng thực tế.")

st.markdown("""
<div class="hero">
  <div class="eyebrow">HỌC MÁY · ĐÁNH GIÁ RỦI RO TÍN DỤNG</div>
  <h1>Hệ thống đánh giá rủi ro tín dụng</h1>
  <p>Đề tài 11 — Khám phá dữ liệu, thử nghiệm mô hình và minh họa dự đoán trên bộ German Credit.</p>
</div>
""", unsafe_allow_html=True)

if page == "Tổng quan":
    st.subheader("Tổng quan dự án")
    st.markdown('<p class="section-note">Tóm tắt dữ liệu và kết quả của lần thực nghiệm gần nhất.</p>', unsafe_allow_html=True)
    summary_text = read_text_if_exists("result_summary.txt") or ""
    import re
    def metric_from_summary(key, default="—"):
        m = re.search(rf"^{re.escape(key)}:\s*([0-9.]+)", summary_text, re.M)
        return float(m.group(1)) if m else default
    total = len(reference)
    good = int((reference.risk == 1).sum())
    bad = int((reference.risk == 2).sum())
    a,b,c,d = st.columns(4)
    a.metric("Tổng số hồ sơ", f"{total:,}")
    b.metric("Nhóm Good", f"{good:,}", f"{good/total:.1%} dữ liệu")
    c.metric("Nhóm Bad", f"{bad:,}", f"{bad/total:.1%} dữ liệu")
    acc = metric_from_summary("Accuracy_test")
    d.metric("Accuracy trên tập kiểm thử", f"{acc:.1%}" if isinstance(acc,float) else acc)
    st.markdown("### Quy trình thực hiện")
    steps = ["Nguồn dữ liệu", "EDA", "Tiền xử lý", "Biểu diễn đặc trưng", "Huấn luyện", "Đánh giá", "Phân tích lỗi"]
    cols = st.columns(len(steps))
    for i,(col,label) in enumerate(zip(cols,steps)):
        with col:
            st.markdown(f'<div class="panel" style="text-align:center;min-height:92px"><div style="font-size:1.35rem;font-weight:800;color:#2457a7">{i+1:02d}</div><div style="font-size:.8rem;font-weight:650">{label}</div></div>', unsafe_allow_html=True)
    st.markdown("### Kết quả nổi bật")
    if comparison is not None:
        valid = comparison[comparison.Model != "MostFrequent"].copy()
        if not valid.empty:
            best = valid.sort_values("Macro_F1_test", ascending=False).iloc[0]
            x,y,z = st.columns(3)
            x.metric("Macro-F1 cao nhất trên tập kiểm thử", f"{best['Macro_F1_test']:.3f}")
            y.metric("Mô hình tương ứng", str(best["Model"]))
            z.metric("Pipeline", str(best["Pipeline"]).replace("Pipeline_", ""))
            st.caption("Lưu ý: mô hình dùng cho phần minh họa được chọn theo Macro-F1 trung bình qua kiểm định chéo (CV) trên tập huấn luyện; bảng trên trình bày kết quả trên tập kiểm thử.")
    st.info("Dùng thanh điều hướng bên trái để nhập hồ sơ mẫu, xem biểu đồ EDA, so sánh mô hình và phân tích các trường hợp dự đoán sai.")

elif page == "Dự đoán hồ sơ mẫu":
    st.subheader("Dự đoán hồ sơ mẫu")
    st.markdown('<p class="section-note">Nhập các thuộc tính theo lựa chọn có sẵn trong bộ dữ liệu, sau đó nhấn nút đánh giá.</p>', unsafe_allow_html=True)
    features = meta["feature_columns"]
    numeric_cols = meta["numeric_columns"]
    with st.form("credit_form_v2"):
        st.markdown("#### 1. Thông tin khoản vay")
        loan_cols = ["duration_months", "credit_amount", "purpose", "installment_rate", "other_installment_plans"]
        other_cols = [c for c in features if c not in loan_cols]
        values = {}
        left,right = st.columns(2)
        for idx,col in enumerate(loan_cols + other_cols):
            container = left if idx % 2 == 0 else right
            with container:
                label = LABELS.get(col, col)
                help_text = DESCRIPTIONS.get(col, "Giá trị theo mã của bộ dữ liệu German Credit.")
                if col in numeric_cols:
                    series = pd.to_numeric(reference[col], errors="coerce").dropna()
                    low, high = float(series.min()), float(series.max())
                    default = float(series.median())
                    if low.is_integer() and high.is_integer():
                        values[col] = st.number_input(label, min_value=int(low), max_value=int(high), value=int(round(default)), step=1, help=help_text)
                    else:
                        values[col] = st.number_input(label, min_value=low, max_value=high, value=default, help=help_text)
                else:
                    options = sorted(reference[col].dropna().astype(str).unique().tolist())
                    values[col] = st.selectbox(label, options, help=help_text, format_func=lambda x: f"Mã lựa chọn: {x}")
        st.markdown(" ")
        submitted = st.form_submit_button("🔎 Đánh giá hồ sơ mẫu", type="primary", use_container_width=True)
    if submitted:
        sample = pd.DataFrame([values], columns=features)
        prediction = int(model.predict(sample)[0])
        st.divider()
        st.markdown("### Kết quả dự đoán minh họa")
        if prediction == 2:
            st.error("**Mô hình dự đoán: Bad** · Hồ sơ thuộc nhóm rủi ro cao theo nhãn của bộ dữ liệu.")
        else:
            st.success("**Mô hình dự đoán: Good** · Hồ sơ thuộc nhóm rủi ro thấp theo nhãn của bộ dữ liệu.")
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(sample)[0]
            classes = list(model.classes_)
            if 2 in classes:
                p_bad = float(proba[classes.index(2)])
                p_good = float(proba[classes.index(1)]) if 1 in classes else 1-p_bad
                m1,m2 = st.columns(2)
                m1.metric("Điểm mô hình cho nhóm rủi ro thấp", f"{p_good:.1%}")
                m2.metric("Điểm mô hình cho nhóm rủi ro cao", f"{p_bad:.1%}")
                st.progress(p_bad, text="Điểm mô hình gán cho nhóm rủi ro cao")
                st.caption("Các điểm trên là đầu ra xác suất của mô hình, không phải xác suất vỡ nợ đã được hiệu chuẩn.")
        with st.expander("Xem lại thông tin vừa nhập"):
            display = pd.DataFrame([values]).rename(columns=LABELS).T.reset_index()
            display.columns = ["Thuộc tính", "Giá trị"]
            st.dataframe(display, use_container_width=True, hide_index=True)
        st.warning("Kết quả chỉ mang tính minh họa học thuật, không phải khuyến nghị cấp hoặc từ chối tín dụng.")

elif page == "Khám phá dữ liệu":
    st.subheader("Khám phá và phân tích dữ liệu (EDA)")
    st.markdown('<p class="section-note">Các biểu đồ được tạo từ lần chạy gần nhất của main.py.</p>', unsafe_allow_html=True)
    if eda is not None:
        st.dataframe(eda.rename(columns={"Metric":"Chỉ tiêu", "Value":"Giá trị", "Column":"Thuộc tính", "Dtype":"Kiểu dữ liệu", "Missing":"Số giá trị thiếu"}), use_container_width=True, hide_index=True)
    chart_files = [
        ("01_target_distribution.png", "Phân bố nhóm rủi ro tín dụng"),
        ("02_numeric_distributions.png", "Phân bố các thuộc tính dạng số"),
        ("03_numeric_boxplots.png", "Biểu đồ hộp - phát hiện giá trị ngoại lệ"),
        ("04_numeric_correlation_heatmap.png", "Ma trận tương quan giữa các thuộc tính số"),
        ("05_categorical_risk_1.png", "Rủi ro theo nhóm phân loại 1"),
        ("05_categorical_risk_2.png", "Rủi ro theo nhóm phân loại 2"),
        ("05_categorical_risk_3.png", "Rủi ro theo nhóm phân loại 3"),
        ("05_categorical_risk_4.png", "Rủi ro theo nhóm phân loại 4"),
    ]
    for start in range(0,len(chart_files),2):
        c1,c2 = st.columns(2)
        for col,(filename,title) in zip((c1,c2),chart_files[start:start+2]):
            path = OUT / filename
            with col:
                if path.exists():
                    st.markdown(f"**{title}**")
                    st.image(str(path), use_container_width=True)
                else:
                    st.caption(f"Chưa có biểu đồ: {filename}. Hãy chạy lại `python main.py`.")

elif page == "Đánh giá mô hình":
    st.subheader("Đánh giá và so sánh mô hình")
    st.markdown('<p class="section-note">So sánh baseline và các phương án Pipeline A / Pipeline B trên cùng tập kiểm thử.</p>', unsafe_allow_html=True)
    if comparison is None:
        st.warning("Chưa tìm thấy bảng kết quả so sánh mô hình. Hãy chạy python main.py trước.")
    else:
        view = comparison.copy()
        numeric_display = ["CV_Macro_F1_mean","CV_Macro_F1_std","Accuracy_test","Precision_Bad","Recall_Bad","F1_Bad","Macro_F1_test","Balanced_Accuracy_test","ROC_AUC_Bad"]
        for col in numeric_display:
            if col in view:
                view[col] = view[col].map(lambda x: "—" if pd.isna(x) else f"{x:.3f}")
        st.dataframe(translate_result_table(view), use_container_width=True, hide_index=True)
        st.markdown("### Biểu đồ so sánh")
        chart_map = [
            ("09_accuracy_comparison.png", "Accuracy"),
            ("07_recall_comparison.png", "Khả năng phát hiện nhóm rủi ro cao (Recall)"),
            ("08_macro_f1_comparison.png", "Điểm Macro-F1"),
            ("06_best_model_confusion_matrix.png", "Confusion Matrix của mô hình được chọn"),
        ]
        for i in range(0,len(chart_map),2):
            c1,c2 = st.columns(2)
            for col,(filename,title) in zip((c1,c2),chart_map[i:i+2]):
                path=OUT/filename
                with col:
                    st.markdown(f"**{title}**")
                    if path.exists(): st.image(str(path), use_container_width=True)
                    else: st.caption(f"Chưa có {filename}")
        st.info("Không nên chỉ căn cứ vào Accuracy. Với nhóm rủi ro cao, cần chú ý Precision, Recall, F1-score và chi phí của từng loại dự đoán sai.")

elif page == "Phân tích lỗi":
    st.subheader("Phân tích lỗi dự đoán")
    st.markdown('<p class="section-note">Xem các trường hợp mô hình dự đoán khác với nhãn thực tế và tổng hợp lỗi.</p>', unsafe_allow_html=True)
    if error_summary is not None:
        st.dataframe(error_summary.rename(columns=COLUMN_NAMES), use_container_width=True, hide_index=True)
    if (OUT/"misclassified_cases.csv").exists():
        errors = pd.read_csv(OUT/"misclassified_cases.csv")
        st.metric("Số hồ sơ được dự đoán sai", f"{len(errors):,}")
        st.dataframe(errors.head(100).rename(columns=COLUMN_NAMES), use_container_width=True, hide_index=True)
        st.caption("Bảng hiển thị tối đa 100 dòng đầu để giao diện tải nhanh; file CSV gốc vẫn chứa toàn bộ trường hợp.")
    else:
        st.info("Chưa có dữ liệu phân tích lỗi. Hãy chạy python main.py trước.")
    text_summary = read_text_if_exists("result_summary.txt")
    if text_summary:
        with st.expander("Xem tóm tắt kết quả thực nghiệm"):
            st.text(text_summary)

st.divider()
st.caption("Đề tài 11 · Đánh giá rủi ro tín dụng khách hàng | Sản phẩm học tập môn Nhập môn học máy")
