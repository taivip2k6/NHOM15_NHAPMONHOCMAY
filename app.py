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
CATEGORY_OPTIONS = {
    "checking_account": {
        "A11": "dưới 0 DM", "A12": "từ 0 đến dưới 200 DM",
        "A13": "từ 200 DM trở lên", "A14": "không có tài khoản",
    },
    "credit_history": {
        "A30": "chưa có tín dụng hoặc đã thanh toán đầy đủ",
        "A31": "các khoản tín dụng tại ngân hàng này đã được thanh toán đúng hạn",
        "A32": "các khoản tín dụng hiện tại được thanh toán đúng hạn",
        "A33": "đã từng thanh toán trễ",
        "A34": "lịch sử tín dụng đáng lưu ý hoặc có khoản tín dụng khác",
    },
    "purpose": {
        "A40": "xe mới", "A41": "xe đã qua sử dụng", "A42": "nội thất / thiết bị",
        "A43": "radio / tivi", "A44": "đồ gia dụng", "A45": "sửa chữa",
        "A46": "giáo dục", "A47": "du lịch", "A48": "đào tạo lại",
        "A49": "kinh doanh", "A410": "mục đích khác",
    },
    "savings_account": {
        "A61": "dưới 100 DM", "A62": "từ 100 đến dưới 500 DM",
        "A63": "từ 500 đến dưới 1.000 DM", "A64": "từ 1.000 DM trở lên",
        "A65": "không có hoặc không rõ tiền tiết kiệm",
    },
    "employment_since": {
        "A71": "thất nghiệp", "A72": "dưới 1 năm", "A73": "từ 1 đến dưới 4 năm",
        "A74": "từ 4 đến dưới 7 năm", "A75": "từ 7 năm trở lên",
    },
    "personal_status_sex": {
        "A91": "nam, đã ly hôn / ly thân",
        "A92": "nữ, đã ly hôn / ly thân hoặc đã kết hôn",
        "A93": "nam, độc thân", "A94": "nam, đã kết hôn / góa",
        "A95": "nữ, độc thân",
    },
    "other_debtors": {
        "A101": "không có", "A102": "đồng vay",
        "A103": "người bảo lãnh",
    },
    "property": {
        "A121": "bất động sản", "A122": "bảo hiểm nhân thọ / hợp đồng tiết kiệm",
        "A123": "xe hoặc tài sản khác", "A124": "không rõ hoặc không có tài sản",
    },
    "other_installment_plans": {
        "A141": "ngân hàng", "A142": "cửa hàng", "A143": "không có",
    },
    "housing": {
        "A151": "thuê nhà", "A152": "nhà sở hữu", "A153": "nhà ở miễn phí",
    },
    "job": {
        "A171": "thất nghiệp / lao động phổ thông không cư trú",
        "A172": "lao động phổ thông cư trú", "A173": "nhân viên có tay nghề",
        "A174": "quản lý / tự kinh doanh / chuyên môn cao",
    },
    "telephone": {
        "A191": "không có điện thoại đăng ký", "A192": "có điện thoại đăng ký",
    },
    "foreign_worker": {
        "A201": "có", "A202": "không",
    },
}

# Giữ nguyên thuật ngữ chuyên ngành trong giao diện.
MODEL_NAMES = {
    "KNN": "KNN",
    "KNeighborsClassifier": "KNN",
    "GaussianNB": "GaussianNB",
    "DecisionTree": "DecisionTree",
    "DecisionTreeClassifier": "DecisionTree",
    "MostFrequent": "DummyClassifier (Most Frequent)",
}
PIPELINE_NAMES = {
    "Pipeline_A_StandardScaler": "Pipeline A · StandardScaler",
    "Pipeline_B_MinMaxScaler": "Pipeline B · MinMaxScaler",
    "Baseline": "Baseline",
}
COLUMN_NAMES = {
    "Best_Params": "Best Params",
    "CV_Macro_F1_mean": "CV Macro-F1 Mean",
    "CV_Macro_F1_std": "CV Macro-F1 Std",
    "Accuracy_test": "Accuracy (Test)",
    "Precision_Bad": "Precision (Bad)",
    "Recall_Bad": "Recall (Bad)",
    "F1_Bad": "F1 (Bad)",
    "Macro_F1_test": "Macro-F1 (Test)",
    "Balanced_Accuracy_test": "Balanced Accuracy (Test)",
    "ROC_AUC_Bad": "ROC-AUC (Bad)",
    "Cost_total": "Misclassification Cost*",
    "actual_risk": "Actual Label",
    "predicted_risk": "Predicted Label",
    "correct": "Correct",
    "row_index": "Row Index",
}


def model_name_vi(value):
    value = str(value)
    return MODEL_NAMES.get(value, value)


def pipeline_name_vi(value):
    value = str(value)
    return PIPELINE_NAMES.get(value, value)

def category_option_label(column, value):
    description = CATEGORY_OPTIONS.get(column, {}).get(str(value))
    return f"{value} · {description}" if description else f"{value} · mã trong Dataset"

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
.footnote {padding:.8rem 1rem; border:1px solid #dce7f5; border-radius:10px; background:#f0f6fd; color:#47617f; font-size:.84rem;}
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

pages = {
    "Tổng quan": "Tóm tắt dữ liệu và kết quả của lần thực nghiệm gần nhất.",
    "Dự đoán hồ sơ mẫu": "Nhập một hồ sơ mẫu để xem dự đoán minh họa của mô hình.",
    "Khám phá dữ liệu": "Quan sát cấu trúc, phân phối nhãn và các đặc trưng của German Credit.",
    "Đánh giá mô hình": "So sánh các mô hình trên cùng Test set và xem các metric quan trọng.",
    "Phân tích lỗi": "Xem các trường hợp mô hình dự đoán sai và các kiểu nhầm lẫn.",
}
selected_rows = pd.DataFrame()
if comparison is not None:
    selected_rows = comparison[
        (comparison["Pipeline"] == meta.get("best_pipeline"))
        & (comparison["Model"] == meta.get("best_model"))
    ]
selected_result = selected_rows.iloc[0] if not selected_rows.empty else None

st.sidebar.markdown("## 📊 CreditRisk")
st.sidebar.caption("ĐỀ TÀI 11 · HỌC PHẦN NHẬP MÔN HỌC MÁY")
page = st.sidebar.radio("ĐIỀU HƯỚNG", list(pages), label_visibility="visible")
st.sidebar.caption(pages[page])
st.sidebar.divider()
st.sidebar.markdown("**MÔ HÌNH ĐANG SỬ DỤNG**")
st.sidebar.markdown(f"**{model_name_vi(meta.get('best_model', '—'))}**")
st.sidebar.caption(pipeline_name_vi(meta.get("best_pipeline", "—")))
st.sidebar.caption(f"{len(meta.get('feature_columns', []))} đặc trưng · {len(reference):,} hồ sơ")
if selected_result is not None and pd.notna(selected_result["CV_Macro_F1_mean"]):
    st.sidebar.metric("CV Macro-F1", f"{selected_result['CV_Macro_F1_mean']:.3f}")
st.sidebar.divider()
with st.sidebar.expander("Giải thích nhanh thuật ngữ"):
    st.markdown(
        "- **EDA**: khám phá và mô tả dữ liệu.\n"
        "- **Feature**: thuộc tính đầu vào của hồ sơ.\n"
        "- **Preprocessing**: xử lý dữ liệu trước khi huấn luyện.\n"
        "- **Model**: thuật toán học từ dữ liệu để dự đoán.\n"
        "- **Training / Test set**: dữ liệu huấn luyện / dữ liệu đánh giá.\n"
        "- **Metric**: chỉ số đo hiệu quả của Model.\n"
        "- **Recall (Bad)**: tỷ lệ hồ sơ Bad được phát hiện."
    )
st.sidebar.warning("Chỉ phục vụ học tập. Không dùng để ra quyết định tín dụng thực tế.")

st.markdown("""
<div class="hero">
  <div class="eyebrow">HỌC MÁY · ĐÁNH GIÁ RỦI RO TÍN DỤNG</div>
  <h1>Hệ thống đánh giá rủi ro tín dụng</h1>
  <p>Đề tài 11 — Khám phá dữ liệu, thử nghiệm mô hình và minh họa dự đoán trên bộ German Credit.</p>
</div>
""", unsafe_allow_html=True)

if page == "Tổng quan":
    st.subheader(page)
    st.markdown(f'<p class="section-note">{pages[page]} Model được chọn theo CV Macro-F1 trên Train set.</p>', unsafe_allow_html=True)
    total = len(reference)
    good = int((reference.risk == 1).sum())
    bad = int((reference.risk == 2).sum())
    a, b, c, d = st.columns(4)
    a.metric("Tổng số hồ sơ", f"{total:,}", help="Số bản ghi trong Statlog German Credit.")
    b.metric("Nhãn Good", f"{good:,}", f"{good / total:.1%} tổng dữ liệu")
    c.metric("Nhãn Bad", f"{bad:,}", f"{bad / total:.1%} tổng dữ liệu")
    if selected_result is not None:
        d.metric("Recall (Bad) · Test", f"{selected_result['Recall_Bad']:.1%}")
    else:
        d.metric("Recall (Bad) · Test", "—")

    st.markdown("### Kết quả mô hình được chọn")
    if selected_result is not None:
        x, y, z, w = st.columns(4)
        x.metric("CV Macro-F1 (Train)", f"{selected_result['CV_Macro_F1_mean']:.3f}")
        y.metric("Macro-F1 (Test)", f"{selected_result['Macro_F1_test']:.3f}")
        z.metric("Accuracy (Test)", f"{selected_result['Accuracy_test']:.1%}")
        w.metric("ROC-AUC (Bad)", f"{selected_result['ROC_AUC_Bad']:.3f}")
        st.caption(
            f"Mô hình đang dùng: **{model_name_vi(meta.get('best_model', '—'))}** · "
            f"{pipeline_name_vi(meta.get('best_pipeline', '—'))}. "
            "CV Macro-F1 dùng để chọn Model; Test set chỉ dùng cho đánh giá cuối cùng."
        )
    else:
        st.info("Chưa tìm thấy kết quả của mô hình trong model_comparison.csv. Hãy chạy lại `python main.py`.")

    left, right = st.columns([1, 1.35], gap="large")
    with left:
        st.markdown("### EDA · Phân bố nhãn")
        risk_distribution = pd.DataFrame(
            {"Số hồ sơ": [good, bad]}, index=["Good · rủi ro thấp", "Bad · rủi ro cao"]
        )
        st.bar_chart(risk_distribution, color="#3478c8")
        st.caption("Bộ dữ liệu có tỷ lệ Good/Bad không cân bằng.")
    with right:
        st.markdown("### Quy trình Machine Learning")
        steps = [
            ("01", "Nguồn dữ liệu & EDA", "German Credit · 1.000 hồ sơ · 20 features"),
            ("02", "Preprocessing", "Imputation · One-Hot Encoding · Scaling"),
            ("03", "Training & Model Selection", "3 Models · 2 Pipelines · Stratified 5-Fold CV"),
            ("04", "Evaluation & Error Analysis", "Test set · metrics · Confusion Matrix"),
        ]
        for number, title, detail in steps:
            st.markdown(
                f'<div class="panel" style="display:flex;gap:.9rem;align-items:center;margin:.45rem 0;padding:.75rem 1rem">'
                f'<div style="font-weight:800;color:#3478c8;font-size:1rem">{number}</div>'
                f'<div><div style="font-weight:700;color:#17243a">{title}</div>'
                f'<div class="small-muted">{detail}</div></div></div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        '<div class="footnote"><b>Lưu ý:</b> kết quả chỉ minh họa trên bộ dữ liệu nhỏ/cũ; '
        'Recall (Bad) còn hạn chế. Không dùng để ra quyết định tín dụng thực tế.</div>',
        unsafe_allow_html=True,
    )

elif page == "Dự đoán hồ sơ mẫu":
    st.subheader(page)
    st.markdown(
        '<p class="section-note">Nhập thông tin theo từng nhóm, sau đó chọn '
        '<b>Đánh giá hồ sơ mẫu</b> để xem kết quả.</p>',
        unsafe_allow_html=True,
    )
    st.info(
        "Các giá trị mặc định được lấy từ dữ liệu tham khảo. Với lựa chọn dạng mã, "
        "mô tả tiếng Việt được hiển thị bên cạnh mã gốc; Model vẫn nhận đúng mã của Dataset."
    )
    features = meta["feature_columns"]
    numeric_cols = meta["numeric_columns"]
    with st.form("credit_form_v2"):
        values = {}
        input_groups = [
            ("1 · Khoản vay", ["duration_months", "credit_amount", "purpose", "installment_rate", "other_installment_plans"]),
            ("2 · Lịch sử và năng lực tài chính", ["checking_account", "credit_history", "savings_account", "employment_since", "other_debtors", "property", "existing_credits", "job"]),
            ("3 · Thông tin cá nhân", ["personal_status_sex", "residence_since", "age_years", "housing", "liable_people", "telephone", "foreign_worker"]),
        ]
        for group_index, (group_title, group_columns) in enumerate(input_groups):
            with st.expander(f"{group_title} · {len(group_columns)} trường", expanded=(group_index == 0)):
                left, right = st.columns(2, gap="large")
                for idx, col in enumerate(group_columns):
                    container = left if idx % 2 == 0 else right
                    with container:
                        label = LABELS.get(col, col)
                        help_text = DESCRIPTIONS.get(col, "Giá trị theo mã của bộ dữ liệu German Credit.")
                        if col in numeric_cols:
                            series = pd.to_numeric(reference[col], errors="coerce").dropna()
                            low, high = float(series.min()), float(series.max())
                            default = float(series.median())
                            if low.is_integer() and high.is_integer():
                                values[col] = st.number_input(
                                    label, min_value=int(low), max_value=int(high),
                                    value=int(round(default)), step=1, help=help_text,
                                )
                            else:
                                values[col] = st.number_input(
                                    label, min_value=low, max_value=high, value=default, help=help_text,
                                )
                        else:
                            options = sorted(reference[col].dropna().astype(str).unique().tolist())
                            values[col] = st.selectbox(
                                label, options, help=help_text,
                                format_func=lambda value, column=col: category_option_label(column, value),
                            )
        st.caption(
            "Các nhóm 2 và 3 có thể mở rộng khi cần. Giá trị số mặc định là trung vị; "
            "giá trị phân loại mặc định là một mã có trong dữ liệu."
        )
        submitted = st.form_submit_button("🔎 Đánh giá hồ sơ mẫu", type="primary", use_container_width=True)
    if submitted:
        sample = pd.DataFrame([values], columns=features)
        prediction = int(model.predict(sample)[0])
        predicted_label = "Bad · rủi ro cao" if prediction == 2 else "Good · rủi ro thấp"
        st.markdown("### Kết quả dự đoán minh họa")
        if prediction == 2:
            st.error(f"**Mô hình phân loại hồ sơ vào nhóm {predicted_label}.**")
        else:
            st.success(f"**Mô hình phân loại hồ sơ vào nhóm {predicted_label}.**")
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(sample)[0]
            classes = list(model.classes_)
            if 2 in classes:
                p_bad = float(proba[classes.index(2)])
                p_good = float(proba[classes.index(1)]) if 1 in classes else 1 - p_bad
                m1, m2 = st.columns(2)
                m1.metric("Model Score · Good", f"{p_good:.1%}")
                m2.metric("Model Score · Bad", f"{p_bad:.1%}")
                st.progress(p_bad, text="Model Score cho lớp Bad")
                st.caption("Đây là đầu ra predict_proba, không phải xác suất vỡ nợ đã được calibration.")
        with st.expander("Xem lại thông tin hồ sơ"):
            display = pd.DataFrame([values]).rename(columns=LABELS).T.astype(str).reset_index()
            display.columns = ["Thuộc tính", "Giá trị"]
            st.dataframe(display, use_container_width=True, hide_index=True)
        st.markdown(
            '<div class="footnote"><b>Chỉ mang tính minh họa học thuật.</b> '
            'Không sử dụng kết quả này để cấp hoặc từ chối tín dụng.</div>',
            unsafe_allow_html=True,
        )

elif page == "Khám phá dữ liệu":
    st.subheader(page)
    st.markdown('<p class="section-note">Các thống kê và biểu đồ được tạo từ dữ liệu German Credit và lần chạy gần nhất của pipeline.</p>', unsafe_allow_html=True)
    total = len(reference)
    missing_count = int(reference.isna().sum().sum())
    duplicate_count = int(reference.duplicated().sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Số hồ sơ", f"{total:,}")
    m2.metric("Số đặc trưng", f"{len(meta.get('feature_columns', []))}")
    m3.metric("Giá trị thiếu", f"{missing_count:,}")
    m4.metric("Dòng trùng lặp", f"{duplicate_count:,}")
    if eda is not None:
        with st.expander("Xem bảng tóm tắt EDA", expanded=False):
            st.dataframe(eda, use_container_width=True, hide_index=True)
    target_tab, numeric_tab, categorical_tab = st.tabs(
        ["Target Distribution", "Numeric Features", "Categorical Features"]
    )
    with target_tab:
        st.markdown("#### Target Distribution · Good / Bad")
        chart_path = OUT / "01_target_distribution.png"
        if chart_path.exists():
            st.image(str(chart_path), caption="Phân bố nhãn Good / Bad", use_container_width=True)
        else:
            st.info("Chưa có biểu đồ phân bố nhãn. Hãy chạy lại `python main.py`.")
    with numeric_tab:
        st.markdown("#### Numeric Feature Analysis")
        numeric_charts = [
            ("02_numeric_distributions.png", "Numeric Feature Distributions"),
            ("03_numeric_boxplots.png", "Boxplot · Outlier Inspection"),
            ("04_numeric_correlation_heatmap.png", "Numeric Feature Correlation"),
        ]
        for filename, title in numeric_charts:
            chart_path = OUT / filename
            if chart_path.exists():
                st.markdown(f"**{title}**")
                st.image(str(chart_path), use_container_width=True)
            else:
                st.info(f"Chưa có biểu đồ `{filename}`. Hãy chạy lại `python main.py`.")
    with categorical_tab:
        st.markdown("#### Categorical Feature Analysis · Bad Rate")
        categorical_charts = [
            ("05_categorical_risk_1.png", "Rủi ro theo nhóm tài khoản"),
            ("05_categorical_risk_2.png", "Rủi ro theo lịch sử tín dụng"),
            ("05_categorical_risk_3.png", "Rủi ro theo mục đích vay"),
            ("05_categorical_risk_4.png", "Rủi ro theo tình trạng nhà ở"),
        ]
        for start in range(0, len(categorical_charts), 2):
            columns = st.columns(2, gap="large")
            for column, (filename, title) in zip(columns, categorical_charts[start:start + 2]):
                chart_path = OUT / filename
                with column:
                    st.markdown(f"**{title}**")
                    if chart_path.exists():
                        st.image(str(chart_path), use_container_width=True)
                    else:
                        st.caption(f"Chưa có biểu đồ `{filename}`. Hãy chạy lại `python main.py`.")

elif page == "Đánh giá mô hình":
    st.subheader(page)
    st.markdown('<p class="section-note">So sánh Baseline và các phương án Pipeline A / Pipeline B trên cùng Test set. Model sử dụng trong demo được chọn theo CV Macro-F1 trên Train set.</p>', unsafe_allow_html=True)
    if comparison is None:
        st.warning("Chưa tìm thấy bảng kết quả so sánh mô hình. Hãy chạy python main.py trước.")
    else:
        if selected_result is not None:
            st.success(
                f"Model đang dùng: **{model_name_vi(meta.get('best_model', '—'))}** · "
                f"{pipeline_name_vi(meta.get('best_pipeline', '—'))} "
                f"(CV Macro-F1 = {selected_result['CV_Macro_F1_mean']:.3f})."
            )
        metric_options = {
            "Macro-F1 (Test)": "Macro_F1_test",
            "Recall (Bad)": "Recall_Bad",
            "Precision (Bad)": "Precision_Bad",
            "Accuracy (Test)": "Accuracy_test",
            "ROC-AUC (Bad)": "ROC_AUC_Bad",
        }
        selected_metric = st.selectbox("Chọn chỉ số để so sánh", list(metric_options))
        metric_column = metric_options[selected_metric]
        chart_data = comparison.copy()
        chart_data["Phương án"] = chart_data.apply(
            lambda row: (
                "Baseline" if row["Model"] == "MostFrequent"
                else f"{row['Model']} · "
                f"{'A' if row['Pipeline'] == 'Pipeline_A_StandardScaler' else 'B'}"
            ),
            axis=1,
        )
        st.bar_chart(
            chart_data.set_index("Phương án")[[metric_column]].dropna(),
            color="#3478c8",
            y_label=selected_metric,
        )
        st.caption("A = StandardScaler · B = MinMaxScaler · Baseline = DummyClassifier (most_frequent). Metric được tính trên Test set.")
        view = comparison.copy()
        numeric_display = [
            "CV_Macro_F1_mean", "CV_Macro_F1_std", "Accuracy_test",
            "Precision_Bad", "Recall_Bad", "F1_Bad", "Macro_F1_test",
            "Balanced_Accuracy_test", "ROC_AUC_Bad",
        ]
        for col in numeric_display:
            if col in view:
                view[col] = view[col].map(lambda value: "—" if pd.isna(value) else f"{value:.3f}")
        st.markdown("### Model Comparison")
        st.dataframe(translate_result_table(view), use_container_width=True, hide_index=True)
        confusion_path = OUT / "06_best_model_confusion_matrix.png"
        if confusion_path.exists():
            st.markdown("### Confusion Matrix · Selected Model")
            st.image(str(confusion_path), caption="Actual label × Predicted label", use_container_width=True)
        with st.expander("Thiết lập thực nghiệm & phạm vi GridSearchCV"):
            setup_left, setup_right = st.columns(2)
            setup_left.markdown(
                f"- Train/Test split: **{1 - meta.get('test_size', 0.2):.0%} / {meta.get('test_size', 0.2):.0%}**\n"
                f"- `stratify=y`, `random_state={meta.get('random_state', 42)}`\n"
                f"- Cross-validation: **Stratified {meta.get('cv_folds', 5)}-Fold**\n"
                "- Model selection metric: **Macro-F1 (CV)**"
            )
            setup_right.markdown(
                "- Preprocessing nằm trong `sklearn Pipeline` để fit riêng trong từng CV fold.\n"
                "- Search trên Train set; Test set chỉ dùng đánh giá cuối cùng.\n"
                "- So sánh KNN, GaussianNB, DecisionTree và DummyClassifier Baseline."
            )
        st.markdown(
            '<div class="footnote"><b>Cách đọc metric:</b> không đánh giá chỉ bằng Accuracy. '
            'Recall (Bad) cho biết tỷ lệ hồ sơ Bad được phát hiện; cần xem đồng thời Precision, '
            'F1, Macro-F1, Balanced Accuracy, ROC-AUC và Confusion Matrix.</div>',
            unsafe_allow_html=True,
        )
        st.caption("* Misclassification Cost dùng giả định minh họa trong code: Good→Bad = 1, Bad→Good = 5.")

elif page == "Phân tích lỗi":
    st.subheader(page)
    st.markdown('<p class="section-note">Tập trung vào các trường hợp mô hình dự đoán khác nhãn thật; lọc theo nhóm để xem kiểu nhầm lẫn.</p>', unsafe_allow_html=True)
    if error_summary is not None:
        st.dataframe(
            error_summary.rename(columns={"Loại kết quả": "Error Type", "Số lượng": "Count"}),
            use_container_width=True,
            hide_index=True,
        )
    if (OUT / "misclassified_cases.csv").exists():
        errors = pd.read_csv(OUT / "misclassified_cases.csv")
        good_as_bad = int(((errors["actual_risk"] == 1) & (errors["predicted_risk"] == 2)).sum())
        bad_as_good = int(((errors["actual_risk"] == 2) & (errors["predicted_risk"] == 1)).sum())
        e1, e2, e3 = st.columns(3)
        e1.metric("Misclassified Samples", f"{len(errors):,}")
        e2.metric("False Positive · Good → Bad", f"{good_as_bad:,}")
        e3.metric("False Negative · Bad → Good", f"{bad_as_good:,}")
        if bad_as_good > good_as_bad:
            st.warning(
                "Trong Test set, False Negative nhiều hơn False Positive. "
                "Model đã bỏ sót một số mẫu mang nhãn Bad; cần diễn giải cùng kích thước mẫu và giới hạn dữ liệu."
            )
        elif good_as_bad > bad_as_good:
            st.info(
                "Trong Test set, False Positive nhiều hơn False Negative. "
                "Một số mẫu Good bị phân loại nhầm thành Bad; đây là kiểu lỗi cần nêu trong Error Analysis."
            )
        actual_filter, predicted_filter = st.columns(2)
        label_options = {
            "Tất cả": None,
            "Good · nhãn 1": 1,
            "Bad · nhãn 2": 2,
        }
        actual_choice = actual_filter.selectbox("Lọc theo nhãn thật", list(label_options), key="actual_error_filter")
        predicted_choice = predicted_filter.selectbox("Lọc theo nhãn dự đoán", list(label_options), key="predicted_error_filter")
        filtered_errors = errors
        if label_options[actual_choice] is not None:
            filtered_errors = filtered_errors[filtered_errors["actual_risk"] == label_options[actual_choice]]
        if label_options[predicted_choice] is not None:
            filtered_errors = filtered_errors[filtered_errors["predicted_risk"] == label_options[predicted_choice]]
        st.caption(f"Đang hiển thị {len(filtered_errors):,} / {len(errors):,} trường hợp sai.")
        st.dataframe(
            filtered_errors.head(100).rename(columns={**LABELS, **COLUMN_NAMES}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Bảng hiển thị tối đa 100 dòng đầu; file CSV trong outputs vẫn chứa toàn bộ trường hợp sai.")
    else:
        st.info("Chưa có dữ liệu phân tích lỗi. Hãy chạy python main.py trước.")
    text_summary = read_text_if_exists("result_summary.txt")
    if text_summary:
        with st.expander("Xem tóm tắt kết quả thực nghiệm"):
            st.text(text_summary)

st.divider()
st.caption("Đề tài 11 · Đánh giá rủi ro tín dụng khách hàng | Sản phẩm học tập môn Nhập môn học máy")
