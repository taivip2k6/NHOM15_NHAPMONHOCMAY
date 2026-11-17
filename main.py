# -*- coding: utf-8 -*-
"""ĐỀ TÀI 11 - ĐÁNH GIÁ RỦI RO TÍN DỤNG KHÁCH HÀNG
Dataset: Statlog German Credit Data (1 = Good, 2 = Bad)
Chạy: python main.py

Quy trình: EDA -> chia train/test -> Pipeline A/B -> GridSearchCV trên train
-> đánh giá test một lần -> phân tích lỗi và lưu kết quả.
"""
from pathlib import Path
import zipfile
import warnings
import json
import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_ZIP = BASE_DIR / "statlog+german+credit+data.zip"
DATA_FILE = BASE_DIR / "german.data"
OUT_DIR = BASE_DIR / "outputs"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

COLUMNS = [
    "checking_account", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_account", "employment_since",
    "installment_rate", "personal_status_sex", "other_debtors",
    "residence_since", "property", "age_years",
    "other_installment_plans", "housing", "existing_credits", "job",
    "liable_people", "telephone", "foreign_worker", "risk"
]


def load_data():
    """Đọc dữ liệu; tự giải nén từ ZIP nếu german.data chưa tồn tại."""
    if not DATA_FILE.exists():
        if not DATA_ZIP.exists():
            raise FileNotFoundError("Không tìm thấy german.data hoặc file ZIP dữ liệu.")
        with zipfile.ZipFile(DATA_ZIP) as archive:
            member = next((n for n in archive.namelist() if n.endswith("german.data")), None)
            if member is None:
                raise FileNotFoundError("Trong ZIP không có german.data.")
            DATA_FILE.write_bytes(archive.read(member))
    data = pd.read_csv(DATA_FILE, sep=r"\s+", header=None, names=COLUMNS)
    if data.shape[1] != len(COLUMNS) or len(data) == 0:
        raise ValueError("Dữ liệu không đúng định dạng Statlog German Credit.")
    if not set(data["risk"].unique()).issubset({1, 2}):
        raise ValueError("Nhãn risk phải gồm 1 (Good) và 2 (Bad).")
    return data


def make_preprocessor(numeric_cols, categorical_cols, scaler):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ])


def to_dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else matrix


def make_pipeline(model, preprocessor, dense_for_nb=False):
    steps = [("preprocess", preprocessor)]
    if dense_for_nb:
        steps.append(("to_dense", FunctionTransformer(to_dense, accept_sparse=True)))
    steps.append(("model", model))
    return Pipeline(steps)


def save_eda(data, numeric_cols, categorical_cols):
    OUT_DIR.mkdir(exist_ok=True)
    summary = [
        ("Số dòng", len(data)), ("Số cột gồm nhãn", data.shape[1]),
        ("Số thuộc tính đầu vào", data.shape[1] - 1),
        ("Số giá trị thiếu", int(data.isna().sum().sum())),
        ("Số dòng trùng lặp", int(data.duplicated().sum())),
        ("Khách hàng Good (nhãn 1)", int((data.risk == 1).sum())),
        ("Khách hàng Bad (nhãn 2)", int((data.risk == 2).sum())),
        ("Số thuộc tính số", len(numeric_cols)),
        ("Số thuộc tính phân loại", len(categorical_cols)),
    ]
    pd.DataFrame(summary, columns=["Chỉ tiêu", "Giá trị"]).to_csv(
        OUT_DIR / "eda_summary.csv", index=False, encoding="utf-8-sig")
    data.describe(include="all").transpose().to_csv(
        OUT_DIR / "eda_describe.csv", encoding="utf-8-sig")
    data.isna().sum().rename("missing_count").to_csv(
        OUT_DIR / "missing_values.csv", encoding="utf-8-sig")
    data.duplicated().rename("is_duplicate").to_csv(
        OUT_DIR / "duplicate_rows.csv", encoding="utf-8-sig")

    counts = data.risk.map({1: "Good - rủi ro thấp", 2: "Bad - rủi ro cao"}).value_counts()
    ax = counts.reindex(["Good - rủi ro thấp", "Bad - rủi ro cao"]).plot(
        kind="bar", figsize=(7, 4))
    ax.set_title("Phân bố nhãn rủi ro tín dụng")
    ax.set_xlabel("Nhóm khách hàng")
    ax.set_ylabel("Số lượng")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_target_distribution.png", dpi=160)
    plt.close()

    if numeric_cols:
        data[numeric_cols].hist(figsize=(12, 8), bins=20)
        plt.suptitle("Phân bố các thuộc tính dạng số")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "02_numeric_distributions.png", dpi=160)
        plt.close()

        fig, ax = plt.subplots(figsize=(12, 6))
        data[numeric_cols].plot(kind="box", ax=ax, rot=35)
        ax.set_title("Boxplot - phát hiện giá trị ngoại lệ ở thuộc tính số")
        ax.set_ylabel("Giá trị")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "03_numeric_boxplots.png", dpi=160)
        plt.close()

        corr = data[numeric_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(9, 7))
        image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=75, ha="right")
        ax.set_yticks(range(len(corr.index)), corr.index)
        ax.set_title("Ma trận tương quan giữa các thuộc tính số")
        fig.colorbar(image, ax=ax, label="Hệ số tương quan")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "04_numeric_correlation_heatmap.png", dpi=160)
        plt.close()
        corr.to_csv(OUT_DIR / "numeric_correlation.csv", encoding="utf-8-sig")

    selected_cat = [c for c in ["checking_account", "credit_history", "purpose", "housing"]
                    if c in categorical_cols]
    for idx, col in enumerate(selected_cat, start=1):
        rate = pd.crosstab(data[col], data["risk"], normalize="index")
        rate = rate.rename(columns={1: "Good", 2: "Bad"})
        ax = rate.plot(kind="bar", stacked=True, figsize=(9, 4))
        ax.set_title(f"Tỷ lệ nhóm rủi ro theo {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Tỷ lệ")
        ax.legend(title="Nhãn")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"05_categorical_risk_{idx}.png", dpi=160)
        plt.close()


def calculate_cost(y_true, y_pred):
    """Chi phí minh họa: Good bị gán Bad = 1; Bad bị gán Good = 5."""
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2])
    return int(cm[0, 1] * 1 + cm[1, 0] * 5)


def get_model_searches(numeric_cols, categorical_cols, cv):
    """Hai pipeline tiền xử lý x hai thuật toán; tham số được chọn chỉ trên train."""
    models = {
        "KNN": (KNeighborsClassifier(), {
            "model__n_neighbors": [3, 5, 7, 9],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        }, False),
        "GaussianNB": (GaussianNB(), {
            "model__var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7],
        }, True),
        "DecisionTree": (DecisionTreeClassifier(random_state=RANDOM_STATE), {
            "model__max_depth": [3, 5, 7, None],
            "model__min_samples_leaf": [1, 5, 10],
            "model__criterion": ["gini", "entropy"],
        }, False),
    }
    scalers = {"Pipeline_A_StandardScaler": StandardScaler(),
               "Pipeline_B_MinMaxScaler": MinMaxScaler()}
    searches = []
    for pipe_name, scaler in scalers.items():
        for model_name, (model, grid, dense) in models.items():
            prep = make_preprocessor(numeric_cols, categorical_cols, scaler)
            estimator = make_pipeline(model, prep, dense_for_nb=dense)
            searches.append((pipe_name, model_name, GridSearchCV(
                estimator=estimator, param_grid=grid, scoring="f1_macro",
                cv=cv, n_jobs=-1, refit=True, return_train_score=True,
                error_score="raise"
            )))
    return searches


def main():
    OUT_DIR.mkdir(exist_ok=True)
    data = load_data()
    X = data.drop(columns="risk")
    y = data["risk"].astype(int)
    numeric_cols = X.select_dtypes(exclude="object").columns.tolist()
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    save_eda(data, numeric_cols, categorical_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    result_rows, fitted = [], {}
    cv_rows = []
    for pipe_name, model_name, search in get_model_searches(numeric_cols, categorical_cols, cv):
        print(f"Đang tìm tham số: {pipe_name} / {model_name} ...")
        search.fit(X_train, y_train)
        estimator = search.best_estimator_
        pred = estimator.predict(X_test)
        cm = confusion_matrix(y_test, pred, labels=[1, 2])
        key = (pipe_name, model_name)
        fitted[key] = estimator
        result_rows.append({
            "Pipeline": pipe_name, "Model": model_name,
            "Best_Params": str(search.best_params_),
            "CV_Macro_F1_mean": search.best_score_,
            "CV_Macro_F1_std": float(search.cv_results_["std_test_score"][search.best_index_]),
            "Accuracy_test": accuracy_score(y_test, pred),
            "Precision_Bad": precision_score(y_test, pred, pos_label=2, zero_division=0),
            "Recall_Bad": recall_score(y_test, pred, pos_label=2, zero_division=0),
            "F1_Bad": f1_score(y_test, pred, pos_label=2, zero_division=0),
            "Macro_F1_test": f1_score(y_test, pred, average="macro"),
            "Balanced_Accuracy_test": __import__("sklearn.metrics").metrics.balanced_accuracy_score(y_test, pred),
            "ROC_AUC_Bad": (
                __import__("sklearn.metrics").metrics.roc_auc_score(
                    (y_test == 2).astype(int), estimator.predict_proba(X_test)[:, list(estimator.classes_).index(2)]
                ) if hasattr(estimator, "predict_proba") else np.nan
            ),
            "Cost_total": calculate_cost(y_test, pred),
        })
        cv_table = pd.DataFrame(search.cv_results_)
        cv_table.insert(0, "Pipeline", pipe_name)
        cv_table.insert(1, "Model", model_name)
        cv_rows.append(cv_table)
        pd.DataFrame(cm, index=["Actual_Good", "Actual_Bad"],
                     columns=["Pred_Good", "Pred_Bad"]).to_csv(
            OUT_DIR / f"confusion_{pipe_name}_{model_name}.csv", encoding="utf-8-sig")
        with open(OUT_DIR / f"classification_{pipe_name}_{model_name}.txt", "w", encoding="utf-8") as f:
            f.write(classification_report(y_test, pred, labels=[1, 2],
                target_names=["Good - rủi ro thấp", "Bad - rủi ro cao"],
                zero_division=0, digits=4))

    results = pd.DataFrame(result_rows)
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_row = {
        "Pipeline": "Baseline", "Model": "MostFrequent", "Best_Params": "{}",
        "CV_Macro_F1_mean": np.nan, "CV_Macro_F1_std": np.nan,
        "Accuracy_test": accuracy_score(y_test, baseline_pred),
        "Precision_Bad": precision_score(y_test, baseline_pred, pos_label=2, zero_division=0),
        "Recall_Bad": recall_score(y_test, baseline_pred, pos_label=2, zero_division=0),
        "F1_Bad": f1_score(y_test, baseline_pred, pos_label=2, zero_division=0),
        "Macro_F1_test": f1_score(y_test, baseline_pred, average="macro"),
        "Balanced_Accuracy_test": __import__("sklearn.metrics").metrics.balanced_accuracy_score(y_test, baseline_pred),
        "ROC_AUC_Bad": np.nan,
        "Cost_total": calculate_cost(y_test, baseline_pred),
    }
    results = pd.concat([results, pd.DataFrame([baseline_row])], ignore_index=True)
    results.to_csv(OUT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
    if cv_rows:
        pd.concat(cv_rows, ignore_index=True).to_csv(
            OUT_DIR / "hyperparameter_search_details.csv", index=False, encoding="utf-8-sig")

    # Chọn mô hình theo Macro-F1 CV trên train; không dùng test để lựa chọn.
    best_idx = results[results.Model != "MostFrequent"]["CV_Macro_F1_mean"].idxmax()
    best_row = results.loc[best_idx]
    best_key = (best_row["Pipeline"], best_row["Model"])
    best_model = fitted[best_key]
    best_pred = best_model.predict(X_test)

    # Lưu mô hình và metadata để tái sử dụng trong ứng dụng demo.
    joblib.dump(best_model, OUT_DIR / "best_credit_risk_model.joblib")
    metadata = {
        "project": "De tai 11 - Credit Risk",
        "target_column": "risk",
        "label_mapping": {"1": "Good", "2": "Bad"},
        "feature_columns": X.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "best_pipeline": best_key[0],
        "best_model": best_key[1],
        "best_params": str(best_row["Best_Params"]),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "warning": "Chi phuc vu hoc tap, khong dung de quyet dinh cap tin dung thuc te."
    }
    with open(OUT_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Lưu xác suất dự đoán để có thể phân tích ngưỡng phân loại.
    if hasattr(best_model, "predict_proba"):
        probabilities = best_model.predict_proba(X_test)
        classes = list(best_model.classes_)
        bad_col = classes.index(2)
        probability_table = pd.DataFrame({
            "actual_risk": y_test.to_numpy(),
            "predicted_risk": best_pred,
            "probability_bad": probabilities[:, bad_col]
        }, index=X_test.index)
        probability_table.to_csv(
            OUT_DIR / "test_probabilities_best_model.csv",
            index=True, index_label="row_index", encoding="utf-8-sig"
        )

    error_table = X_test.copy()
    error_table.insert(0, "actual_risk", y_test)
    error_table["predicted_risk"] = best_pred
    error_table["correct"] = error_table.actual_risk == error_table.predicted_risk
    error_table.to_csv(OUT_DIR / "test_predictions_best_model.csv", index=True,
                       index_label="row_index", encoding="utf-8-sig")
    error_table.loc[~error_table.correct].to_csv(
        OUT_DIR / "misclassified_cases.csv", index=True, index_label="row_index",
        encoding="utf-8-sig")

    cm = confusion_matrix(y_test, best_pred, labels=[1, 2])
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Rủi ro thấp", "Rủi ro cao"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {best_key[0]} / {best_key[1]}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "06_best_model_confusion_matrix.png", dpi=160)
    plt.close()

    plot_data = results[results.Model != "MostFrequent"].copy()
    plot_data["Label"] = plot_data["Model"].replace({"KNeighborsClassifier": "KNN", "GaussianNB": "Naive Bayes", "DecisionTreeClassifier": "Cây quyết định"}) + "\n" + plot_data["Pipeline"].replace({"Pipeline_A_StandardScaler": "Luồng A - StandardScaler", "Pipeline_B_MinMaxScaler": "Luồng B - MinMaxScaler"})
    for metric, title, filename in [
        ("Recall_Bad", "So sánh Recall - nhóm rủi ro cao", "07_recall_comparison.png"),
        ("Macro_F1_test", "So sánh Macro-F1 trên tập kiểm thử", "08_macro_f1_comparison.png"),
        ("Accuracy_test", "So sánh Accuracy trên tập kiểm thử", "09_accuracy_comparison.png"),
    ]:
        ax = plot_data.set_index("Label")[metric].plot(kind="bar", figsize=(11, 5))
        ax.set_title(title)
        ax.set_ylabel({"Recall_Bad": "Recall - nhóm rủi ro cao", "Macro_F1_test": "Điểm Macro-F1", "Accuracy_test": "Accuracy"}.get(metric, metric))
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mô hình / pipeline")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(OUT_DIR / filename, dpi=160)
        plt.close()

    # Phân tích lỗi định lượng của mô hình được chọn.
    tn_good, fp_good_as_bad, fn_bad_as_good, tp_bad = cm.ravel()
    error_summary = pd.DataFrame([
        ("Good dự đoán đúng Good", int(tn_good)),
        ("Good dự đoán nhầm Bad", int(fp_good_as_bad)),
        ("Bad dự đoán nhầm Good (bỏ sót Bad)", int(fn_bad_as_good)),
        ("Bad dự đoán đúng Bad", int(tp_bad)),
        ("Tổng mẫu test", int(len(y_test))),
        ("Tổng dự đoán sai", int((best_pred != y_test).sum())),
    ], columns=["Loại kết quả", "Số lượng"])
    error_summary.to_csv(OUT_DIR / "error_analysis_summary.csv", index=False, encoding="utf-8-sig")

    with open(OUT_DIR / "result_summary.txt", "w", encoding="utf-8") as f:
        f.write("KẾT QUẢ THỰC NGHIỆM - ĐỀ TÀI 11\n")
        f.write("Nhãn: 1 = Good; 2 = Bad.\n")
        f.write(f"Số mẫu: {len(data)}; train: {len(X_train)}; test: {len(X_test)}.\n")
        f.write(f"Chia dữ liệu: test_size={TEST_SIZE}, random_state={RANDOM_STATE}, stratify=y.\n")
        f.write(f"Cross-validation: Stratified {CV_FOLDS}-Fold trên train.\n")
        f.write(f"Mô hình chọn theo CV Macro-F1: {best_key[0]} / {best_key[1]}.\n")
        f.write(f"Tham số tốt nhất: {best_row['Best_Params']}\n")
        for metric in ["Accuracy_test", "Precision_Bad", "Recall_Bad", "F1_Bad", "Macro_F1_test", "Cost_total"]:
            f.write(f"{metric}: {best_row[metric]}\n")
        f.write("\nPhân tích lỗi (Actual Good/Bad x Predicted Good/Bad):\n")
        f.write(str(cm) + "\n")
        f.write(f"Good bị dự đoán nhầm Bad: {fp_good_as_bad}.\n")
        f.write(f"Bad bị dự đoán nhầm Good: {fn_bad_as_good}.\n")
        f.write("Lưu ý: chi phí 1/5 là giả định minh họa theo cách mã hóa trong dự án; cần nêu rõ khi diễn giải.\n")
        f.write("Dữ liệu và mô hình chỉ phục vụ học tập, không dùng để quyết định cấp tín dụng thực tế.\n")

    print("\n=== BẢNG SO SÁNH MÔ HÌNH ===")
    print(results[["Pipeline", "Model", "CV_Macro_F1_mean", "Accuracy_test",
                   "Recall_Bad", "Macro_F1_test", "Cost_total"]].round(4).to_string(index=False))
    print(f"\nMô hình chọn theo CV Macro-F1: {best_key[0]} / {best_key[1]}")
    print(f"Kết quả đã lưu tại: {OUT_DIR}")


if __name__ == "__main__":
    main()
