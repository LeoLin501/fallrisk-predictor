import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import matplotlib
from matplotlib import font_manager

# 設定中文字型（適用於 Windows）
font_path = "C:/Windows/Fonts/msjh.ttc"  # 微軟正黑體
font_prop = font_manager.FontProperties(fname=font_path)
matplotlib.rcParams['font.family'] = font_prop.get_name()
matplotlib.rcParams['axes.unicode_minus'] = False

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ModuleNotFoundError:
    print("Streamlit 模組未安裝。請使用 pip install streamlit 安裝。")
    STREAMLIT_AVAILABLE = False

if STREAMLIT_AVAILABLE:
    st.set_page_config(page_title="AI 跌倒風險預測平臺", layout="wide")
    st.title("🤖 AI 跌倒風險預測平臺")

    st.sidebar.header("Step 1: 上傳 Excel 資料")
    uploaded_file = st.sidebar.file_uploader("請選擇包含預測欄位與跌倒標籤的 Excel 檔案", type=[".xlsx"])

    risk_score_mapping = {
        "利尿劑": 3,
        "麻醉止痛劑": 3,
        "緩瀉劑": 2,
        "鎮靜安眠藥": 3,
        "降血壓藥": 2,
        "降血糖藥": 2,
        "抗組織胺": 1,
        "肌肉鬆弛劑": 3,
        "抗憂鬱劑": 3,
    }

    if uploaded_file:
        data = pd.read_excel(uploaded_file)

        st.subheader("📋 資料預覽")
        st.dataframe(data.head())

        if "跌倒" not in data.columns:
            st.error("找不到 '跌倒' 欄位。請確認資料正確。")
        else:
            # 藥物轉換為風險分數
            if "藥物" in data.columns:
                data["藥物風險分數"] = data["藥物"].map(risk_score_mapping).fillna(0)

            # 特徵工程
            data["跌倒史"] = data["跌倒史"].astype(int)
            data["性別"] = data["性別"].map({"男": 1, "女": 0})

            features = ["年齡", "性別", "行動力", "認知分數", "藥物風險分數", "跌倒史"]
            X = data[features]
            y = data["跌倒"]

            st.sidebar.header("Step 2: 模型參數設定")
            n_estimators = st.sidebar.slider("樹的數量 (n_estimators)", 10, 500, 100, step=10)
            max_depth = st.sidebar.slider("最大深度 (max_depth)", 1, 50, 10, step=1)
            threshold = st.sidebar.slider("高風險預測門檻 (風險機率)", 0.0, 1.0, 0.7, step=0.05)

            st.sidebar.header("Step 3: 點擊進行預測")
            if st.sidebar.button("開始預測"):
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                model.fit(X, y)

                y_proba = model.predict_proba(X)[:, 1]
                y_pred = [1 if p > threshold else 0 for p in y_proba]

                data_result = data.copy()
                data_result["預測風險機率"] = y_proba
                data_result["預測是否高風險"] = y_pred

                st.subheader("✅ 預測結果")
                st.dataframe(data_result)

                st.subheader("📈 模型分類報告")
                report_df = pd.DataFrame(classification_report(y, y_pred, output_dict=True)).transpose()
                st.dataframe(report_df)

                st.subheader("🧮 混淆矩陣")
                cm = confusion_matrix(y, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["無跌倒", "跌倒"],
                            yticklabels=["無跌倒", "跌倒"], ax=ax)
                ax.set_xlabel("預測", fontproperties=font_prop)
                ax.set_ylabel("實際", fontproperties=font_prop)
                ax.set_title("混淆矩陣", fontproperties=font_prop)
                plt.tight_layout()
                st.pyplot(fig)

                st.subheader("📊 跌倒風險因子（Top 10）")
                feat_importances = pd.Series(model.feature_importances_, index=X.columns)
                top_features = feat_importances.sort_values(ascending=True).tail(10)

                fig2, ax2 = plt.subplots(figsize=(8, 6))
                top_features.plot(kind="barh", color="skyblue", ax=ax2)
                ax2.set_title("影響跌倒的主要因素", fontproperties=font_prop)
                ax2.set_xlabel("重要性", fontproperties=font_prop)
                ax2.set_ylabel("變項", fontproperties=font_prop)
                plt.tight_layout()
                st.pyplot(fig2)

                st.subheader("📄 匯出結果")
                def to_excel(df):
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                        df.to_excel(writer, index=False, sheet_name="預測名單")
                    output.seek(0)
                    return output

                st.download_button("下載完整名單", data=to_excel(data_result), file_name="all_predictions.xlsx")
                st.download_button("下載高風險名單", data=to_excel(data_result[data_result["預測是否高風險"] == 1]),
                                  file_name="high_risk_cases.xlsx")

    else:
        st.info("請先上傳包含 '跌倒' 欄位的 Excel 資料以進行預測。")
