import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# ART Coverage Prediction App
# Developed by Flora
# --------------------------------------------------

st.set_page_config(
    page_title="ART Coverage Prediction App",
    page_icon="🏥",
    layout="wide"
)

st.title("ART Coverage Prediction App")
st.caption("Developed by Flora")

st.write(
    """
    This app predicts ART coverage among HIV-positive pregnant women in PMTCT programmes.
    The dataset is stored inside the project folder and used to train the model.
    """
)

st.markdown("---")

# --------------------------------------------------
# Load dataset from GitHub folder
# --------------------------------------------------

@st.cache_data
def load_data():
    file_path = "ART COVERAGE AMONG HIV PREGNANT WOMEN DATASETS.xlsx"
    xls = pd.ExcelFile(file_path)
    sheet_name = xls.sheet_names[0]
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    return data, sheet_name


try:
    df, sheet_used = load_data()
except Exception as e:
    st.error("Dataset could not be loaded.")
    st.write(
        "Make sure the Excel file is in the same folder as app.py and is named exactly:"
    )
    st.code("ART COVERAGE AMONG HIV PREGNANT WOMEN DATASETS.xlsx")
    st.write("Actual error:")
    st.exception(e)
    st.stop()

st.success(f"Dataset loaded successfully from sheet: {sheet_used}")
st.write("Dataset shape:", df.shape)

with st.expander("Preview dataset"):
    st.dataframe(df.head(), use_container_width=True)

# --------------------------------------------------
# Data cleaning and target creation
# --------------------------------------------------

st.subheader("1. Data Cleaning and Target Creation")

df.columns = df.columns.astype(str).str.strip()

rename_map = {
    "PMTCT_STAT (Pregnant women tested for HIV)": "PMTCT_STAT",
    "PMTCT_STAT_POS (HIV positive Women identified)": "PMTCT_STAT_POS",
    "PMTCT_ART (HIV positive women on ART)": "PMTCT_ART",
    "PMTCT_EID (Babies of HIV positive women tested)": "PMTCT_EID"
}

df = df.rename(columns=rename_map)

required_columns = [
    "COPCC",
    "SNU1",
    "SNU2",
    "FY",
    "PMTCT_STAT",
    "PMTCT_STAT_POS",
    "PMTCT_ART",
    "PMTCT_EID"
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error("The dataset is missing required columns.")
    st.write(missing_columns)
    st.stop()

numeric_cols = [
    "FY",
    "PMTCT_STAT",
    "PMTCT_STAT_POS",
    "PMTCT_ART",
    "PMTCT_EID"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["ART_Coverage"] = np.where(
    df["PMTCT_STAT_POS"] > 0,
    df["PMTCT_ART"] / df["PMTCT_STAT_POS"],
    np.nan
)

df["ART_Coverage_Percent"] = df["ART_Coverage"] * 100

above_100 = df[df["ART_Coverage_Percent"] > 100].copy()

df_model = df.copy()
df_model = df_model.replace([np.inf, -np.inf], np.nan)
df_model = df_model.dropna(subset=["ART_Coverage_Percent"])

df_model = df_model[
    (df_model["ART_Coverage_Percent"] >= 0) &
    (df_model["ART_Coverage_Percent"] <= 100)
]

c1, c2, c3 = st.columns(3)

c1.metric("Original Records", f"{len(df):,}")
c2.metric("Records Above 100% Removed", f"{len(above_100):,}")
c3.metric("Final Modelling Records", f"{len(df_model):,}")

with st.expander("Target variable summary"):
    st.dataframe(
        df_model["ART_Coverage_Percent"].describe().to_frame(),
        use_container_width=True
    )

if len(above_100) > 0:
    with st.expander("Records above 100% ART coverage"):
        st.dataframe(
            above_100[
                [
                    "COPCC",
                    "SNU1",
                    "SNU2",
                    "FY",
                    "PMTCT_STAT_POS",
                    "PMTCT_ART",
                    "ART_Coverage_Percent"
                ]
            ],
            use_container_width=True
        )

# --------------------------------------------------
# Data exploration
# --------------------------------------------------

st.subheader("2. Data Exploration")

eda1, eda2 = st.columns(2)

with eda1:
    st.write("Average ART Coverage by Financial Year")
    fy_coverage = (
        df_model.groupby("FY")["ART_Coverage_Percent"]
        .mean()
        .reset_index()
    )
    st.line_chart(
        fy_coverage,
        x="FY",
        y="ART_Coverage_Percent"
    )

with eda2:
    st.write("Average ART Coverage by Region")
    region_coverage = (
        df_model.groupby("SNU1")["ART_Coverage_Percent"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    st.bar_chart(
        region_coverage,
        x="SNU1",
        y="ART_Coverage_Percent"
    )

with st.expander("Programme indicator summary"):
    st.dataframe(
        df_model[
            [
                "PMTCT_STAT",
                "PMTCT_STAT_POS",
                "PMTCT_ART",
                "PMTCT_EID",
                "ART_Coverage_Percent"
            ]
        ].describe(),
        use_container_width=True
    )

# --------------------------------------------------
# Machine learning preparation
# --------------------------------------------------

st.subheader("3. Machine Learning Preparation")

safe_features = [
    "COPCC",
    "SNU1",
    "SNU2",
    "FY",
    "PMTCT_STAT",
    "PMTCT_STAT_POS",
    "PMTCT_EID"
]

X = df_model[safe_features]
y = df_model["ART_Coverage_Percent"]

st.write("Features used in the model:")
st.write(safe_features)

st.info(
    "PMTCT_ART is not used as a predictor because ART coverage is calculated from PMTCT_ART. "
    "Using it would create data leakage."
)

# --------------------------------------------------
# Model training and evaluation
# --------------------------------------------------

st.subheader("4. Model Training and Evaluation")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=300,
        random_state=42
    ),
    "Gradient Boosting Regressor": GradientBoostingRegressor(
        random_state=42
    ),
    "Neural Network Regressor": MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42
    )
}

results = []
trained_models = {}

with st.spinner("Training models..."):
    for name, model in models.items():
        reg = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        trained_models[name] = reg

results_df = pd.DataFrame(results).sort_values(by="RMSE").reset_index(drop=True)

st.success("Models trained successfully.")

st.dataframe(results_df, use_container_width=True)

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

m1, m2, m3 = st.columns(3)

m1.metric("Best Model", best_model_name)
m2.metric("Lowest RMSE", f"{results_df.iloc[0]['RMSE']:.3f}")
m3.metric("R²", f"{results_df.iloc[0]['R2']:.3f}")

st.write("Model Comparison by RMSE")
st.bar_chart(results_df, x="Model", y="RMSE")

# --------------------------------------------------
# Prediction interface
# --------------------------------------------------

st.subheader("5. Prediction Interface")

st.write(
    """
    Enter PMTCT programme details below. The app will predict ART coverage percentage.
    """
)

left, right = st.columns([1.2, 1])

with left:
    copcc_options = sorted(df_model["COPCC"].dropna().astype(str).unique().tolist())
    copcc = st.selectbox("Country / COPCC", copcc_options)

    snu1_options = sorted(df_model["SNU1"].dropna().astype(str).unique().tolist())
    snu1 = st.selectbox("Region / SNU1", snu1_options)

    snu2_options = sorted(df_model["SNU2"].dropna().astype(str).unique().tolist())
    snu2 = st.selectbox("District / SNU2", snu2_options)

    fy_min = int(df_model["FY"].min())
    fy_max = int(df_model["FY"].max())

    fy = st.number_input(
        "Financial Year",
        min_value=fy_min,
        max_value=fy_max + 5,
        value=fy_max
    )

    pmtct_stat = st.number_input(
        "Pregnant women tested for HIV (PMTCT_STAT)",
        min_value=0,
        value=int(df_model["PMTCT_STAT"].median())
    )

    pmtct_stat_pos = st.number_input(
        "HIV-positive pregnant women identified (PMTCT_STAT_POS)",
        min_value=1,
        value=int(df_model["PMTCT_STAT_POS"].median())
    )

    pmtct_eid = st.number_input(
        "Babies of HIV-positive women tested (PMTCT_EID)",
        min_value=0,
        value=int(df_model["PMTCT_EID"].median())
    )

input_data = pd.DataFrame([{
    "COPCC": copcc,
    "SNU1": snu1,
    "SNU2": snu2,
    "FY": fy,
    "PMTCT_STAT": pmtct_stat,
    "PMTCT_STAT_POS": pmtct_stat_pos,
    "PMTCT_EID": pmtct_eid
}])

with right:
    st.write("Prediction Result")

    if st.button("Predict ART Coverage", use_container_width=True):
        prediction = best_model.predict(input_data)[0]

        display_prediction = max(0, min(100, prediction))

        st.metric("Predicted ART Coverage", f"{display_prediction:.2f}%")

        if display_prediction < 90:
            st.error("Interpretation: ART coverage may need follow-up.")
        elif display_prediction < 95:
            st.warning("Interpretation: ART coverage is moderate.")
        else:
            st.success("Interpretation: ART coverage is high.")

st.markdown("---")

st.subheader("Input Record Used")
st.dataframe(input_data, use_container_width=True)

st.markdown("---")

with st.expander("About this app"):
    st.write(
        """
        This app trains regression models using the PMTCT dataset stored in the project folder.
        The target variable is ART coverage percentage.

        ART Coverage (%) = (PMTCT_ART / PMTCT_STAT_POS) × 100

        Records above 100% are removed from the modelling dataset because they are treated as
        data quality exceptions.

        The prediction should be used as a decision-support estimate, not as a replacement for
        programme review or data quality assessment.
        """
    )

st.markdown(
    """
    <div style='text-align: center; font-size: 14px; color: gray;'>
        Developed by <b>Flora</b>
    </div>
    """,
    unsafe_allow_html=True
)
