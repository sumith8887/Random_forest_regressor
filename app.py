import pandas as pd
import streamlit as st
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Insurance Charges Prediction", layout="wide")
st.title("Insurance Charges Prediction using Random Forest Regressor")
st.caption("Streamlit app built for the insurance dataset.")

DATA_FILE = "data/insurance.csv"
ALT_DATA_FILE = "data/insurence.csv"  # in case your folder uses this spelling
TARGET_COL = "charges"

NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@st.cache_data
def load_data():
    for file_path in [DATA_FILE, ALT_DATA_FILE]:
        if Path(file_path).exists():
            df = pd.read_csv(file_path)
            df.columns = [c.strip().lower() for c in df.columns]
            return df
    return None


@st.cache_resource
def train_model(df):
    missing = [c for c in ALL_FEATURES + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError("Missing expected columns: " + ", ".join(missing))

    X = df[ALL_FEATURES].copy()
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")

    valid_mask = y.notna()
    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                max_depth=None,
                n_jobs=-1
            )),
        ]
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": mean_squared_error(y_test, preds) ** 0.5,
        "r2": r2_score(y_test, preds),
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
    }

    return model, metrics, df


df = load_data()

if df is None:
    st.error("Dataset file not found. Please place it at data/insurance.csv")
    st.stop()

try:
    model, metrics, df = train_model(df)
except Exception as exc:
    st.error(str(exc))
    st.stop()

left, right = st.columns([1.05, 0.95])

with left:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    with st.expander("Show dataset summary"):
        st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
        st.write(df.describe(include="all"))

with right:
    st.subheader("Model Performance")
    st.metric("MAE", f"{metrics['mae']:.2f}")
    st.metric("RMSE", f"{metrics['rmse']:.2f}")
    st.metric("R² Score", f"{metrics['r2']:.3f}")
    st.write("Train shape:", metrics["train_shape"])
    st.write("Test shape:", metrics["test_shape"])

st.divider()
st.subheader("Predict Insurance Charges")

c1, c2 = st.columns(2)

with c1:
    age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
    children = st.number_input("Children", min_value=0, max_value=20, value=0, step=1)

with c2:
    sex = st.selectbox("Sex", sorted(df["sex"].dropna().astype(str).unique().tolist()))
    smoker = st.selectbox("Smoker", sorted(df["smoker"].dropna().astype(str).unique().tolist()))
    region = st.selectbox("Region", sorted(df["region"].dropna().astype(str).unique().tolist()))

input_df = pd.DataFrame(
    [[age, bmi, children, sex, smoker, region]],
    columns=ALL_FEATURES
)

if st.button("Predict Charges"):
    predicted_charges = float(model.predict(input_df)[0])
    st.success(f"Predicted Insurance Charges: ${predicted_charges:,.2f}")

st.caption("Educational demo only.")