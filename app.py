import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="☕ Coffee Shop Sales Analysis & Prediction",
    layout="wide"
)

st.title("☕ Coffee Shop Sales Analysis & Prediction")
st.caption("EDA + Interactive Machine Learning Dashboard")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("CoffeeShopSales-cleaned.csv")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["month"] = df["transaction_date"].dt.month_name()
    return df

df = load_data()

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("🔎 Filter Data")

location = st.sidebar.selectbox(
    "Store Location",
    ["All"] + sorted(df["store_location"].unique())
)

category = st.sidebar.selectbox(
    "Product Category",
    ["All"] + sorted(df["product_category"].unique())
)

weekday = st.sidebar.selectbox(
    "Weekday",
    ["All"] + sorted(df["weekday"].unique())
)

filtered_df = df.copy()

if location != "All":
    filtered_df = filtered_df[filtered_df["store_location"] == location]

if category != "All":
    filtered_df = filtered_df[filtered_df["product_category"] == category]

if weekday != "All":
    filtered_df = filtered_df[filtered_df["weekday"] == weekday]

# ---------------- KPI METRICS ----------------
st.subheader("📌 Key Metrics")

c1, c2, c3, c4 = st.columns(4)

c1.metric("💰 Total Revenue", f"₹ {filtered_df['total_amount'].sum():,.0f}")
c2.metric("🧾 Transactions", filtered_df.shape[0])
c3.metric("📦 Quantity Sold", int(filtered_df["transaction_qty"].sum()))
c4.metric("🛒 Avg Bill Value", f"₹ {filtered_df['total_amount'].mean():.0f}")

# ---------------- EDA SECTION ----------------
st.divider()
st.subheader("📊 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.write("### Sales by Store Location")
    st.dataframe(
        filtered_df.groupby("store_location")["total_amount"]
        .sum()
        .sort_values(ascending=False)
    )

with col2:
    st.write("### Sales by Month")
    st.dataframe(
        filtered_df.groupby("month")["total_amount"]
        .sum()
        .sort_values(ascending=False)
    )

st.write("### Top 10 Products by Quantity")
st.dataframe(
    filtered_df.groupby("product_type")["transaction_qty"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

st.write("### Average Sales by Weekday")
st.dataframe(
    filtered_df.groupby("weekday")["total_amount"]
    .mean()
    .sort_values(ascending=False)
)

# ---------------- PIVOT TABLES ----------------
st.divider()
st.subheader("📐 Pivot Table Explorer")

pivot_choice = st.selectbox(
    "Select Pivot View",
    [
        "Category vs Store Location",
        "Weekday vs Store Location",
        "Month vs Product Category"
    ]
)

if pivot_choice == "Category vs Store Location":
    pivot = pd.pivot_table(
        filtered_df,
        index="product_category",
        columns="store_location",
        values="total_amount",
        aggfunc="sum"
    )

elif pivot_choice == "Weekday vs Store Location":
    pivot = pd.pivot_table(
        filtered_df,
        index="weekday",
        columns="store_location",
        values="total_amount",
        aggfunc="sum"
    )

else:
    pivot = pd.pivot_table(
        filtered_df,
        index="month",
        columns="product_category",
        values="transaction_qty",
        aggfunc="sum"
    )

st.dataframe(pivot)

# ---------------- MACHINE LEARNING ----------------
st.divider()
st.subheader("🤖 Sales Prediction Model")

st.markdown("Linear Regression model to predict **Total Sales Amount**")

if st.button("🚀 Train Model"):

    model_df = df.drop(
        ["transaction_date", "transaction_time", "week"],
        axis=1,
        errors="ignore"
    )

    le = LabelEncoder()
    cat_cols = [
        "store_location",
        "product_category",
        "product_type",
        "product_detail",
        "weekday",
        "month"
    ]

    for col in cat_cols:
        model_df[col] = le.fit_transform(model_df[col])

    X = model_df.drop("total_amount", axis=1)
    y = model_df["total_amount"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)

    st.success("✅ Model Trained Successfully")

    c1, c2 = st.columns(2)
    c1.metric("📈 R² Score", f"{r2*100:.2f}%")
    c2.metric("📊 Training Size", X_train.shape[0])

    st.write("### Sample Predictions")
    pred_df = pd.DataFrame({
        "Actual Sales": y_test.values[:10],
        "Predicted Sales": y_pred[:10]
    })
    st.dataframe(pred_df)

# ---------------- RAW DATA ----------------
with st.expander("📄 View Filtered Raw Data"):
    st.dataframe(filtered_df)

# ---------------- FOOTER ----------------
st.caption("Coffee Shop Sales Analysis & Prediction | Streamlit Project")
