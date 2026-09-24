"""
=====================================================================
 PREDICTIVE ANALYTICS USING HISTORICAL DATA
 A beginner-friendly Streamlit web application
=====================================================================
Workflow:
    Data Upload -> Data Cleaning -> Data Visualization ->
    Model Training -> Prediction -> Accuracy Evaluation

Model used : Linear Regression (scikit-learn)
Charts     : Plotly
Run with   : streamlit run app.py
"""

# ---------------------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------------------
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------
# 2. PAGE SETTINGS AND CONSTANTS
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Predictive Analytics Using Historical Data",
    page_icon="📈",
    layout="wide",
)

MIN_ROWS = 10  # minimum number of rows needed to train a model

PAGES = [
    "🏠 Home",
    "📂 Dataset",
    "🧹 Data Preprocessing",
    "📊 Data Visualization",
    "🤖 Model & Evaluation",
    "🔮 Future Forecast",
    "🎓 Explanation & Viva",
]

# Words used to guess which column is the date / the target
DATE_HINTS = ["date", "time", "day", "month", "year", "period"]
TARGET_HINTS = ["sales", "value", "revenue", "price", "amount", "demand",
                "count", "total", "quantity", "profit", "close"]


# ---------------------------------------------------------------------
# 3. LOOK AND FEEL (custom CSS for a modern design)
# ---------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        .hero {
            background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
            padding: 2.2rem 2rem; border-radius: 18px; color: white;
            margin-bottom: 1.5rem;
        }
        .hero h1 { color: white; margin: 0 0 0.5rem 0; font-size: 2.2rem; }
        .hero p  { color: #e0f2fe; font-size: 1.1rem; margin: 0; }
        .step {
            background: #eef2ff; color: #1e1b4b; padding: 1rem 0.8rem;
            border-radius: 14px; border-left: 5px solid #4f46e5;
            min-height: 140px; text-align: center;
        }
        .step .num {
            background: #4f46e5; color: white; width: 28px; height: 28px;
            border-radius: 50%; display: inline-block; line-height: 28px;
            font-weight: bold; margin-bottom: 6px;
        }
        .step small { color: #4338ca; }
        .note {
            background: #ecfeff; color: #164e63; padding: 1rem 1.2rem;
            border-radius: 12px; border-left: 5px solid #06b6d4;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# 4. DATA LOADING HELPERS
# ---------------------------------------------------------------------
def make_demo_data():
    """Create a demo dataset: 2 years of daily sales with a growing trend,
    a weekly pattern, random noise and a few deliberately missing values
    (so that the data-cleaning step can be demonstrated)."""
    rng = np.random.default_rng(42)  # fixed seed -> same data every time
    dates = pd.date_range("2023-01-01", periods=730, freq="D")
    t = np.arange(len(dates))
    weekly_pattern = 10 * np.sin(2 * np.pi * dates.dayofweek / 7)
    sales = 200 + 0.6 * t + weekly_pattern + rng.normal(0, 8, len(dates))

    df = pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "Sales": sales.round(2)})
    missing_positions = rng.choice(len(df), size=15, replace=False)
    df.loc[missing_positions, "Sales"] = np.nan
    return df


def read_uploaded_csv(uploaded_file):
    """Read an uploaded CSV file. Tries UTF-8 first, then Latin-1."""
    for encoding in ("utf-8", "latin-1"):
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("The file could not be read. Please save it as a normal CSV file.")
    df.columns = [str(c).strip() for c in df.columns]  # clean column names
    return df


# ---------------------------------------------------------------------
# 5. AUTOMATIC COLUMN DETECTION
# ---------------------------------------------------------------------
def detect_date_column(df):
    """Guess which column contains dates. Returns None if nothing is found."""
    # (a) A column that is already a datetime type
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col

    # (b) A text column where most values can be converted to dates
    candidates = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        sample = df[col].dropna().head(50)
        if len(sample) == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() >= 0.8:
            candidates.append(col)

    # Prefer a candidate whose name looks like a date column
    for col in candidates:
        if any(hint in col.lower() for hint in DATE_HINTS):
            return col
    if candidates:
        return candidates[0]

    # (c) A numeric column named like a date that holds years (2019, 2020 ...)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and any(h in col.lower() for h in DATE_HINTS):
            values = df[col].dropna()
            if len(values) > 0 and values.between(1800, 2200).all():
                return col
    return None


def detect_target_column(df, date_col):
    """Guess which numeric column we should forecast."""
    numeric_cols = [c for c in df.columns
                    if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
    for col in numeric_cols:
        if any(hint in col.lower() for hint in TARGET_HINTS):
            return col
    if numeric_cols:
        return numeric_cols[0]
    # Text columns that mostly contain numbers (e.g. "$1,200")
    for col in df.columns:
        if col != date_col and clean_numbers(df[col]).notna().mean() >= 0.8:
            return col
    return None


# ---------------------------------------------------------------------
# 6. DATA PREPROCESSING (cleaning)
# ---------------------------------------------------------------------
def convert_to_dates(series):
    """Convert a column to datetime. Unreadable values become NaT (missing)."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series):
        # Numbers such as 2019, 2020 ... are treated as years
        nums = pd.to_numeric(series, errors="coerce")
        valid = nums.dropna()
        if len(valid) > 0 and valid.between(1800, 2200).all():
            return pd.to_datetime(nums.round().astype("Int64").astype(str),
                                  format="%Y", errors="coerce")
        raise ValueError(
            "The selected date column contains plain numbers, not dates. "
            "Please choose a column with dates such as 2024-01-31."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(series, errors="coerce")


def clean_numbers(series):
    """Convert a column to numbers. Text like 'abc' becomes NaN (missing)."""
    if not pd.api.types.is_numeric_dtype(series):
        # Remove commas, spaces and currency symbols, e.g. "$1,200" -> "1200"
        series = series.astype(str).str.replace(r"[,\s$₹€£%]", "", regex=True)
    numbers = pd.to_numeric(series, errors="coerce")
    return numbers.replace([np.inf, -np.inf], np.nan)


def preprocess(df, date_col, target_col):
    """Clean the dataset and return (clean_dataframe, summary_dictionary)."""
    if date_col == target_col:
        raise ValueError("The date column and the target column cannot be the same. "
                         "Please choose two different columns.")

    data = df[[date_col, target_col]].copy()
    rows_before = len(data)

    # Step 1: convert the date column and remove rows with invalid dates
    data[date_col] = convert_to_dates(data[date_col])
    invalid_dates = int(data[date_col].isna().sum())
    data = data.dropna(subset=[date_col])
    if data.empty:
        raise ValueError(f"No value in column '{date_col}' could be read as a date. "
                         "Please select the correct date column.")

    # Step 2: convert the target column to numbers and count problems
    missing_values = int(data[target_col].isna().sum())
    data[target_col] = clean_numbers(data[target_col])
    invalid_values = int(data[target_col].isna().sum()) - missing_values

    if int(data[target_col].notna().sum()) < MIN_ROWS:
        raise ValueError(
            f"Not enough valid data: column '{target_col}' has only "
            f"{int(data[target_col].notna().sum())} valid number(s), but at least "
            f"{MIN_ROWS} are needed. Please select a numeric target column (for example Sales)."
        )

    # Step 3: merge duplicate dates (average) and sort by date
    duplicate_dates = int(data.duplicated(subset=[date_col]).sum())
    data = (data.groupby(date_col, as_index=False)[target_col].mean()
                .sort_values(date_col).reset_index(drop=True))

    # Step 4: fill the remaining missing values using linear interpolation
    values_filled = int(data[target_col].isna().sum())
    data[target_col] = data[target_col].interpolate(method="linear", limit_direction="both")

    if len(data) < MIN_ROWS:
        raise ValueError(f"The dataset has only {len(data)} usable rows. "
                         f"At least {MIN_ROWS} rows are needed to train a model.")
    if data[target_col].nunique() < 2:
        raise ValueError("The target column has the same value everywhere, "
                         "so there is no trend to predict.")

    freq = detect_frequency(data[date_col])
    summary = {
        "rows_before": rows_before,
        "rows_after": len(data),
        "invalid_dates": invalid_dates,
        "missing_values": missing_values,
        "invalid_values": invalid_values,
        "values_filled": values_filled,
        "duplicate_dates": duplicate_dates,
        "start_date": data[date_col].iloc[0],
        "end_date": data[date_col].iloc[-1],
        "period_name": period_name(freq),
    }
    return data, summary


# ---------------------------------------------------------------------
# 7. DATE / FREQUENCY HELPERS
# ---------------------------------------------------------------------
def detect_frequency(dates):
    """Detect how often data is recorded (daily, weekly, monthly ...)."""
    try:
        return pd.infer_freq(pd.DatetimeIndex(dates))
    except (ValueError, TypeError):
        return None


def period_name(freq):
    """Turn a pandas frequency code into a readable word."""
    if freq is None:
        return "periods"
    code = freq.upper().lstrip("0123456789")
    if code in ("T", "MIN") or code.endswith("MIN"):
        return "minutes"
    if code.startswith("B") and not code.startswith(("BM", "BQ", "BY", "BA")):
        return "business days"
    if code.startswith("D"):
        return "days"
    if code.startswith("W"):
        return "weeks"
    if code.startswith("Q"):
        return "quarters"
    if code.startswith("M"):
        return "months"
    if code.startswith(("Y", "A")):
        return "years"
    if code.startswith("H"):
        return "hours"
    return "periods"


def get_future_dates(dates, periods):
    """Create the dates for the future forecast."""
    last_date = dates.iloc[-1]
    freq = detect_frequency(dates)
    if freq is not None:
        try:
            future = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
            if len(future) == periods:
                return future
        except (ValueError, TypeError):
            pass
    # Fallback: use the typical (median) gap between dates
    step = dates.diff().median()
    return pd.DatetimeIndex([last_date + step * i for i in range(1, periods + 1)])


def fmt_dates(values):
    """Format dates as text for tables."""
    s = pd.Series(values).reset_index(drop=True)
    has_time = bool((s.dt.hour != 0).any() or (s.dt.minute != 0).any())
    return s.dt.strftime("%Y-%m-%d %H:%M" if has_time else "%Y-%m-%d")


# ---------------------------------------------------------------------
# 8. MODEL TRAINING, EVALUATION AND FORECASTING
# ---------------------------------------------------------------------
def train_and_evaluate(clean, target_col, test_pct):
    """Train a Linear Regression model and measure its accuracy.

    Idea: every row gets a time number (0, 1, 2, 3 ...). Linear Regression
    learns a straight line:   value = intercept + slope x time_number
    """
    y = clean[target_col].to_numpy(dtype=float)
    n = len(y)
    X = np.arange(n).reshape(-1, 1)  # time number for each row

    # Time-based split: older data = training, newest data = testing
    split = int(round(n * (1 - test_pct / 100)))
    split = max(2, min(split, n - 2))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)          # TRAINING
    pred_test = model.predict(X_test)    # TESTING
    pred_all = model.predict(X)          # line drawn over the whole history

    mse = mean_squared_error(y_test, pred_test)
    mean_level = float(np.mean(np.abs(y_test)))
    metrics = {
        "MAE": mean_absolute_error(y_test, pred_test),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_test, pred_test),
        "MAE %": (mean_absolute_error(y_test, pred_test) / mean_level * 100)
                 if mean_level != 0 else float("nan"),
    }

    # For the FUTURE forecast we re-train the model on ALL available data,
    # because the newest data is the most useful for predicting what comes next.
    final_model = LinearRegression()
    final_model.fit(X, y)

    return {
        "model": model, "final_model": final_model, "split": split,
        "y": y, "pred_all": pred_all, "pred_test": pred_test, "metrics": metrics,
    }


def make_forecast(result, clean, date_col, periods):
    """Predict the next `periods` time steps."""
    n = len(clean)
    X_future = np.arange(n, n + periods).reshape(-1, 1)
    predictions = result["final_model"].predict(X_future)
    future_dates = get_future_dates(clean[date_col], periods)
    rmse = result["metrics"]["RMSE"]
    table = pd.DataFrame({
        "Date": future_dates,
        "Predicted value": predictions,
        "Lower estimate": predictions - rmse,
        "Upper estimate": predictions + rmse,
    })
    return table


# ---------------------------------------------------------------------
# 9. CHART HELPERS (Plotly)
# ---------------------------------------------------------------------
def style_chart(fig, title, y_title):
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title=y_title,
        hovermode="x unified", height=430,
        legend=dict(orientation="h", y=1.12, x=0),
        margin=dict(l=10, r=10, t=80, b=10),
    )
    return fig


def chart_history(clean, date_col, target_col):
    window = 7 if len(clean) >= 60 else 3
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=clean[date_col], y=clean[target_col], mode="lines",
        name="Historical data", line=dict(color="#4f46e5", width=1.5)))
    fig.add_trace(go.Scatter(
        x=clean[date_col], y=clean[target_col].rolling(window).mean(), mode="lines",
        name=f"{window}-period moving average", line=dict(color="#f59e0b", width=3)))
    return style_chart(fig, "Historical Trend", target_col)


def chart_actual_vs_predicted(clean, date_col, target_col, result):
    split, y = result["split"], result["y"]
    dates = clean[date_col]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates[:split], y=y[:split], mode="lines",
                             name="Actual (training data)",
                             line=dict(color="#4f46e5", width=1.5)))
    fig.add_trace(go.Scatter(x=dates[split:], y=y[split:], mode="lines",
                             name="Actual (testing data)",
                             line=dict(color="#f59e0b", width=2)))
    fig.add_trace(go.Scatter(x=dates[:split], y=result["pred_all"][:split], mode="lines",
                             name="Model fit (training)",
                             line=dict(color="#10b981", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=dates[split:], y=result["pred_test"], mode="lines",
                             name="Predicted (testing)",
                             line=dict(color="#ef4444", width=3, dash="dash")))
    return style_chart(fig, "Actual vs Predicted", target_col)


def chart_forecast(clean, date_col, target_col, forecast):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=clean[date_col], y=clean[target_col], mode="lines",
        name="Historical data", line=dict(color="#4f46e5", width=1.5)))
    # Shaded band = typical error range (prediction +/- RMSE)
    fdates = list(forecast["Date"])
    fig.add_trace(go.Scatter(
        x=fdates + fdates[::-1],
        y=list(forecast["Upper estimate"]) + list(forecast["Lower estimate"])[::-1],
        fill="toself", fillcolor="rgba(239,68,68,0.15)",
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
        name="Typical error range (± RMSE)"))
    fig.add_trace(go.Scatter(
        x=forecast["Date"], y=forecast["Predicted value"], mode="lines",
        name="Forecast", line=dict(color="#ef4444", width=3, dash="dash")))
    return style_chart(fig, "Future Forecast", target_col)


# ---------------------------------------------------------------------
# 10. PAGES
# ---------------------------------------------------------------------
def show_home():
    st.markdown(
        """
        <div class="hero">
            <h1>📈 Predictive Analytics Using Historical Data</h1>
            <p>Learn from the past to forecast the future - upload your data,
            train a simple model, and see predictions in seconds.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2])
    with left:
        st.subheader("What is predictive analytics?")
        st.write(
            "**Predictive analytics** means using **historical data** (what already "
            "happened) together with mathematical models to estimate **what is likely "
            "to happen in the future**. Companies use it to forecast sales, stock "
            "demand, prices, website traffic, weather and much more."
        )
        st.write(
            "In this project we use **Linear Regression** - a simple model that finds "
            "the straight-line trend in past data and extends that line into the future."
        )
    with right:
        st.subheader("Project objective")
        st.markdown(
            """
            - Clean and prepare historical data
            - Visualize past trends
            - Train and test a prediction model
            - Measure the model's accuracy
            - Forecast future values
            """
        )

    st.subheader("Project workflow")
    steps = [
        ("1", "Data Upload", "Upload a CSV or use the demo data"),
        ("2", "Data Cleaning", "Fix dates, missing and invalid values"),
        ("3", "Data Visualization", "Explore trends with charts"),
        ("4", "Model Training", "Linear Regression learns the trend"),
        ("5", "Prediction", "Forecast future values"),
        ("6", "Accuracy Evaluation", "MAE, MSE, RMSE and R²"),
    ]
    columns = st.columns(len(steps))
    for col, (num, title, text) in zip(columns, steps):
        col.markdown(
            f'<div class="step"><div class="num">{num}</div><br><b>{title}</b>'
            f'<br><small>{text}</small></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown(
        '<div class="note"><b>How to start:</b> use the <b>sidebar</b> on the left. '
        'Choose the <i>Demo dataset</i> (or upload your own CSV) and then open the pages '
        'one by one: Dataset → Data Preprocessing → Data Visualization → '
        'Model & Evaluation → Future Forecast.</div>',
        unsafe_allow_html=True,
    )


def show_dataset(df, data_name, date_col, target_col):
    st.header("📂 Dataset")
    st.caption(f"Currently using: **{data_name}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Number of rows", f"{df.shape[0]:,}")
    c2.metric("Number of columns", df.shape[1])
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

    st.subheader("Column names")
    st.write(", ".join(f"`{c}`" for c in df.columns))

    st.subheader("Dataset table")
    st.dataframe(df, height=380)

    st.subheader("Column details")
    info = pd.DataFrame({
        "Column": list(df.columns),
        "Data type": df.dtypes.astype(str).to_numpy(),
        "Missing values": df.isna().sum().to_numpy(),
    })
    st.dataframe(info, hide_index=True)

    st.info(f"Selected columns  →  Date column: **{date_col}**   |   "
            f"Target column (to predict): **{target_col}**  "
            "(you can change them in the sidebar)")


def show_preprocessing(df, clean, summary, date_col, target_col):
    st.header("🧹 Data Preprocessing")
    st.write("Real-world data is messy. Before training a model we clean the data "
             "so that the model receives correct, ordered values.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows before", f"{summary['rows_before']:,}")
    c2.metric("Rows after", f"{summary['rows_after']:,}")
    c3.metric("Missing values found", summary["missing_values"])
    c4.metric("Values filled", summary["values_filled"])

    st.subheader("Preprocessing summary")
    steps = pd.DataFrame({
        "Step": [
            "Convert date column to date format",
            "Remove rows with invalid dates",
            "Convert target column to numbers",
            "Invalid (non-numeric) values found",
            "Duplicate dates merged (averaged)",
            "Sort data by date (oldest → newest)",
            "Fill missing values (linear interpolation)",
        ],
        "Result": [
            f"'{date_col}' converted",
            f"{summary['invalid_dates']} row(s) removed",
            f"'{target_col}' converted",
            f"{summary['invalid_values']} value(s) treated as missing",
            f"{summary['duplicate_dates']} duplicate(s)",
            f"{summary['start_date']:%Y-%m-%d} to {summary['end_date']:%Y-%m-%d}",
            f"{summary['values_filled']} value(s) filled",
        ],
    })
    st.dataframe(steps, hide_index=True)
    st.caption(f"Detected time step of the data: **{summary['period_name']}**")

    st.subheader("Missing values per column (original dataset)")
    missing = df.isna().sum()
    if missing.sum() == 0:
        st.success("No missing values were found in the original dataset. ✅")
    else:
        fig = px.bar(x=missing.index.astype(str), y=missing.to_numpy(),
                     labels={"x": "Column", "y": "Missing values"},
                     color_discrete_sequence=["#ef4444"])
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig)

    left, right = st.columns(2)
    with left:
        st.subheader("Before cleaning")
        st.dataframe(df[[date_col, target_col]].head(10), hide_index=True)
    with right:
        st.subheader("After cleaning")
        preview = clean.head(10).copy()
        preview[date_col] = fmt_dates(preview[date_col]).to_numpy()
        st.dataframe(preview, hide_index=True)


def show_visualization(clean, date_col, target_col, summary):
    st.header("📊 Data Visualization")
    st.write("Charts help us **see** the pattern (trend) hidden inside the numbers.")

    st.plotly_chart(chart_history(clean, date_col, target_col))
    st.caption("The orange line is a moving average - it smooths out daily ups and "
               "downs so the overall trend is easier to see.")

    left, right = st.columns(2)
    with left:
        fig = px.histogram(clean, x=target_col, nbins=30,
                           color_discrete_sequence=["#06b6d4"],
                           title="Distribution of values")
        fig.update_layout(height=380)
        st.plotly_chart(fig)
    with right:
        monthly = (clean.assign(Month=clean[date_col].dt.to_period("M").astype(str))
                        .groupby("Month", as_index=False)[target_col].mean())
        if len(monthly) >= 3:
            fig = px.bar(monthly, x="Month", y=target_col,
                         color_discrete_sequence=["#4f46e5"],
                         title="Average value per month")
            fig.update_layout(height=380)
            st.plotly_chart(fig)
        else:
            st.info("The monthly chart needs data from at least 3 different months.")

    st.subheader("Summary statistics")
    st.dataframe(clean[target_col].describe().round(2).to_frame(name=target_col))


def show_model(clean, date_col, target_col, result, test_pct):
    st.header("🤖 Model Training & Evaluation")
    m = result["metrics"]
    split = result["split"]
    n = len(clean)
    dates = clean[date_col]

    st.subheader("How does the model work?")
    st.markdown(
        """
        **Linear Regression** draws the best straight line through the historical data.
        Every row is given a time number (0, 1, 2, 3 ...) and the model learns:

        > **Predicted value = Intercept + Slope × Time number**

        *Slope* tells us how much the value goes up (or down) in every time step.
        Once the line is learned, we extend it into the future to make predictions.
        """
    )

    st.subheader("Training and testing data")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training rows", f"{split:,}")
    c2.metric("Testing rows", f"{n - split:,}")
    c3.metric("Split", f"{100 - test_pct}% / {test_pct}%")
    c4.metric("Model", "Linear Regression")
    st.caption(
        f"Training data: {dates.iloc[0]:%Y-%m-%d} to {dates.iloc[split - 1]:%Y-%m-%d}   |   "
        f"Testing data: {dates.iloc[split]:%Y-%m-%d} to {dates.iloc[-1]:%Y-%m-%d}  "
        "(the newest data is kept aside to test the model)"
    )

    slope = float(result["model"].coef_[0])
    intercept = float(result["model"].intercept_)
    direction = "increases" if slope > 0 else "decreases"
    st.success(
        f"Learned line:  {target_col} = {intercept:,.2f} + ({slope:,.4f} × time number).  "
        f"On average the value **{direction} by {abs(slope):,.4f}** per time step."
    )

    st.subheader("Model accuracy (measured on testing data)")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("MAE", f"{m['MAE']:,.2f}")
    k2.metric("MSE", f"{m['MSE']:,.2f}")
    k3.metric("RMSE", f"{m['RMSE']:,.2f}")
    k4.metric("R² Score", f"{m['R2']:.3f}")

    if m["R2"] >= 0.8:
        st.success("✅ Good fit: the model explains most of the pattern in the data.")
    elif m["R2"] >= 0.5:
        st.info("ℹ️ Fair fit: the model captures part of the pattern. "
                "Predictions are useful but not very precise.")
    else:
        st.warning("⚠️ Weak fit: the data may not follow a straight-line trend, "
                   "so predictions should be used carefully.")

    if not np.isnan(m["MAE %"]):
        st.caption(f"On average, predictions are off by about **{m['MAE %']:.1f}%** "
                   "of the typical value.")

    with st.expander("📘 What do these metrics mean? (simple explanation)", expanded=True):
        st.markdown(
            """
            - **MAE (Mean Absolute Error):** the *average size of the mistakes*. If MAE = 10,
              the predictions are wrong by 10 units on average. **Lower is better.**
            - **MSE (Mean Squared Error):** the average of the *squared* mistakes. Big mistakes
              are punished much more. **Lower is better.**
            - **RMSE (Root Mean Squared Error):** the square root of MSE. It is in the *same
              unit as your data*, so it is easier to understand. **Lower is better.**
            - **R² Score:** how much of the ups and downs in the data the model explains.
              1.0 = perfect, 0 = as bad as guessing the average, negative = worse than that.
              **Higher is better.**
            """
        )

    st.subheader("Actual vs Predicted")
    st.plotly_chart(chart_actual_vs_predicted(clean, date_col, target_col, result))

    st.subheader("Testing results table")
    test_table = pd.DataFrame({
        "Date": fmt_dates(dates.iloc[split:]).to_numpy(),
        "Actual": result["y"][split:].round(2),
        "Predicted": result["pred_test"].round(2),
    })
    test_table["Error (Actual - Predicted)"] = (test_table["Actual"] - test_table["Predicted"]).round(2)
    st.dataframe(test_table, hide_index=True, height=300)


def show_forecast(clean, date_col, target_col, result, periods, summary):
    st.header("🔮 Future Forecast")
    forecast = make_forecast(result, clean, date_col, periods)
    unit = summary["period_name"]

    st.write(f"Predicting the next **{periods} {unit}** using the model "
             "(re-trained on all historical data).")

    last_actual = float(clean[target_col].iloc[-1])
    last_pred = float(forecast["Predicted value"].iloc[-1])
    change = last_pred - last_actual

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last actual value", f"{last_actual:,.2f}")
    c2.metric(f"Predicted at end ({forecast['Date'].iloc[-1]:%Y-%m-%d})", f"{last_pred:,.2f}")
    c3.metric("Expected change", f"{change:,.2f}")
    c4.metric("Forecast period", f"{periods} {unit}")

    slope = float(result["final_model"].coef_[0])
    if slope > 0:
        st.success("📈 The overall trend is **upward** - values are expected to keep increasing.")
    elif slope < 0:
        st.warning("📉 The overall trend is **downward** - values are expected to keep decreasing.")
    else:
        st.info("➡️ The overall trend is **flat**.")

    st.plotly_chart(chart_forecast(clean, date_col, target_col, forecast))
    st.caption("Red dashed line = forecast. The shaded band shows the typical error "
               "(± RMSE) found while testing the model.")

    st.subheader("Forecast table")
    table = forecast.copy()
    table["Date"] = fmt_dates(table["Date"]).to_numpy()
    table = table.round(2)
    st.dataframe(table, hide_index=True, height=350)

    st.download_button(
        "⬇️ Download forecast as CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="forecast.csv",
        mime="text/csv",
    )
    st.info("Note: a forecast is an estimate, not a guarantee. The further into the "
            "future we predict, the less reliable the result becomes.")


def show_viva():
    st.header("🎓 Project Explanation & Viva Questions")

    st.subheader("Project explanation")
    explanation = [
        ("1. Objective",
         "Use past (historical) data to forecast future values and measure how accurate the forecast is."),
        ("2. Technologies used",
         "Python; Streamlit (web app); Pandas and NumPy (data handling); scikit-learn "
         "(Linear Regression and metrics); Plotly (interactive charts)."),
        ("3. How the project works",
         "Load data → clean it → visualize the trend → train the model on older data → "
         "test it on newer data → measure accuracy → predict the future."),
        ("4. Dataset",
         "A CSV file with a date column and a numeric column (for example Date, Sales). "
         "The app includes a demo dataset of 730 daily records and can also read your own CSV. "
         "It detects the columns automatically, or you can pick them in the sidebar."),
        ("5. Data preprocessing",
         "Dates are converted and invalid dates removed; the target is converted to numbers; "
         "duplicate dates are averaged; data is sorted by date; missing values are filled "
         "by linear interpolation (a straight line between the neighbouring known values)."),
        ("6. Model used",
         "Linear Regression. Each row gets a time number (0, 1, 2, ...) and the model learns "
         "the line: value = intercept + slope × time number."),
        ("7. Training and testing",
         "The older part of the timeline (default 80%) is training data; the newest part "
         "(default 20%) is testing data. We split by time, not randomly, because we predict forward."),
        ("8. Evaluation metrics",
         "MAE, MSE, RMSE and R², calculated on the testing data only."),
        ("9. Future prediction",
         "The model is re-trained on all data and the line is extended for the chosen number of "
         "periods. The shaded band shows ± RMSE, the typical error seen during testing."),
        ("10. Visualization",
         "Historical trend with moving average, histogram, monthly averages, Actual vs Predicted "
         "and the future forecast chart - all interactive."),
    ]
    for title, text in explanation:
        with st.expander(title):
            st.write(text)

    st.subheader("Possible teacher questions and simple answers")
    questions = [
        ("What is predictive analytics?",
         "Using past data and mathematical models to estimate what will probably happen in the "
         "future, for example forecasting next month's sales."),
        ("Why did you choose this project?",
         "Forecasting is used in almost every industry (sales, stock demand, weather). It also "
         "lets me learn the full data workflow: cleaning, visualizing, modelling and evaluating."),
        ("What is historical data?",
         "Data recorded in the past over time, such as daily sales for the last two years. "
         "Each record has a date and a value."),
        ("What model did you use?",
         "Linear Regression from the scikit-learn library."),
        ("Why did you choose Linear Regression?",
         "It is simple, fast and easy to explain. It works well when data follows a general upward "
         "or downward trend, and it gives a clear result: a slope showing how much the value "
         "changes in each period."),
        ("What is training data?",
         "The older part of the data that the model learns from."),
        ("What is testing data?",
         "The newest part of the data, kept hidden during learning and used afterwards to check "
         "whether the model predicts correctly."),
        ("What is MAE?",
         "Mean Absolute Error - the average size of the model's mistakes. MAE = 9 means the "
         "predictions are off by about 9 units on average. Lower is better."),
        ("What is RMSE?",
         "Root Mean Squared Error - square the errors, average them, then take the square root. "
         "It is in the same unit as the data and punishes big mistakes more than MAE does. "
         "Lower is better."),
        ("What is R²?",
         "It tells how much of the ups and downs in the data the model explains. 1 is perfect, "
         "0 is no better than guessing the average. R² = 0.84 means the model explains about 84% "
         "of the pattern. Higher is better."),
        ("How does the prediction work?",
         "The model learned a line from history. To predict the next day it takes the next time "
         "number and puts it into the line's equation. Repeating this for each future day gives "
         "the forecast."),
        ("What happens when we upload a CSV?",
         "The app reads the file, shows the table, detects the date and value columns, cleans the "
         "data, trains the model and updates all charts and tables. If something is wrong (no date "
         "column, too few rows, invalid values) it shows a clear message instead of crashing."),
        ("How does the graph show future trends?",
         "The blue line is real history. The red dashed line continues from where the history ends, "
         "following the learned trend. The shaded area shows the typical error, so we can see how "
         "uncertain the prediction is."),
        ("What is a limitation of your project?",
         "Linear Regression assumes a straight-line trend, so it cannot fully capture seasonality "
         "or sudden changes. A future improvement would be a model such as ARIMA or Prophet."),
    ]
    for question, answer in questions:
        with st.expander(f"❓ {question}"):
            st.write(answer)


# ---------------------------------------------------------------------
# 11. MAIN PROGRAM
# ---------------------------------------------------------------------
def main():
    inject_css()

    # ----- Sidebar: navigation -----
    st.sidebar.title("📈 Predictive Analytics")
    page = st.sidebar.radio("Navigation", PAGES)
    st.sidebar.divider()

    # The Home page needs no data
    if page == PAGES[0]:
        st.sidebar.info("Pick a page above. Start with **Dataset**.")
        show_home()
        return
    if page == PAGES[6]:
        show_viva()
        return

    # ----- Sidebar: choose the data source -----
    st.sidebar.subheader("📂 Data source")
    source = st.sidebar.radio(
        "Choose data source",
        ["Demo dataset (Date, Sales)", "Upload my own CSV"],
    )

    df = None
    data_name = ""
    if source.startswith("Demo"):
        df = make_demo_data()
        data_name = "Demo dataset (sample_data.csv)"
        st.sidebar.download_button(
            "⬇️ Download demo CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="sample_data.csv",
            mime="text/csv",
        )
    else:
        uploaded = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded is not None:
            try:
                df = read_uploaded_csv(uploaded)
                data_name = uploaded.name
            except pd.errors.EmptyDataError:
                st.sidebar.error("The uploaded file is empty.")
            except Exception as err:
                st.sidebar.error(f"Could not read the file: {err}")

    # ----- Handle: no dataset / unusable dataset -----
    if df is None:
        st.header(page)
        st.warning("⚠️ No dataset loaded yet. Upload a CSV file in the sidebar, "
                   "or switch to the **Demo dataset** to try the project.")
        return
    if df.empty or df.shape[1] < 2:
        st.header(page)
        st.error("The dataset must contain at least 2 columns (a date column and a "
                 "numeric column) and at least one row.")
        return

    # ----- Sidebar: choose columns (auto-detected, but changeable) -----
    st.sidebar.subheader("🧭 Select columns")
    columns = list(df.columns)
    auto_date = detect_date_column(df)
    if auto_date is None:
        st.sidebar.warning("No date column detected automatically. Please choose one.")
    date_col = st.sidebar.selectbox(
        "Date column", columns,
        index=columns.index(auto_date) if auto_date in columns else 0,
    )
    auto_target = detect_target_column(df, date_col)
    target_col = st.sidebar.selectbox(
        "Target column (value to predict)", columns,
        index=columns.index(auto_target) if auto_target in columns else min(1, len(columns) - 1),
    )

    # ----- Sidebar: model settings -----
    st.sidebar.subheader("⚙️ Model settings")
    test_pct = st.sidebar.slider("Testing data size (%)", 10, 40, 20, step=5)
    slider_periods = st.sidebar.select_slider(
        "Future periods to predict", options=[7, 14, 30, 60, 90, 180], value=30)
    custom_periods = st.sidebar.number_input(
        "Or type a custom number (0 = use slider)", min_value=0, max_value=365, value=0)
    periods = int(custom_periods) if custom_periods > 0 else int(slider_periods)

    # ----- Run the selected page; show friendly errors instead of crashing -----
    try:
        if page == PAGES[1]:
            show_dataset(df, data_name, date_col, target_col)
            return

        clean, summary = preprocess(df, date_col, target_col)

        if page == PAGES[2]:
            show_preprocessing(df, clean, summary, date_col, target_col)
        elif page == PAGES[3]:
            show_visualization(clean, date_col, target_col, summary)
        else:
            result = train_and_evaluate(clean, target_col, test_pct)
            if page == PAGES[4]:
                show_model(clean, date_col, target_col, result, test_pct)
            else:
                show_forecast(clean, date_col, target_col, result, periods, summary)

    except ValueError as err:
        st.header(page)
        st.error(f"⚠️ {err}")
        st.info("Tip: check the **Date column** and **Target column** selected in the sidebar.")
    except Exception as err:
        st.header(page)
        st.error(f"Something unexpected went wrong: {err}")
        st.info("Please check that your CSV has a date column and a numeric column.")


main()
