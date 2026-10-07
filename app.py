import streamlit as st
import pandas as pd

def suggest_role(series):
    values = series.dropna()

    if values.empty:
        return "Empty"

    column_name = str(series.name).strip().lower()
    uniqueness_ratio = values.nunique() / len(values)

    looks_like_id = (
        column_name == "id"
        or column_name.endswith("_id")
        or column_name.endswith(" id")
    )

    if looks_like_id and uniqueness_ratio >= 0.9:
        return "Possible identifier"

    if pd.api.types.is_bool_dtype(series):
        return "Boolean"

    if pd.api.types.is_numeric_dtype(series):
        if (
            pd.api.types.is_integer_dtype(series)
            and values.nunique() <= 20
        ):
            return "Low-cardinality integer (possibly a category/code)"
        return "Numeric"

    unique_count = values.nunique()
    category_limit = min(20, max(2, int(len(values) * 0.3)))

    if unique_count <= category_limit:
        return "Categorical text"

    return "High-cardinality text"

st.set_page_config(page_title="CSV Explorer", page_icon="📊")

st.title("CSV Explorer")
st.write("Upload a CSV to start exploring your data.")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Could not read this CSV: {error}")
    else:
        st.success(f"Loaded: {uploaded_file.name}")

        row_col, column_col = st.columns(2)
        row_col.metric("Rows", df.shape[0])
        column_col.metric("Columns", df.shape[1])

        st.subheader("Preview")
        st.dataframe(df.head(10), width="stretch")

        st.subheader("Data quality")

        missing_cells = int(df.isna().sum().sum())
        empty_rows = int(df.isna().all(axis=1).sum())
        duplicate_rows = int(df.duplicated().sum())

        missing_col, empty_col, duplicate_col = st.columns(3)
        missing_col.metric("Missing cells", missing_cells)
        empty_col.metric("Fully empty rows", empty_rows)
        duplicate_col.metric("Duplicate rows", duplicate_rows)

        missing_summary = pd.DataFrame(
            {
                "Column": df.columns,
                "Missing values": df.isna().sum().to_numpy(),
                "Missing %": (df.isna().mean() * 100).round(1).to_numpy(),
                "Pandas dtype": df.dtypes.astype(str).to_numpy(),
                "Unique values": df.nunique(dropna=True).to_numpy(),
            }
        )

        st.write("Column Overview")
        st.dataframe(missing_summary, width="stretch")

        st.subheader("Inspect a column")

        selected_column = st.selectbox("Choose a column", df.columns)
        selected_series = df[selected_column]

        st.write(f"Pandas dtype: `{selected_series.dtype}`")
        st.write(f"Suggested role: **{suggest_role(selected_series)}**")
        st.caption(
    "This is a heuristic based on data type and distinct values. "
    "Column meaning still matters; integer values may be category codes."
)
        st.write(f"Unique non-missing values: {selected_series.nunique(dropna=True)}")

        value_counts = (
            selected_series
            .value_counts(dropna=False)
            .head(10)
            .rename_axis("Value")
            .reset_index(name="Count")
        )

        st.write("Most common values")
        st.dataframe(value_counts, width="stretch")

        if pd.api.types.is_numeric_dtype(selected_series):
            st.write("Numeric summary")
            st.dataframe(
                selected_series.describe().to_frame(name="Value"),
                width="stretch",
            )
            sentinel_candidates = {-1, -9, -99, -999, 999, 9999}
            sentinel_counts = selected_series.value_counts()
            sentinel_counts = sentinel_counts[
                sentinel_counts.index.isin(sentinel_candidates)
            ]

            if not sentinel_counts.empty:
                st.write("Possible sentinel values")
                st.caption(
                    "These values are sometimes used as placeholders. "
                    "Review them in context; they may be valid measurements."
                )

                sentinel_summary = pd.DataFrame(
                    {
                        "Value": sentinel_counts.index,
                        "Count": sentinel_counts.to_numpy(),
                        "Percent of non-missing": (
                            sentinel_counts / selected_series.count() * 100
                        ).round(1).to_numpy(),
                    }
                )

                st.dataframe(sentinel_summary, width="stretch", hide_index=True)