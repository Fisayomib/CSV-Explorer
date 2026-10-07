import streamlit as st
import pandas as pd
import re
import plotly.express as px
import numpy as np

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

        rows_with_missing_mask = df.isna().any(axis=1)
        rows_with_missing = int(rows_with_missing_mask.sum())

        missing_col, empty_col, duplicate_col, affected_rows_col = st.columns(4)
        missing_col.metric("Missing cells", missing_cells)
        empty_col.metric("Fully empty rows", empty_rows)
        duplicate_col.metric("Duplicate rows", duplicate_rows)
        affected_rows_col.metric("Rows with missing values", rows_with_missing)

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

        st.subheader("Things to review")

        suggestions = []

        if missing_cells:
            top_missing = missing_summary.sort_values(
                "Missing %", ascending=False
            ).iloc[0]
            suggestions.append(
                f"Start by reviewing {top_missing['Column']}: "
                f"{int(top_missing['Missing values'])} missing values "
                f"({top_missing['Missing %']}%). Check what they mean "
                "before deciding whether to fill or remove them."
            )

        if empty_rows:
            suggestions.append(
                f"There are {empty_rows:,} fully empty rows to review."
            )

        if duplicate_rows:
            suggestions.append(
                f"There are {duplicate_rows:,} repeated rows to inspect "
                "before deciding whether they should be removed."
            )

        if suggestions:
            for suggestion in suggestions:
                st.info(suggestion)
        else:
            st.success("No missing cells, fully empty rows, or exact duplicates were found.")

        with st.expander(f"Inspect rows with missing values ({rows_with_missing})"):
            if rows_with_missing:
                st.dataframe(
                    df.loc[rows_with_missing_mask].head(100),
                    width="stretch",
                )
                st.caption("Showing up to 100 affected rows.")
            else:
                st.write("No rows contain missing values.")

        duplicate_rows_mask = df.duplicated(keep=False)

        with st.expander(f"Inspect exact duplicate rows ({duplicate_rows} repeated copies)"):
            duplicate_records = df.loc[duplicate_rows_mask]

            if not duplicate_records.empty:
                st.dataframe(
                    duplicate_records.head(100),
                    width="stretch",
                )
                st.caption("Showing up to 100 rows that are exact duplicates.")
            else:
                st.write("No exact duplicate rows found.")

        st.subheader("Find rows by text")

        text_columns = [
            column for column in df.columns
            if pd.api.types.is_string_dtype(df[column])
        ]

        if not text_columns:
            st.info("No text columns found to search.")
        else:
            filter_column = st.selectbox(
                "Text column",
                text_columns,
                key="text_filter_column",
            )
            pattern = st.text_input(
                "Search pattern",
                placeholder="For example: kg|tonne",
            )
            st.caption("Search is case-insensitive. In a pattern, | means OR.")

            if pattern:
                try:
                    match_mask = (
                        df[filter_column]
                        .astype("string")
                        .str.contains(pattern, case=False, na=False, regex=True)
                    )
                    matching_rows = df.loc[match_mask]

                    st.write(f"Matching rows: {len(matching_rows)} of {len(df)}")
                    st.dataframe(matching_rows.head(100), width="stretch")
                except re.error as error:
                    st.error(f"Invalid search pattern: {error}")

        st.subheader("Preview a derived column")
        st.caption(
            "Enter up to two text patterns and the number to assign to each. "
            "Rule 1 takes priority if a row matches both; unmatched rows stay blank."
        )

        if not text_columns:
            st.info("A derived-column rule needs at least one text column.")
        else:
            rule_source_column = st.selectbox(
                "Text column to use",
                text_columns,
                key="derive_source",
            )

            rule_pattern_1 = st.text_input(
                "Rule 1 pattern",
                placeholder="50 kg",
                key="derive_pattern_1",
            )
            rule_value_1 = st.number_input(
                "Rule 1 value",
                value=50.0,
                key="derive_value_1",
            )

            rule_pattern_2 = st.text_input(
                "Rule 2 pattern",
                placeholder="1/2 tonne",
                key="derive_pattern_2",
            )
            rule_value_2 = st.number_input(
                "Rule 2 value",
                value=500.0,
                key="derive_value_2",
            )

            output_column = st.text_input(
                "New column name",
                value="derived_value",
                key="derive_output",
            )

            if output_column.strip():
                if output_column in df.columns:
                    st.warning("Choose a new column name that does not already exist.")
                else:
                    rule_pairs = [
                        (rule_pattern_1, rule_value_1),
                        (rule_pattern_2, rule_value_2),
                    ]
                    active_rules = [
                        (pattern, value)
                        for pattern, value in rule_pairs
                        if pattern.strip()
                    ]

                    if not active_rules:
                        st.info("Enter at least one rule pattern.")
                    else:
                        try:
                            conditions = []
                            values = []

                            for pattern, value in active_rules:
                                condition = (
                                    df[rule_source_column]
                                    .astype("string")
                                    .str.contains(
                                        pattern,
                                        case=False,
                                        na=False,
                                        regex=True,
                                    )
                                )
                                conditions.append(condition)
                                values.append(value)

                            match_mask = pd.Series(False, index=df.index)
                            for condition in conditions:
                                match_mask |= condition

                            preview = df[[rule_source_column]].copy()
                            preview[output_column] = np.select(
                                conditions,
                                values,
                                default=np.nan,
                            )

                            st.write(
                                f"Rows matching at least one rule: "
                                f"{int(match_mask.sum())} of {len(df)}"
                            )
                            st.dataframe(preview.head(100), width="stretch")
                        except re.error as error:
                            st.error(f"Invalid search pattern: {error}")

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

        st.write("Distribution")

        if selected_series.dropna().empty:
            st.info("No non-missing values to chart.")
        elif suggest_role(selected_series) == "Possible identifier":
            st.info("This column looks like an identifier, so a chart may not be useful.")
        elif (
            pd.api.types.is_numeric_dtype(selected_series)
            and suggest_role(selected_series) == "Numeric"
        ):
            chart_data = selected_series.dropna().to_frame(name="Value")
            figure = px.histogram(
                chart_data,
                x="Value",
                nbins=30,
                title=f"Distribution of {selected_column}",
            )
            st.plotly_chart(figure, width="stretch")
        else:
            chart_counts = (
                selected_series.dropna().astype("string").value_counts().head(15)
            )
            chart_data = chart_counts.rename_axis("Value").reset_index(name="Count")
            figure = px.bar(
                chart_data,
                x="Count",
                y="Value",
                orientation="h",
                title=f"Most common values in {selected_column}",
            )
            st.plotly_chart(figure, width="stretch")

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