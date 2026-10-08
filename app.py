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

    identifier_names = {
        "id", "zip", "zipcode", "zip_code", "postal_code", "postcode",
        "phone", "telephone", "mobile",
    }
    looks_like_identifier = (
        column_name in identifier_names
        or column_name.endswith("_id")
        or column_name.endswith(" id")
    )

    if looks_like_identifier and (
        uniqueness_ratio >= 0.9 or column_name in identifier_names - {"id"}
    ):
        return "Possible identifier or code"

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


def column_type_clue(series):
    values = series.dropna()

    if values.empty:
        return "There are no non-missing values to inspect."

    if pd.api.types.is_bool_dtype(series):
        return "The values are stored as true/false."

    if pd.api.types.is_numeric_dtype(series):
        return "Pandas read these values as numbers. Check the column name too: codes such as ZIP codes can look numeric."

    text_values = values.astype("string").str.strip()
    numeric_text = (
        text_values
        .str.replace(r"[$,%]", "", regex=True)
        .str.replace(",", "", regex=False)
    )
    parsed = pd.to_numeric(numeric_text, errors="coerce")
    parse_ratio = parsed.notna().mean()

    if parse_ratio >= 0.8:
        if text_values.str.contains("%", regex=False, na=False).any():
            return "These values look numeric but are stored as text and include percent signs."
        if text_values.str.contains(r"[$,]", regex=True, na=False).any():
            return "These values look numeric but are stored as text and include currency symbols or commas."
        return "These values look numeric but are stored as text."

    return "Pandas read these values as text. The column's meaning may need a human check."
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

                            prepared_df = df.copy()
                            prepared_df[output_column] = np.select(
                                conditions,
                                values,
                                default=np.nan,
                            )
                            preview = prepared_df[
                                [rule_source_column, output_column]
                            ]

                            st.write(
                                f"Rows matching at least one rule: "
                                f"{int(match_mask.sum())} of {len(df)}"
                            )
                            st.dataframe(preview.head(100), width="stretch")

                            file_stem = uploaded_file.name.rsplit(".", 1)[0]
                            st.download_button(
                                "Download prepared CSV",
                                data=prepared_df.to_csv(index=False).encode("utf-8"),
                                file_name=f"{file_stem}_prepared.csv",
                                mime="text/csv",
                                key="download_prepared_csv",
                            )
                        except re.error as error:
                            st.error(f"Invalid search pattern: {error}")

        st.subheader("Inspect a column")
        st.caption(
            "Choose one column for a closer look. These checks are clues to review, "
            "not automatic decisions or changes to your data."
        )

        selected_column = st.selectbox("Choose a column", df.columns)
        selected_series = df[selected_column]
        suggested_role = suggest_role(selected_series)
        non_missing = selected_series.dropna()
        missing_count = int(selected_series.isna().sum())
        missing_percent = (
            missing_count / len(selected_series) * 100
            if len(selected_series)
            else 0
        )
        unique_count = int(non_missing.nunique())
        unique_ratio = (
            unique_count / len(non_missing)
            if len(non_missing)
            else 0
        )
        non_missing_counts = non_missing.value_counts()
        dominant_count = (
            int(non_missing_counts.iloc[0])
            if not non_missing_counts.empty
            else 0
        )
        dominant_percent = (
            dominant_count / len(non_missing) * 100
            if len(non_missing)
            else 0
        )

        st.write(f"**Pandas stored type:** `{selected_series.dtype}`")
        st.write(f"**Possible role:** {suggested_role}")
        st.caption(column_type_clue(selected_series))

        missing_metric, distinct_metric, ratio_metric, dominant_metric = st.columns(4)
        missing_metric.metric("Missing", f"{missing_count:,}", f"{missing_percent:.1f}% of rows")
        distinct_metric.metric("Different values", f"{unique_count:,}")
        ratio_metric.metric("Distinct-value ratio", f"{unique_ratio:.1%}")
        dominant_metric.metric("Most common value", f"{dominant_percent:.1f}%")

        if unique_count == 1 and len(non_missing):
            st.warning("This column has one non-missing value throughout, so it may add little information to a model.")
        elif dominant_percent >= 95:
            st.info(
                f"The most common value appears in {dominant_percent:.1f}% of non-missing rows. "
                "This column may be nearly constant."
            )

        if unique_ratio >= 0.9 and len(non_missing) > 1:
            st.info(
                "Nearly every non-missing row has a different value. "
                "This can indicate an ID, though the column's meaning matters."
            )

        st.write("Most common values")
        if non_missing_counts.empty:
            st.info("There are no non-missing values to count.")
        else:
            value_counts = non_missing_counts.head(10).rename_axis("Value").reset_index(name="Count")
            value_counts["Percent of non-missing"] = (
                value_counts["Count"] / len(non_missing) * 100
            ).round(1)
            st.dataframe(value_counts, width="stretch", hide_index=True)

        st.write("Raw samples")
        st.caption("Compare a few original values from the beginning, end, and a reproducible random sample.")
        sample_parts = []
        for sample_label, sample_values in [
            ("First rows", selected_series.head(3)),
            ("Random rows", selected_series.sample(min(3, len(selected_series)), random_state=42) if len(selected_series) else selected_series),
            ("Last rows", selected_series.tail(3)),
        ]:
            if not sample_values.empty:
                sample_parts.append(
                    pd.DataFrame(
                        {
                            "Sample": sample_label,
                            "Data row": sample_values.index + 1,
                            "Value": sample_values.astype("object").to_numpy(),
                        }
                    )
                )
        if sample_parts:
            st.dataframe(pd.concat(sample_parts, ignore_index=True), width="stretch", hide_index=True)

        placeholder_counts = []
        numeric_placeholders = {-1, -9, -99, -999, 999, 9999}
        text_placeholders = {"n/a", "na", "unknown", "?", "null"}
        for value, count in non_missing_counts.items():
            if isinstance(value, str) and value.strip().casefold() in text_placeholders:
                placeholder_counts.append({"Possible placeholder": value, "Count": int(count)})
            elif (
                pd.api.types.is_numeric_dtype(selected_series)
                and not pd.api.types.is_bool_dtype(selected_series)
                and value in numeric_placeholders
            ):
                placeholder_counts.append({"Possible placeholder": value, "Count": int(count)})

        if placeholder_counts:
            st.write("Values to check as possible placeholders")
            st.dataframe(pd.DataFrame(placeholder_counts), width="stretch", hide_index=True)
            st.caption(
                "These values may be valid. For example, zero can be a real measurement, "
                "so check the column's meaning before treating any value as missing."
            )

        if pd.api.types.is_numeric_dtype(selected_series) and not pd.api.types.is_bool_dtype(selected_series):
            st.write("Numeric details")
            if not non_missing.empty:
                summary = non_missing.describe().to_frame(name="Value")
                summary.loc["median"] = non_missing.median()
                summary.loc["skewness"] = non_missing.skew() if len(non_missing) > 2 else np.nan
                st.dataframe(summary, width="stretch")

                zero_count = int((non_missing == 0).sum())
                negative_count = int((non_missing < 0).sum())
                zero_percent = zero_count / len(non_missing) * 100
                st.write(
                    f"Zero values: **{zero_count:,} ({zero_percent:.1f}%)** · "
                    f"Negative values: **{negative_count:,}**"
                )

                q1 = non_missing.quantile(0.25)
                q3 = non_missing.quantile(0.75)
                iqr = q3 - q1
                lower_fence = q1 - 1.5 * iqr
                upper_fence = q3 + 1.5 * iqr
                outlier_count = int(((non_missing < lower_fence) | (non_missing > upper_fence)).sum())
                st.caption(
                    f"IQR check: {outlier_count:,} values fall outside the usual 1.5×IQR range. "
                    "This flags unusual values for review; it does not mean they are errors."
                )

            if pd.api.types.is_integer_dtype(selected_series) and unique_count <= 20:
                st.info(
                    "This is an integer column with few different values. "
                    "It may represent categories or codes rather than a continuous measurement."
                )

        elif not pd.api.types.is_bool_dtype(selected_series) and not non_missing.empty:
            st.write("Category details")
            rare_count = int((non_missing_counts / len(non_missing) < 0.01).sum())
            st.write(
                f"Categories appearing in under 1% of non-missing rows: **{rare_count}**"
            )

            normalized_labels = (
                non_missing.astype("string")
                .str.strip()
                .str.casefold()
            )
            label_variants = []
            raw_labels = non_missing.astype("string")
            for _, group in raw_labels.groupby(normalized_labels):
                distinct_labels = group.dropna().drop_duplicates()
                if len(distinct_labels) > 1:
                    label_variants.append(
                        {
                            "Possible formatting variants": ", ".join(
                                distinct_labels.astype(str).head(5)
                            ),
                            "Variant count": len(distinct_labels),
                        }
                    )
            if label_variants:
                st.write("Possible case or whitespace differences")
                st.dataframe(
                    pd.DataFrame(label_variants).head(10),
                    width="stretch",
                    hide_index=True,
                )
                st.caption(
                    "These values differ only by letter case or surrounding spaces. "
                    "Similar spellings and synonyms still need human review."
                )

        st.write("Distribution")
        if non_missing.empty:
            st.info("No non-missing values to chart.")
        elif suggested_role == "Possible identifier or code":
            st.info(
                "This may be an identifier or code, so a distribution chart may not be useful."
            )
        elif (
            pd.api.types.is_numeric_dtype(selected_series)
            and not pd.api.types.is_bool_dtype(selected_series)
            and suggested_role == "Numeric"
        ):
            chart_data = non_missing.to_frame(name="Value")
            figure = px.histogram(
                chart_data,
                x="Value",
                nbins=30,
                title=f"Distribution of {selected_column}",
            )
            st.plotly_chart(figure, width="stretch")
        else:
            chart_counts = non_missing.astype("string").value_counts().head(15)
            chart_data = chart_counts.rename_axis("Value").reset_index(name="Count")
            figure = px.bar(
                chart_data,
                x="Count",
                y="Value",
                orientation="h",
                title=f"Most common values in {selected_column}",
            )
            st.plotly_chart(figure, width="stretch")

