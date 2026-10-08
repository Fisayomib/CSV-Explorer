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

        st.subheader("Convert measurement units")
        st.caption(
            "Use this when the measurement and its unit are in separate columns, "
            "for example Weight = 25 and Weight Unit = kg. The app creates a new "
            "standardized column and keeps the uploaded data unchanged."
        )

        numeric_columns = [
            column for column in df.columns
            if pd.api.types.is_numeric_dtype(df[column])
            and not pd.api.types.is_bool_dtype(df[column])
        ]
        unit_columns = [
            column for column in df.columns
            if column not in numeric_columns
            and (
                pd.api.types.is_string_dtype(df[column])
                or df[column].dtype == object
            )
        ]

        unit_definitions = {
            "kg": ("mass", 1.0),
            "g": ("mass", 0.001),
            "mg": ("mass", 0.000001),
            "lb": ("mass", 0.45359237),
            "oz": ("mass", 0.028349523125),
            "tonne": ("mass", 1000.0),
            "ml": ("volume", 1.0),
            "l": ("volume", 1000.0),
        }
        unit_aliases = {
            "kg": "kg", "kilogram": "kg", "kilograms": "kg",
            "g": "g", "gram": "g", "grams": "g",
            "mg": "mg", "milligram": "mg", "milligrams": "mg",
            "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb",
            "oz": "oz", "ounce": "oz", "ounces": "oz",
            "tonne": "tonne", "tonnes": "tonne", "ton": "tonne", "tons": "tonne",
            "ml": "ml", "milliliter": "ml", "milliliters": "ml",
            "millilitre": "ml", "millilitres": "ml",
            "l": "l", "liter": "l", "liters": "l", "litre": "l", "litres": "l",
        }

        if not numeric_columns:
            st.info("I couldn't find a numeric measurement column to convert.")
        elif not unit_columns:
            st.info(
                "I couldn't find a text column containing units. "
                "This converter expects the number and unit in separate columns."
            )
        else:
            measurement_column = st.selectbox(
                "Which column contains the measurement?",
                numeric_columns,
                key="unit_measurement_column",
            )
            unit_column = st.selectbox(
                "Which column contains its unit?",
                unit_columns,
                key="unit_label_column",
            )

            raw_units = (
                df[unit_column]
                .dropna()
                .astype("string")
                .str.strip()
            )
            observed_canonical_units = {}
            for raw_unit in raw_units.drop_duplicates():
                canonical = unit_aliases.get(str(raw_unit).casefold())
                if canonical:
                    observed_canonical_units.setdefault(canonical, str(raw_unit))

            if not observed_canonical_units:
                st.info(
                    "I couldn't recognize common mass or volume units in this column yet. "
                    "The column inspection above can show you its values."
                )
            else:
                observed_units = list(observed_canonical_units)
                source_unit = st.selectbox(
                    "Convert values from",
                    observed_units,
                    format_func=lambda unit: f"{observed_canonical_units[unit]} ({unit})",
                    key="unit_source",
                )
                source_dimension = unit_definitions[source_unit][0]
                target_units = [
                    unit for unit, (dimension, _) in unit_definitions.items()
                    if dimension == source_dimension and unit != source_unit
                ]

                if not target_units:
                    st.info("There isn't another supported unit in this measurement group.")
                else:
                    preferred_target = (
                        "lb" if source_dimension == "mass" and source_unit == "kg"
                        else "kg" if source_dimension == "mass"
                        else "l" if source_unit == "ml"
                        else "ml"
                    )
                    target_unit = st.selectbox(
                        "Convert to",
                        target_units,
                        index=target_units.index(preferred_target)
                        if preferred_target in target_units else 0,
                        key="unit_target",
                    )

                    output_column = re.sub(
                        r"\W+",
                        "_",
                        f"{measurement_column}_{target_unit}",
                    ).strip("_")
                    if output_column in df.columns:
                        st.warning(
                            f"The output column {output_column!r} already exists. "
                            "Choose a different measurement column or remove/rename that existing column."
                        )
                    else:
                        normalized_units = (
                            df[unit_column]
                            .astype("string")
                            .str.strip()
                            .str.casefold()
                            .map(unit_aliases)
                        )
                        numeric_values = pd.to_numeric(
                            df[measurement_column],
                            errors="coerce",
                        )
                        source_mask = normalized_units == source_unit
                        target_mask = normalized_units == target_unit
                        source_rows = source_mask & numeric_values.notna()
                        target_rows = target_mask & numeric_values.notna()
                        other_rows = (
                            ~(source_mask | target_mask)
                            & numeric_values.notna()
                        )

                        conversion_factor = (
                            unit_definitions[source_unit][1]
                            / unit_definitions[target_unit][1]
                        )
                        converted_values = pd.Series(
                            np.nan,
                            index=df.index,
                            dtype="float64",
                        )
                        converted_values.loc[source_rows] = (
                            numeric_values.loc[source_rows] * conversion_factor
                        )
                        converted_values.loc[target_rows] = numeric_values.loc[target_rows]

                        prepared_df = df.copy()
                        prepared_df[output_column] = converted_values

                        st.write(
                            f"**{source_unit} → {target_unit}** multiplies the measurement "
                            f"by **{conversion_factor:.8g}**."
                        )
                        converted_metric, kept_metric, other_metric = st.columns(3)
                        converted_metric.metric("Converted rows", f"{int(source_rows.sum()):,}")
                        kept_metric.metric("Already in target unit", f"{int(target_rows.sum()):,}")
                        other_metric.metric("Other or unrecognized units", f"{int(other_rows.sum()):,}")

                        preview = df[[measurement_column, unit_column]].copy()
                        preview[output_column] = converted_values
                        st.write("Check a few results before downloading.")
                        st.dataframe(preview.head(20), width="stretch")

                        file_stem = uploaded_file.name.rsplit(".", 1)[0]
                        st.download_button(
                            "Download converted CSV",
                            data=prepared_df.to_csv(index=False).encode("utf-8"),
                            file_name=f"{file_stem}_converted.csv",
                            mime="text/csv",
                            key="download_unit_conversion",
                        )
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

            all_value_counts = non_missing_counts.rename_axis("Value").reset_index(name="Count")
            all_value_counts["Percent of non-missing"] = (
                all_value_counts["Count"] / len(non_missing) * 100
            ).round(1)
            with st.expander(f"See all {unique_count:,} distinct values"):
                st.caption(
                    "This table lists every distinct non-missing value and how often it appears. "
                    "It is especially useful for checking unit or category columns."
                )
                st.dataframe(
                    all_value_counts,
                    width="stretch",
                    hide_index=True,
                    height=350,
                )

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

