import streamlit as st
import pandas as pd
import re
import plotly.express as px
import numpy as np
import ast
import operator

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




def evaluate_formula(expression, dataframe):
    """Evaluate a small, safe expression language over dataframe columns."""
    tree = ast.parse(expression, mode="eval")

    binary_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.BitAnd: operator.and_,
        ast.BitOr: operator.or_,
        ast.BitXor: operator.xor,
    }
    comparison_operators = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, (ast.List, ast.Tuple)):
            return [evaluate(item) for item in node.elts]

        if isinstance(node, ast.BinOp) and type(node.op) in binary_operators:
            return binary_operators[type(node.op)](
                evaluate(node.left),
                evaluate(node.right),
            )

        if isinstance(node, ast.UnaryOp):
            value = evaluate(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return +value
            if isinstance(node.op, ast.Invert):
                return ~value
            if isinstance(node.op, ast.Not):
                return ~value if isinstance(value, pd.Series) else not value

        if isinstance(node, ast.BoolOp):
            values = [evaluate(value) for value in node.values]
            operation = operator.and_ if isinstance(node.op, ast.And) else operator.or_
            result = values[0]
            for value in values[1:]:
                result = operation(result, value)
            return result

        if isinstance(node, ast.Compare):
            left = evaluate(node.left)
            result = None
            for comparison, comparator in zip(node.ops, node.comparators):
                right = evaluate(comparator)
                if type(comparison) in comparison_operators:
                    current = comparison_operators[type(comparison)](left, right)
                elif isinstance(comparison, (ast.In, ast.NotIn)):
                    current = left.isin(right) if isinstance(left, pd.Series) else left in right
                    if isinstance(comparison, ast.NotIn):
                        current = ~current if isinstance(current, pd.Series) else not current
                else:
                    raise ValueError("That comparison is not supported.")
                result = current if result is None else operator.and_(result, current)
                left = right
            return result

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.keywords:
                raise ValueError("Use positional arguments in formulas.")
            function = node.func.id
            arguments = [evaluate(argument) for argument in node.args]

            if function == "col" and len(arguments) == 1 and isinstance(arguments[0], str):
                column = arguments[0]
                if column not in dataframe.columns:
                    raise ValueError(f"Column {column!r} was not found.")
                return dataframe[column]

            if function == "where" and len(arguments) == 3:
                condition = arguments[0]
                if isinstance(condition, pd.Series):
                    condition = condition.fillna(False)
                return np.where(condition, arguments[1], arguments[2])

            if function == "contains" and len(arguments) == 2:
                return (
                    arguments[0]
                    .astype("string")
                    .str.contains(arguments[1], case=False, na=False, regex=True)
                )

            if function == "to_number" and len(arguments) == 1:
                return pd.to_numeric(arguments[0], errors="coerce")

            if function == "is_missing" and len(arguments) == 1:
                return arguments[0].isna() if isinstance(arguments[0], pd.Series) else pd.isna(arguments[0])

            if function == "is_in" and len(arguments) == 2:
                return arguments[0].isin(arguments[1])

            if function == "fill_missing" and len(arguments) == 2:
                return arguments[0].fillna(arguments[1])

            if function == "lower" and len(arguments) == 1:
                return arguments[0].astype("string").str.lower()

            if function == "strip" and len(arguments) == 1:
                return arguments[0].astype("string").str.strip()

            if function == "abs" and len(arguments) == 1:
                return abs(arguments[0])

            if function == "round" and len(arguments) in (1, 2):
                digits = arguments[1] if len(arguments) == 2 else 0
                return arguments[0].round(digits) if isinstance(arguments[0], pd.Series) else round(arguments[0], digits)

            raise ValueError(
                f"Function {function!r} is not supported. "
                "See the examples for available formula functions."
            )

        raise ValueError(
            "This part of the formula isn't supported. Use column references like col(\"Sales\") "
            "and the listed functions."
        )

    return evaluate(tree.body)
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
st.set_page_config(page_title="CSV Explorer", page_icon="📊", layout="wide")

        st.markdown("""<style>
:root {
  --canvas: #f6f3ed;
  --surface: #fffefa;
  --ink: #263746;
  --heading: #24425c;
  --muted: #687986;
  --line: #e5e0d7;
  --teal: #668b7a;
  --blue-wash: #eaf1f5;
  --amber-wash: #fbf2df;
}
.stApp, [data-testid="stAppViewContainer"] {
  background: var(--canvas);
  color: var(--ink);
}
[data-testid="stHeader"] { background: rgba(246, 243, 237, 0.92); }
.block-container {
  max-width: 1260px;
  padding-top: 2.2rem;
  padding-bottom: 4rem;
}
h1, h2, h3 {
  color: var(--heading) !important;
  letter-spacing: -0.025em;
}
p, label, [data-testid="stMarkdownContainer"] {
  color: var(--ink);
}
[data-testid="stCaptionContainer"] p {
  color: var(--muted) !important;
}
[data-testid="stMetric"] {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 1rem 1.1rem;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; }
[data-testid="stMetricValue"] { color: var(--heading) !important; }
.stTabs [data-baseweb="tab-list"] {
  gap: 0.35rem;
  border-bottom: 1px solid var(--line);
}
.stTabs [data-baseweb="tab"] {
  color: var(--muted);
  padding: 0.8rem 1.1rem;
  background: transparent;
}
.stTabs [aria-selected="true"] {
  color: var(--heading) !important;
  font-weight: 650;
}
.stTabs [data-baseweb="tab-highlight"] { background: var(--teal); }
[data-testid="stDataFrame"] {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 12px;
  overflow: hidden;
}
[data-testid="stExpander"] {
  background: rgba(255, 254, 250, 0.78);
  border: 1px solid var(--line);
  border-radius: 12px;
}
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
  background: var(--surface);
  border-color: var(--line);
  color: var(--ink);
  border-radius: 10px;
}
[data-testid="stAlert"] { border-radius: 12px; }
button[kind="primary"] {
  background: var(--teal);
  border-color: var(--teal);
}
</style>""", unsafe_allow_html=True)
        st.title("CSV Explorer")
        st.write("A calmer place to understand your data before you use it.")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
            except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
                st.error(f"Could not read this CSV: {error}")
            else:
            st.caption(f"Loaded file: {uploaded_file.name}")
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


                row_metric, column_metric, missing_metric, duplicate_metric = st.columns(4)
                row_metric.metric("Rows", f"{df.shape[0]:,}")
                column_metric.metric("Columns", f"{df.shape[1]:,}")
                missing_metric.metric("Missing cells", f"{missing_cells:,}")
                duplicate_metric.metric("Duplicate rows", f"{duplicate_rows:,}")

                overview_tab, investigate_tab, prepare_tab = st.tabs(
                    ["Overview", "Investigate", "Prepare"]
                )

                with overview_tab:
                    st.subheader("Things to review")
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


                    st.divider()
                    st.subheader("Data preview")
                    st.caption("A quick look at the first rows in your file.")
                    st.dataframe(df.head(10), width="stretch")
                    st.divider()
                    st.subheader("Column overview")
                    st.caption("Compare missing values, storage types, and distinct values across columns.")
                    st.dataframe(missing_summary, width="stretch", hide_index=True)

                with investigate_tab:
                    st.subheader("Investigate the data")
                    st.caption("Choose a specific issue or column to understand what is in the dataset.")
                    rows_with_missing_mask = df.isna().any(axis=1)
                    rows_with_missing = int(rows_with_missing_mask.sum())
                    empty_rows = int(df.isna().all(axis=1).sum())
                    st.write("Missing values by column")
                    missing_columns = missing_summary.loc[
                        missing_summary["Missing values"] > 0
                    ].sort_values("Missing values", ascending=False)
                    if missing_columns.empty:
                        st.success("No missing values were found.")
                    else:
                        st.caption("Columns with the most missing values appear first.")
                        st.dataframe(
                            missing_columns[
                                ["Column", "Missing values", "Missing %", "Pandas dtype"]
                            ],
                            width="stretch",
                            hide_index=True,
                        )
                    st.caption(
                        f"Rows with any missing value: {rows_with_missing:,} · "
                        f"Fully empty rows: {empty_rows:,}"
                    )
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


                with prepare_tab:
                    st.subheader("Prepare a copy of your data")
                    st.caption(
                        "Create a new column with a formula, check the result, "
                        "then download a prepared copy. Your uploaded file stays unchanged."
                    )
            st.subheader("Create a column with a formula")
            st.caption(
                "Write a formula using any columns in your dataset. The preview shows the new result; "
                "your uploaded data stays unchanged."
            )

            with st.expander("Formula examples"):
                st.write("Convert kilograms to pounds, while keeping existing pound values unchanged:")
                st.code(
                    'where(col("Weight Unit") == "kg", '
                    'col("Weight") * 2.20462, col("Weight"))'
                )
                st.write("Calculate price per item:")
                st.code('col("Sales") / col("Quantity")')
                st.write("Standardize text labels:")
                st.code('lower(strip(col("City")))')
                st.write("Create a category from a numeric value:")
                st.code('where(col("Age") < 18, "minor", "adult")')
                st.caption(
                    "For a unit conversion, the formula must account for the unit in each row. "
                    "The example assumes the only units are kg and lb."
                )

            formula = st.text_area(
                "Formula",
                placeholder='For example: col("Sales") / col("Quantity")',
                key="derived_formula",
                help='Use col("Exact column name") to refer to a column, including names with spaces.',
            )
            output_column = st.text_input(
                "New column name",
                value="derived_value",
                key="formula_output_column",
            )

            if formula.strip():
                if not output_column.strip():
                    st.info("Enter a name for the new column.")
                elif output_column in df.columns:
                    st.warning("Choose a new column name that does not already exist.")
                else:
                    try:
                        formula_result = evaluate_formula(formula, df)
                        if isinstance(formula_result, pd.Series):
                            output_values = formula_result.reindex(df.index)
                        elif np.isscalar(formula_result):
                            output_values = pd.Series(
                                [formula_result] * len(df),
                                index=df.index,
                            )
                        else:
                            result_array = np.asarray(formula_result)
                            if len(result_array) != len(df):
                                raise ValueError(
                                    "The formula result must produce one value per row."
                                )
                            output_values = pd.Series(result_array, index=df.index)

                        referenced_columns = list(dict.fromkeys(
                            re.findall(
                                r"col\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
                                formula,
                            )
                        ))
                        missing_references = [
                            column for column in referenced_columns
                            if column not in df.columns
                        ]
                        if missing_references:
                            raise ValueError(
                                f"Column {missing_references[0]!r} was not found."
                            )

                        prepared_df = df.copy()
                        prepared_df[output_column] = output_values
                        preview_columns = [
                            column for column in referenced_columns
                            if column in df.columns
                        ]
                        preview = df[preview_columns].copy() if preview_columns else pd.DataFrame(index=df.index)
                        preview[output_column] = output_values

                        result_count = int(output_values.notna().sum())
                        blank_count = int(output_values.isna().sum())
                        result_metric, blank_metric = st.columns(2)
                        result_metric.metric("Rows with a result", f"{result_count:,}")
                        blank_metric.metric("Blank results", f"{blank_count:,}")

                        st.write("Preview the inputs and new column:")
                        st.dataframe(preview.head(20), width="stretch")

                        file_stem = uploaded_file.name.rsplit(".", 1)[0]
                        st.download_button(
                            "Download prepared CSV",
                            data=prepared_df.to_csv(index=False).encode("utf-8"),
                            file_name=f"{file_stem}_prepared.csv",
                            mime="text/csv",
                            key="download_formula_csv",
                        )
                    except (
                        SyntaxError,
                        ValueError,
                        TypeError,
                        ZeroDivisionError,
                        AttributeError,
                        re.error,
                    ) as error:
                        st.error(
                            f"Couldn't apply that formula: {error}. "
                            "Check the column names and formula examples above."
                        )
            else:
                st.info("Enter a formula to preview a new column.")

