import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Page title
st.markdown('''
# **EDA App **

This is a simplified Exploratory Data Analysis (EDA) app built using **Streamlit**.


---
''')

# File uploader
with st.sidebar.header("1. Upload your CSV data"):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

df = None  # Initialize dataframe

# Load data
if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    st.info("Awaiting CSV upload. Or press the button below to use example data.")
    if st.button("Use Example Dataset"):
        df = pd.DataFrame(np.random.rand(100, 5), columns=list("ABCDE"))

# If df is loaded, show analysis
if df is not None:
    st.subheader("📋 Dataset Preview")
    st.dataframe(df)



    st.subheader("📊 Data Summary")
    st.write(df.describe())

    st.subheader("🧩 Missing Values")
    st.write(df.isnull().sum())
    # 🧹 Handle missing data
    st.subheader("🧹 Handle Missing Data")
    if st.checkbox("Fill missing values"):
        method = st.radio("Choose method", ["Mean", "Median", "Mode (for all)"])
        if method == "Mean":
            for col in df.select_dtypes(include='number').columns:
                df[col].fillna(df[col].mean(), inplace=True)
        elif method == "Median":
            for col in df.select_dtypes(include='number').columns:
                df[col].fillna(df[col].median(), inplace=True)
        elif method == "Mode (for all)":
            for col in df.columns:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col].fillna(mode_val[0], inplace=True)
    elif st.checkbox("Drop rows with missing values"):
        df.dropna(inplace=True)

    st.write("✅ Missing values handled. Here's the updated dataset:")
    st.dataframe(df)

    st.subheader("🔗 Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("📈 Plot Distributions")
    selected_col_dist = st.selectbox("Select a column to plot distribution", df.columns)
    if pd.api.types.is_numeric_dtype(df[selected_col_dist]):
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col_dist], kde=True, ax=ax)
        st.pyplot(fig)
    else:
        st.write("Selected column is non-numeric, skipping histogram.")

    # ✅ Column-Level Exploration
    st.subheader("📌 Column-Level Exploration")
    selected_col = st.selectbox("Select a column to explore", df.columns, key="col_explore")

    st.write(f"**Data Type:** {df[selected_col].dtype}")
    st.write(f"**Unique Values:** {df[selected_col].nunique()}")
    st.write("**Summary Statistics:**")
    st.write(df[selected_col].describe())

    if pd.api.types.is_object_dtype(df[selected_col]) or pd.api.types.is_categorical_dtype(df[selected_col]):
        st.write("**Value Counts:**")
        st.write(df[selected_col].value_counts())
        fig, ax = plt.subplots()
        sns.countplot(y=df[selected_col], order=df[selected_col].value_counts().index, ax=ax)
        st.pyplot(fig)

    elif pd.api.types.is_numeric_dtype(df[selected_col]):
        st.write("**Histogram with KDE:**")
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col], kde=True, ax=ax)
        st.pyplot(fig)

        st.write("**Box Plot (for outlier detection):**")
        fig2, ax2 = plt.subplots()
        sns.boxplot(x=df[selected_col], ax=ax2)
        st.pyplot(fig2)

    else:
        st.warning("Selected column is neither numeric nor categorical.")



    # 📊 Advanced Visualizations
    st.subheader("📊 Advanced Visualizations")

    # Scatter Plot
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) >= 2:
        st.markdown("**Scatter Plot**")
        x_col = st.selectbox("Select X-axis", numeric_cols, key="x_axis")
        y_col = st.selectbox("Select Y-axis", numeric_cols, key="y_axis")
        fig3, ax3 = plt.subplots()
        sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax3)
        st.pyplot(fig3)

    # Pair Plot
    if st.checkbox("Show Pair Plot for numeric columns"):
        selected_pair_cols = st.multiselect("Select numeric columns for pairplot", numeric_cols, default=numeric_cols[:3])
        if len(selected_pair_cols) >= 2:
            pair_fig = sns.pairplot(df[selected_pair_cols])
            st.pyplot(pair_fig)

    # Grouped Boxplot
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if len(cat_cols) > 0 and len(numeric_cols) > 0:
        st.markdown("**Box Plot grouped by Categorical Column**")
        group_col = st.selectbox("Select Categorical Column", cat_cols)
        value_col = st.selectbox("Select Numeric Column", numeric_cols, key="val_col")
        fig4, ax4 = plt.subplots()
        sns.boxplot(x=group_col, y=value_col, data=df, ax=ax4)
        plt.xticks(rotation=45)
        st.pyplot(fig4)


    # Normal Boxplot for Numeric Column
    if len(numeric_cols) > 0:
        st.markdown("**Boxplot for a Numeric Column**")
        selected_box_col = st.selectbox("Select a numeric column for boxplot", numeric_cols, key="boxplot_col")
        fig6, ax6 = plt.subplots()
        sns.boxplot(x=df[selected_box_col], ax=ax6)
        ax6.set_xlabel(selected_box_col)
        st.pyplot(fig6)
        # Pie Chart for Categorical Column
        if len(cat_cols) > 0:
            st.markdown("**Pie Chart for a Categorical Column**")
            pie_col = st.selectbox("Select a categorical column", cat_cols, key="pie_col")
            pie_data = df[pie_col].value_counts()
            fig5, ax5 = plt.subplots()
            ax5.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
            ax5.axis('equal')
            st.pyplot(fig5)



            # 📁 Data Export
            st.subheader("📁 Export Cleaned Data")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='cleaned_dataset.csv',
                mime='text/csv'
            )




