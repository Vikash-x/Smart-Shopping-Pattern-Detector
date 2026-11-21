
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(
    page_title="Smart Shopping Pattern Detector",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Smart Shopping Pattern Detector")
st.write("Discover shopping patterns using Market Basket Analysis and Apriori Algorithm.")

@st.cache_data
def load_data():
    data = pd.read_csv("Groceries_dataset.csv")
    data = data.dropna()
    data["itemDescription"] = (
        data["itemDescription"].astype(str).str.strip().str.lower()
    )
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)
    transactions = (
        data.groupby(["Member_number", "Date"])["itemDescription"]
        .apply(lambda x: sorted(set(x)))
        .tolist()
    )
    return data, transactions

try:
    data, transactions = load_data()

    st.sidebar.header("Analysis Settings")
    min_support = st.sidebar.slider(
        "Minimum Support", 0.001, 0.02, 0.002, 0.001
    )
    min_confidence = st.sidebar.slider(
        "Minimum Confidence", 0.01, 0.50, 0.05, 0.01
    )

    encoder = TransactionEncoder()
    encoded = encoder.fit(transactions).transform(transactions)
    basket_df = pd.DataFrame(encoded, columns=encoder.columns_)

    frequent_items = apriori(
        basket_df,
        min_support=min_support,
        use_colnames=True,
        max_len=3
    )

    if frequent_items.empty:
        st.warning("No frequent itemsets found. Please reduce Minimum Support.")
        st.stop()

    rules = association_rules(
        frequent_items,
        metric="confidence",
        min_threshold=min_confidence
    )

    st.subheader("📊 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(transactions):,}")
    c2.metric("Customers", f"{data['Member_number'].nunique():,}")
    c3.metric("Products", f"{data['itemDescription'].nunique():,}")
    c4.metric("Association Rules", f"{len(rules):,}")

    st.divider()

    st.subheader("🏆 Top 10 Most Purchased Products")
    top_products = data["itemDescription"].value_counts().head(10)

    fig, ax = plt.subplots(figsize=(10, 5))
    top_products.sort_values().plot(kind="barh", ax=ax)
    ax.set_xlabel("Number of Purchases")
    ax.set_ylabel("Product")
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.subheader("🔍 Product Recommendation")
    products = sorted(data["itemDescription"].unique())
    selected_product = st.selectbox("Select a product", products)

    if not rules.empty:
        recommendations = rules[
            rules["antecedents"].apply(
                lambda x: len(x) == 1 and selected_product in x
            )
        ].copy()

        recommendations = recommendations[
            recommendations["lift"] > 1
        ].sort_values(
            ["lift", "confidence"], ascending=False
        )

        if not recommendations.empty:
            recommendations["Recommended Product"] = (
                recommendations["consequents"].apply(
                    lambda x: ", ".join(sorted(x))
                )
            )
            recommendations["Confidence (%)"] = (
                recommendations["confidence"] * 100
            ).round(2)
            st.success("Recommended products based on shopping patterns:")
            st.dataframe(
                recommendations[
                    ["Recommended Product", "Confidence (%)", "lift", "support"]
                ].head(10),
                use_container_width=True
            )
        else:
            st.info("No strong recommendation found for this product. Try another product.")
    else:
        st.info("No association rules found. Reduce Minimum Confidence.")

    st.divider()

    st.subheader("📋 Association Rules")
    if not rules.empty:
        display_rules = rules.copy()
        display_rules["If Customer Buys"] = display_rules["antecedents"].apply(
            lambda x: ", ".join(sorted(x))
        )
        display_rules["Then May Buy"] = display_rules["consequents"].apply(
            lambda x: ", ".join(sorted(x))
        )
        st.dataframe(
            display_rules[
                ["If Customer Buys", "Then May Buy", "support", "confidence", "lift"]
            ].sort_values("lift", ascending=False).head(50),
            use_container_width=True
        )
    else:
        st.info("No rules available with the current settings.")

    st.caption("Project: Smart Shopping Pattern Detector | Algorithm: Apriori")

except Exception as e:
    st.error(f"Something went wrong: {e}")
    st.info("Please check that Groceries_dataset.csv is uploaded in the repository.")
