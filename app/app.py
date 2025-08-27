# app.py
import streamlit as st
import pandas as pd
import plotly.express as px

from models import LeavePredictor

# ========== Load Data ==========
@st.cache_data
def load_data():
    df = pd.read_csv("data/Employee.csv")
        # AgeBand
    bins = [18,25,30,35,40,45,50,60,100]
    labels = ['18-24','25-29','30-34','35-39','40-44','45-49','50-59','60+']
    df['AgeBand'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    df = df.drop_duplicates()
    return df


df = load_data()

# ========== Streamlit Layout ==========
st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")

st.title("📊 Employee Attrition Dashboard")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔑 KPIs", "📈 Attrition by Factors", "💡 Insights", "📂 Raw Data", "🤖Predictions"])

# ========== TAB 1: KPIs ==========
with tab1:
    st.subheader("Key Metrics")

    total_employees = len(df)
    left_employees = df[df["LeaveOrNot"] == 1].shape[0]
    stayed_employees = df[df["LeaveOrNot"] == 0].shape[0]
    attrition_rate = round((left_employees / total_employees) * 100, 2)

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("👥 Total Employees", total_employees)
    kpi2.metric("🚪 Employees Left", left_employees)
    kpi3.metric("📉 Attrition Rate", f"{attrition_rate}%")


    st.write("Key Attrition Chart")
    categorical_features = df.columns

    for col in categorical_features:
        if col != 'LeaveOrNot':
            # Compute counts + percentages
            temp = (
                df.groupby([col, "LeaveOrNot"])
                .size()
                .reset_index(name="Count")
            )
            temp["Percentage"] = temp.groupby(col)["Count"].transform(lambda x: 100 * x / x.sum())
            
            # Map attrition labels to more readable form (optional)
            temp["LeaveOrNot"] = temp["LeaveOrNot"].map({0: "Stayed", 1: "Left"})
            
            # Plot with custom colors
            fig = px.bar(
                temp,
                x=col,
                y="Percentage",
                color="LeaveOrNot",
                barmode="group",
                text="Percentage",
                title=f"Attrition Percentage by {col}",
                color_discrete_map={"Stayed": "green", "Left": "red"}  # custom colors
            )
            
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(xaxis={'categoryorder': 'total descending'}, yaxis_title="Percentage (%)")
            fig.show()
            st.plotly_chart(fig, use_container_width=True)



# ========== TAB 2: ATTRITION BY FACTORS ==========
with tab2:
    st.subheader("Attrition by Factors")

    # User selects variables
    var1 = st.selectbox("Select First Variable (x-axis)", ["AgeBand", "Gender", "Education", "City", "PaymentTier", "JoiningYear", "EverBenched"])
    var2 = st.selectbox("Select Second Variable (facet)", ["Gender", "Education", "City", "PaymentTier", "JoiningYear", "ExperienceInCurrentDomain"])

    def plot_leave_percentage(df, var1, var2, target="LeaveOrNot"):
        grouped = (
            df.groupby([var1, var2, target])
            .size()
            .reset_index(name="count")
        )
        grouped["percent"] = grouped.groupby([var1, var2])["count"].transform(lambda x: x / x.sum() * 100)

        fig = px.bar(
            grouped,
            x=var1,
            y="percent",
            color=target,
            barmode="group",
            facet_col=var2,
            text=grouped["percent"].round(1).astype(str) + '%',
            title=f"Attrition % by {var1} × {var2}",
            color_discrete_map={"Leave": "red", "Stay": "green"}  # ✅ set custom colors
        )
        fig.update_layout(yaxis_title="Percentage (%)")
        st.plotly_chart(fig, use_container_width=True)

    # Example usage
    plot_leave_percentage(df, var1=var1, var2=var2)


# ========== TAB 3: INSIGHTS ==========
with tab3:
    st.subheader("Insights & Highlights")
    import streamlit as st
import pandas as pd
import os

COMMENTS_FILE = "insights.csv"

with tab3:
    st.subheader("Insights & Highlights")

    # Load existing comments from CSV
    if os.path.exists(COMMENTS_FILE):
        df_comments = pd.read_csv(COMMENTS_FILE)
    else:
        df_comments = pd.DataFrame(columns=["name", "comment"])

    # Input form
    with st.form(key="insight_form"):
        name = st.text_input("Your Name")
        comment = st.text_area("Write your insight/comment here")
        submit = st.form_submit_button("Post Insight")

        if submit:
            if name.strip() == "" or comment.strip() == "":
                st.warning("Please enter both your name and a comment.")
            else:
                # Append new comment to DataFrame
                df_comments = pd.concat(
                    [df_comments, pd.DataFrame([{"name": name, "comment": comment}])],
                    ignore_index=True
                )
                # Save to CSV
                df_comments.to_csv(COMMENTS_FILE, index=False)
                st.success("Insight posted!")

    # Display all comments
    st.subheader("All Insights / Comments")
    if not df_comments.empty:
        for i, row in df_comments[::-1].iterrows():  # newest first
            st.markdown(f"**{row['name']}** says:")
            st.write(row["comment"])
            st.markdown("---")
    else:
        st.info("No insights yet. Be the first to post!")

# ========== TAB 4: RAW DATA ==========
with tab4:
    st.subheader("Employee Data")
    st.dataframe(df)


# -----------------------------------------------
with tab5:
    predictor = LeavePredictor("models/lgbm_model.pkl")
    predictor.render()