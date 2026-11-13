
import streamlit as st
import pandas as pd

st.set_page_config(page_title="InsiderWatch Test")

st.title("InsiderWatch AI - PyArrow Test")

data = pd.DataFrame({
    "Employee ID": ["EMP001", "EMP002", "EMP003"],
    "Activities": [20, 35, 18],
    "Status": ["Normal", "Suspicious", "Normal"]
})

st.metric("Total Employees", len(data))
st.metric("Total Activities", int(data["Activities"].sum()))

st.subheader("Employee Information")

for _, employee in data.iterrows():
    st.write(
        f"{employee['Employee ID']} | "
        f"Activities: {employee['Activities']} | "
        f"Status: {employee['Status']}"
    )

st.success("Basic Streamlit components are working!")
