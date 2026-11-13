
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from datetime import datetime
from html import escape

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent

# Supports both dashboard/app.py and project-root/app.py
ROOT = APP_DIR.parent if APP_DIR.name.lower() == "dashboard" else APP_DIR

DATA_FILE = ROOT / "data" / "activity_logs.csv"
ALERT_FILE = ROOT / "reports" / "security_alerts.csv"
PREDICTION_FILE = ROOT / "reports" / "model_predictions.csv"

st.set_page_config(
    page_title="InsiderWatch AI | SOC Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. PROFESSIONAL SOC DESIGN
# ============================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}

.stApp {
    background: #080d18;
    color: #e5edf7;
}

[data-testid="stHeader"] {
    background: #080d18;
}

[data-testid="stSidebar"] {
    background: #0d1525;
    border-right: 1px solid #203149;
}

[data-testid="stSidebar"] * {
    color: #d8e4f2;
}

h1, h2, h3, h4 {
    color: #f0f6ff !important;
    font-weight: 700;
}

p, label {
    color: #b7c6d9;
}

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #111e32, #101827);
    border: 1px solid #233751;
    border-radius: 14px;
    padding: 18px;
}

div[data-testid="stMetricLabel"] {
    color: #8fa7c3 !important;
}

div[data-testid="stMetricValue"] {
    color: #f3f8ff !important;
    font-weight: 800;
}

.stButton > button,
.stDownloadButton > button {
    background: #102b45;
    color: #65d9f5;
    border: 1px solid #236083;
    border-radius: 9px;
    font-weight: 600;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #17415f;
    color: white;
    border-color: #43c9ed;
}

.soc-banner {
    background: linear-gradient(110deg, #10233a, #102b3b, #102039);
    border: 1px solid #24506b;
    padding: 26px;
    border-radius: 16px;
    margin-bottom: 24px;
}

.soc-eyebrow {
    color: #53d5ef;
    font-size: 12px;
    letter-spacing: 2px;
    font-weight: 800;
}

.soc-title {
    color: #f3f8ff;
    font-size: 32px;
    font-weight: 800;
    margin-top: 8px;
}

.soc-subtitle {
    color: #9bb4cd;
    font-size: 14px;
    margin-top: 6px;
}

.section-label {
    color: #56d4ee;
    font-size: 12px;
    letter-spacing: 1.5px;
    font-weight: 800;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.table-wrapper {
    max-height: 420px;
    overflow: auto;
    border: 1px solid #233751;
    border-radius: 12px;
    background: #101827;
    margin-top: 8px;
}

.insider-table {
    width: 100%;
    border-collapse: collapse;
    color: #dce8f5;
    font-size: 13px;
}

.insider-table th {
    background: #172b42;
    color: #54d5ef;
    padding: 13px;
    text-align: left;
    position: sticky;
    top: 0;
    white-space: nowrap;
}

.insider-table td {
    padding: 12px;
    border-bottom: 1px solid #233751;
    white-space: nowrap;
}

.insider-table tr:hover {
    background: #192a40;
}

.status-active {
    color: #36d6a6;
    font-weight: 700;
}

.footer {
    text-align: center;
    padding: 20px 0;
    color: #8195ad;
    font-size: 12px;
}

hr {
    border-color: #203149;
}
</style>
"""

# Use st.html for custom HTML/CSS instead of displaying HTML as Markdown.
st.html(CSS)

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def load_csv(path):
    """Load a CSV file safely."""

    if not path.exists():
        return pd.DataFrame()

    try:
        data = pd.read_csv(path)
        data.columns = data.columns.astype(str).str.strip()
        return data

    except Exception as error:
        st.warning(f"Unable to read {path.name}: {error}")
        return pd.DataFrame()


def show_table(data, height=350):
    """
    Render HTML tables without st.dataframe() or st.table().
    This avoids Streamlit's PyArrow table serialization.
    """

    if data is None or data.empty:
        st.info("No records available.")
        return

    headers = "".join(
        f"<th>{escape(str(column))}</th>"
        for column in data.columns
    )

    rows = []

    for row in data.itertuples(index=False, name=None):
        cells = []

        for value in row:
            try:
                if pd.isna(value):
                    value = ""
            except (TypeError, ValueError):
                pass

            cells.append(
                f"<td>{escape(str(value))}</td>"
            )

        rows.append(
            "<tr>" + "".join(cells) + "</tr>"
        )

    table_html = f"""
    <div class="table-wrapper" style="max-height:{int(height)}px;">
        <table class="insider-table">
            <thead>
                <tr>{headers}</tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
    </div>
    """

    st.html(table_html)


def section_heading(title, subtitle=None):
    """Display a styled section heading."""

    st.html(
        f'<div class="section-label">{escape(str(title))}</div>'
    )

    if subtitle:
        st.caption(subtitle)


def style_chart(fig, height=360):
    """Apply consistent SOC styling to Plotly charts."""

    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter",
            color="#c5d5e8",
            size=12
        ),
        margin=dict(
            l=25,
            r=25,
            t=45,
            b=30
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(
            gridcolor="#1c2b40",
            zerolinecolor="#1c2b40"
        ),
        yaxis=dict(
            gridcolor="#1c2b40",
            zerolinecolor="#1c2b40"
        )
    )

    return fig


def severity_counts(data):
    """Count alerts by severity."""

    if data.empty or "severity" not in data.columns:
        return pd.DataFrame(
            columns=["severity", "count"]
        )

    return (
        data["severity"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
        .value_counts()
        .rename_axis("severity")
        .reset_index(name="count")
    )


def csv_download(data, filename, label):
    """Provide a CSV download button."""

    if data is None or data.empty:
        st.info("No data available for download.")
        return

    st.download_button(
        label=label,
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=f"download_{filename}"
    )


def clean_employee_id(series):
    return series.fillna("").astype(str).str.strip()


def empty_chart_message(message):
    st.info(message)


def safe_int(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


# ============================================================
# 4. LOAD ACTIVITY DATASET
# ============================================================

df = load_csv(DATA_FILE)

if df.empty:
    st.error("Activity dataset is missing or empty.")

    st.write("Expected dataset location:")
    st.code(str(DATA_FILE))

    st.info(
        "Check that activity_logs.csv exists inside the data folder."
    )

    st.stop()


required_columns = [
    "employee_id",
    "label",
    "login_hour",
    "files_accessed",
    "files_downloaded"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        f"Missing required activity columns: {missing_columns}"
    )

    st.write("Available columns:")
    st.write(", ".join(df.columns.astype(str)))

    st.stop()


# Clean employee IDs.
df["employee_id"] = clean_employee_id(
    df["employee_id"]
)

# Convert numeric fields.
numeric_columns = [
    "label",
    "login_hour",
    "files_accessed",
    "files_downloaded"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# Remove records without employee ID or label.
df = df[
    (df["employee_id"] != "")
    & df["label"].notna()
].copy()

if df.empty:
    st.error("No valid activity records found.")
    st.stop()

# Fill missing measurements.
for column in [
    "login_hour",
    "files_accessed",
    "files_downloaded"
]:
    df[column] = df[column].fillna(0)

df["label"] = df["label"].astype(int)

df["activity_status"] = df["label"].map({
    0: "Normal",
    1: "Suspicious"
}).fillna("Other")


# ============================================================
# 5. LOAD SECURITY ALERTS
# ============================================================

alerts = load_csv(ALERT_FILE)

# Use model predictions if security alerts are unavailable.
if alerts.empty:
    alerts = load_csv(PREDICTION_FILE)

if not alerts.empty:

    if "employee_id" in alerts.columns:
        alerts["employee_id"] = clean_employee_id(
            alerts["employee_id"]
        )

    if "severity" in alerts.columns:
        alerts["severity"] = (
            alerts["severity"]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
            .str.upper()
        )

    if "risk_score" in alerts.columns:
        alerts["risk_score"] = pd.to_numeric(
            alerts["risk_score"],
            errors="coerce"
        )


# ============================================================
# 6. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.html(
        """
        <div style="padding:5px 0 10px 0;">
            <div style="font-size:23px;font-weight:800;color:#54d5ef;">
                🛡️ InsiderWatch AI
            </div>
            <div style="font-size:10px;letter-spacing:1.5px;color:#8fa7c3;">
                SECURITY INTELLIGENCE PLATFORM
            </div>
        </div>
        """
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Security Overview",
            "Employee Intelligence",
            "Threat Detection",
            "Investigation Centre",
            "Dataset Explorer"
        ],
        key="main_navigation"
    )

    st.divider()

    st.markdown("### SYSTEM STATUS")

    st.html(
        '<span class="status-active">● Dashboard Active</span>'
    )

    st.caption(
        "Last refresh: "
        + datetime.now().strftime("%d %b %Y | %H:%M:%S")
    )

    st.divider()

    st.caption("INSIDERWATCH AI")
    st.caption("Academic Cybersecurity Project")
    st.caption("Version 4.0")


# ============================================================
# 7. MAIN HEADER
# ============================================================

st.html(
    """
    <div class="soc-banner">
        <div class="soc-eyebrow">
            SECURITY OPERATIONS CENTRE
        </div>

        <div class="soc-title">
            INSIDERWATCH AI
        </div>

        <div class="soc-subtitle">
            Intelligent Insider Threat Monitoring & Behaviour Analytics
        </div>
    </div>
    """
)


# ============================================================
# 8. SECURITY OVERVIEW
# ============================================================

if page == "Security Overview":

    section_heading(
        "Executive Security Overview",
        "A consolidated view of monitored activities and detected security signals."
    )

    total_activities = len(df)
    normal_count = int((df["label"] == 0).sum())
    suspicious_count = int((df["label"] == 1).sum())
    employee_count = int(df["employee_id"].nunique())

    high_alerts = 0

    if not alerts.empty and "severity" in alerts.columns:
        high_alerts = int(
            (alerts["severity"] == "HIGH").sum()
        )

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "TOTAL ACTIVITIES",
        f"{total_activities:,}"
    )

    k2.metric(
        "MONITORED EMPLOYEES",
        f"{employee_count:,}"
    )

    k3.metric(
        "SUSPICIOUS LABELS",
        f"{suspicious_count:,}"
    )

    k4.metric(
        "HIGH SEVERITY ALERTS",
        f"{high_alerts:,}"
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        section_heading("Activity Classification")

        classification = (
            df["activity_status"]
            .value_counts()
            .rename_axis("Activity")
            .reset_index(name="Count")
        )

        fig = px.pie(
            classification,
            names="Activity",
            values="Count",
            hole=0.65,
            color="Activity",
            color_discrete_map={
                "Normal": "#20c997",
                "Suspicious": "#ff5c70",
                "Other": "#8294a9"
            }
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label"
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    with right:

        section_heading("Login Behaviour")

        login_data = (
            df.groupby("login_hour")
            .size()
            .reset_index(name="Activity Count")
            .sort_values("login_hour")
        )

        fig = px.bar(
            login_data,
            x="login_hour",
            y="Activity Count",
            labels={
                "login_hour": "Login Hour",
                "Activity Count": "Activities"
            }
        )

        fig.update_traces(
            marker_color="#27b8dc"
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    st.divider()

    section_heading(
        "Data Access Behaviour",
        "Compare file downloads across activity classifications."
    )

    fig = px.box(
        df,
        x="activity_status",
        y="files_downloaded",
        color="activity_status",
        color_discrete_map={
            "Normal": "#20c997",
            "Suspicious": "#ff5c70",
            "Other": "#8294a9"
        },
        labels={
            "activity_status": "Activity Classification",
            "files_downloaded": "Files Downloaded"
        }
    )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True
    )

    st.divider()

    section_heading("Threat Severity Distribution")

    if not alerts.empty and "severity" in alerts.columns:

        severity_data = severity_counts(alerts)

        fig = px.bar(
            severity_data,
            x="severity",
            y="count",
            color="severity",
            color_discrete_map={
                "HIGH": "#ff5269",
                "MEDIUM": "#ffb547",
                "LOW": "#25c997",
                "UNKNOWN": "#8294a9"
            },
            labels={
                "severity": "Severity",
                "count": "Alert Count"
            }
        )

        st.plotly_chart(
            style_chart(fig, 300),
            use_container_width=True
        )

    else:
        empty_chart_message(
            "No severity records are currently available."
        )


# ============================================================
# 9. EMPLOYEE INTELLIGENCE
# ============================================================

elif page == "Employee Intelligence":

    section_heading(
        "Employee Intelligence",
        "Analyse employee activity, file access and suspicious behaviour."
    )

    employee_data = (
        df.groupby("employee_id", as_index=False)
        .agg(
            Total_Activities=("label", "size"),
            Files_Accessed=("files_accessed", "sum"),
            Files_Downloaded=("files_downloaded", "sum"),
            Suspicious_Activities=(
                "label",
                lambda values: int((values == 1).sum())
            )
        )
    )

    employee_data["Suspicious_Rate"] = (
        employee_data["Suspicious_Activities"]
        .div(
            employee_data["Total_Activities"].replace(0, 1)
        )
        .mul(100)
        .round(2)
    )

    employee_data = employee_data.sort_values(
        by="Total_Activities",
        ascending=False
    )

    # KPI CARDS

    a, b, c, d = st.columns(4)

    a.metric(
        "TOTAL EMPLOYEES",
        f"{len(employee_data):,}"
    )

    b.metric(
        "TOTAL ACTIVITIES",
        f"{safe_int(employee_data['Total_Activities'].sum()):,}"
    )

    c.metric(
        "FILE DOWNLOADS",
        f"{safe_int(employee_data['Files_Downloaded'].sum()):,}"
    )

    d.metric(
        "SUSPICIOUS RECORDS",
        f"{safe_int(employee_data['Suspicious_Activities'].sum()):,}"
    )

    st.divider()

    # EMPLOYEE SEARCH

    section_heading("Employee Search")

    search = st.text_input(
        "Search by Employee ID",
        placeholder="Example: EMP001",
        key="employee_search"
    )

    visible_employees = employee_data.copy()

    if search.strip():
        visible_employees = visible_employees[
            visible_employees["employee_id"].str.contains(
                search.strip(),
                case=False,
                na=False,
                regex=False
            )
        ]

    st.caption(
        f"Showing {len(visible_employees)} of "
        f"{len(employee_data)} employees"
    )

    section_heading("Employee Activity Records")

    show_table(
        visible_employees,
        height=400
    )

    st.divider()

    # EMPLOYEE CHARTS

    left, right = st.columns(2)

    with left:

        section_heading("Suspicious Activity Ranking")

        ranking = (
            employee_data
            .sort_values(
                "Suspicious_Activities",
                ascending=False
            )
            .head(10)
        )

        if not ranking.empty:

            fig = px.bar(
                ranking,
                x="Suspicious_Activities",
                y="employee_id",
                orientation="h",
                labels={
                    "employee_id": "Employee",
                    "Suspicious_Activities": "Suspicious Records"
                }
            )

            fig.update_traces(
                marker_color="#ff657a"
            )

            fig.update_layout(
                yaxis={"categoryorder": "total ascending"}
            )

            st.plotly_chart(
                style_chart(fig),
                use_container_width=True
            )

        else:
            st.info("No employee ranking data available.")

    with right:

        section_heading("File Download Ranking")

        ranking = (
            employee_data
            .sort_values(
                "Files_Downloaded",
                ascending=False
            )
            .head(10)
        )

        if not ranking.empty:

            fig = px.bar(
                ranking,
                x="Files_Downloaded",
                y="employee_id",
                orientation="h",
                labels={
                    "employee_id": "Employee",
                    "Files_Downloaded": "Files Downloaded"
                }
            )

            fig.update_traces(
                marker_color="#29b9dc"
            )

            fig.update_layout(
                yaxis={"categoryorder": "total ascending"}
            )

            st.plotly_chart(
                style_chart(fig),
                use_container_width=True
            )

        else:
            st.info("No file download data available.")

    st.divider()

    section_heading("Employee Suspicious Activity Rate")

    rate_ranking = (
        employee_data
        .sort_values(
            "Suspicious_Rate",
            ascending=False
        )
        .head(10)
    )

    if not rate_ranking.empty:

        fig = px.bar(
            rate_ranking,
            x="Suspicious_Rate",
            y="employee_id",
            orientation="h",
            labels={
                "employee_id": "Employee",
                "Suspicious_Rate": "Suspicious Activity (%)"
            }
        )

        fig.update_traces(
            marker_color="#b58cff"
        )

        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            xaxis_title="Suspicious Activity (%)"
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    csv_download(
        employee_data,
        "employee_intelligence_report.csv",
        "⬇ Download Employee Intelligence Report"
    )


# ============================================================
# 10. THREAT DETECTION
# ============================================================

elif page == "Threat Detection":

    section_heading(
        "Threat Detection Centre",
        "Review generated security alerts, risk scores and severity."
    )

    if alerts.empty:

        st.warning(
            "No alert records found. Run your detection engine "
            "to generate security alerts."
        )

    else:

        filtered = alerts.copy()

        # SEVERITY FILTER

        if "severity" in filtered.columns:

            severity_options = sorted(
                filtered["severity"]
                .dropna()
                .unique()
                .tolist()
            )

            selected_severity = st.multiselect(
                "Filter alert severity",
                severity_options,
                default=severity_options,
                key="severity_filter"
            )

            filtered = filtered[
                filtered["severity"].isin(selected_severity)
            ]

        # EMPLOYEE FILTER

        if "employee_id" in filtered.columns:

            employees = sorted(
                filtered["employee_id"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_employees = st.multiselect(
                "Filter employees",
                employees,
                default=employees,
                key="alert_employee_filter"
            )

            filtered = filtered[
                filtered["employee_id"].astype(str)
                .isin(selected_employees)
            ]

        st.divider()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "FILTERED ALERTS",
            len(filtered)
        )

        if "severity" in filtered.columns:

            col2.metric(
                "HIGH SEVERITY",
                int((filtered["severity"] == "HIGH").sum())
            )

            col3.metric(
                "MEDIUM SEVERITY",
                int((filtered["severity"] == "MEDIUM").sum())
            )

        else:

            col2.metric("HIGH SEVERITY", "N/A")
            col3.metric("MEDIUM SEVERITY", "N/A")

        st.divider()

        section_heading("Security Alert Records")

        show_table(
            filtered,
            height=400
        )

        st.divider()

        # RISK SCORE RANKING

        if (
            "risk_score" in filtered.columns
            and "employee_id" in filtered.columns
        ):

            section_heading("Employee Risk Score Ranking")

            risk_data = (
                filtered.dropna(subset=["risk_score"])
                .groupby("employee_id", as_index=False)["risk_score"]
                .max()
                .sort_values(
                    "risk_score",
                    ascending=False
                )
                .head(10)
            )

            if not risk_data.empty:

                fig = px.bar(
                    risk_data,
                    x="risk_score",
                    y="employee_id",
                    orientation="h",
                    color="risk_score",
                    color_continuous_scale="Reds",
                    labels={
                        "employee_id": "Employee",
                        "risk_score": "Risk Score"
                    }
                )

                fig.update_layout(
                    yaxis={"categoryorder": "total ascending"}
                )

                st.plotly_chart(
                    style_chart(fig),
                    use_container_width=True
                )

            else:
                st.info("No valid risk scores available.")

        else:
            st.info("Risk score information is unavailable.")

        st.divider()

        # SEVERITY ANALYSIS

        if "severity" in filtered.columns:

            section_heading("Alert Severity Analysis")

            severity_data = severity_counts(filtered)

            if not severity_data.empty:

                fig = px.pie(
                    severity_data,
                    names="severity",
                    values="count",
                    hole=0.6,
                    color="severity",
                    color_discrete_map={
                        "HIGH": "#ff5269",
                        "MEDIUM": "#ffb547",
                        "LOW": "#25c997",
                        "UNKNOWN": "#8294a9"
                    }
                )

                fig.update_traces(
                    textposition="inside",
                    textinfo="percent+label"
                )

                st.plotly_chart(
                    style_chart(fig),
                    use_container_width=True
                )

        st.divider()

        csv_download(
            filtered,
            "security_alerts_filtered.csv",
            "⬇ Download Filtered Alerts"
        )


# ============================================================
# 11. INVESTIGATION CENTRE
# ============================================================

elif page == "Investigation Centre":

    section_heading(
        "Employee Investigation Centre",
        "Review available security records for an individual employee."
    )

    if alerts.empty or "employee_id" not in alerts.columns:

        st.info(
            "Investigation requires alert records containing "
            "an employee_id column."
        )

    else:

        employee_list = sorted(
            alerts.loc[
                alerts["employee_id"] != "",
                "employee_id"
            ].dropna().unique().tolist()
        )

        if not employee_list:

            st.info("No employee records available.")

        else:

            selected_employee = st.selectbox(
                "Select employee for investigation",
                employee_list,
                key="investigation_employee"
            )

            report = alerts[
                alerts["employee_id"] == selected_employee
            ].copy()

            employee_activities = df[
                df["employee_id"] == selected_employee
            ].copy()

            st.divider()

            st.html(
                f"""
                <h3 style="color:#f0f6ff;">
                    Investigation: {escape(str(selected_employee))}
                </h3>
                """
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "ACTIVITY RECORDS",
                len(employee_activities)
            )

            c2.metric(
                "ALERT RECORDS",
                len(report)
            )

            if "risk_score" in report.columns:

                valid_scores = report["risk_score"].dropna()

                highest_score = (
                    f"{valid_scores.max():.1f}/100"
                    if not valid_scores.empty
                    else "N/A"
                )

            else:
                highest_score = "N/A"

            c3.metric(
                "HIGHEST RECORDED RISK",
                highest_score
            )

            st.divider()

            if not report.empty:

                record = report.iloc[-1]

                left, right = st.columns(2)

                with left:

                    section_heading("Latest Available Alert")

                    for column in [
                        "severity",
                        "risk_score",
                        "ai_prediction"
                    ]:

                        if column in record.index:

                            st.write(
                                f"**{column.replace('_', ' ').title()}:** "
                                f"{record[column]}"
                            )

                with right:

                    section_heading("Detection Reasons")

                    if "detection_reasons" in record.index:

                        reasons = record["detection_reasons"]

                        if pd.notna(reasons) and str(reasons).strip():

                            for reason in str(reasons).split(";"):

                                reason = reason.strip()

                                if reason:
                                    st.warning(reason)

                        else:
                            st.info("No detection reasons available.")

                    else:
                        st.info("Detection reasons are unavailable.")

            st.divider()

            section_heading("Employee Alert History")

            show_table(
                report,
                height=300
            )

            st.divider()

            section_heading("Employee Activity History")

            show_table(
                employee_activities,
                height=300
            )

            csv_download(
                report,
                f"investigation_{selected_employee}.csv",
                "⬇ Download Investigation Report"
            )


# ============================================================
# 12. DATASET EXPLORER
# ============================================================

elif page == "Dataset Explorer":

    section_heading(
        "Dataset Explorer",
        "Explore the activity dataset used by the monitoring system."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "TOTAL ROWS",
        f"{len(df):,}"
    )

    c2.metric(
        "TOTAL COLUMNS",
        len(df.columns)
    )

    c3.metric(
        "UNIQUE EMPLOYEES",
        df["employee_id"].nunique()
    )

    st.divider()

    section_heading("Dataset Preview")

    show_table(
        df.head(100),
        height=400
    )

    st.divider()

    section_heading("Column Information")

    column_info = pd.DataFrame({
        "Column": df.columns.astype(str),
        "Data Type": [
            str(df[column].dtype)
            for column in df.columns
        ],
        "Non-Null Values": [
            int(df[column].notna().sum())
            for column in df.columns
        ],
        "Missing Values": [
            int(df[column].isna().sum())
            for column in df.columns
        ]
    })

    show_table(
        column_info,
        height=300
    )

    st.divider()

    section_heading("Statistical Summary")

    summary = (
        df.describe(include="all")
        .transpose()
        .reset_index()
    )

    summary = summary.rename(
        columns={"index": "Column"}
    )

    show_table(
        summary,
        height=400
    )

    st.divider()

    csv_download(
        df,
        "activity_logs_export.csv",
        "⬇ Download Activity Dataset"
    )


# ============================================================
# 13. FOOTER
# ============================================================

st.divider()

st.html(
    """
    <div class="footer">
        <div style="color:#55cce8;font-weight:800;letter-spacing:1px;">
            INSIDERWATCH AI
        </div>

        <div style="margin-top:6px;">
            Intelligent Insider Threat Monitoring & Behaviour Analytics
        </div>

        <div style="margin-top:8px;color:#647991;">
            Academic demonstration system.
            Suspicious activity is an investigation signal,
            not proof of misconduct.
        </div>
    </div>
    """
)
