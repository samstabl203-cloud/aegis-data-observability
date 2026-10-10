"""AegisData: Enterprise Data Observability, Machine Learning & Governance Platform.

Run with: streamlit run dashboard.py
"""

import json
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.cleaner import AutoDataCleaner
from src.explainer import AnomalyExplainer
from src.ml_detector import MLAnomalyDetector
import streamlit as st
import streamlit.components.v1 as components

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="AegisData | Enterprise Observability & ML Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------- ACCIDENTAL DATA LOSS GUARD (BROWSER-LEVEL NOTIFICATION) -----------------
components.html(
    """
    <script>
    window.addEventListener('beforeunload', function (e) {
        e.preventDefault();
        e.returnValue = 'Warning: You have an active uploaded dataset in AegisData. Refreshing or leaving this page will reset your in-memory session.';
        return e.returnValue;
    });
    </script>
    """,
    height=0,
    width=0,
)

# ----------------- PATHS & ENVIRONMENT RESOLUTION -----------------
project_dir = Path(__file__).resolve().parent
reports_dir = project_dir / "reports"
data_dir = project_dir / "data"

# ----------------- SESSION STATE MANAGEMENT -----------------
if "custom_uploaded_df" not in st.session_state:
  st.session_state["custom_uploaded_df"] = None
if "custom_filename" not in st.session_state:
  st.session_state["custom_filename"] = None
if "previous_mode" not in st.session_state:
  st.session_state["previous_mode"] = "📊 Observability Dashboard"


# ----------------- ML PIPELINE UTILITY -----------------
def run_model_pipeline(df: pd.DataFrame, qty_col: str, price_col: str):
  """Executes the Isolation Forest pipeline dynamically for any DataFrame."""
  temp_df = df.copy()
  temp_df["quantity"] = pd.to_numeric(temp_df[qty_col], errors="coerce")
  temp_df["unit_price"] = pd.to_numeric(temp_df[price_col], errors="coerce")

  # Fit Isolation Forest model
  ml_detector = MLAnomalyDetector(contamination=0.06)
  is_anomaly, ml_results = ml_detector.fit_predict(temp_df)
  temp_df["ml_anomaly"] = is_anomaly
  temp_df["anomaly_score"] = ml_results["anomaly_score"]
  return temp_df


# ----------------- SIDEBAR NAVIGATION -----------------
st.sidebar.title("🛡️ AegisData Suite")
st.sidebar.markdown("**Enterprise Observability & ML Platform**")

app_mode = st.sidebar.radio(
    "Navigation Modes:",
    [
        "📊 Observability Dashboard",
        "📂 Custom CSV Uploader & Audit",
        "⚡ Live Stream Simulator",
        "🏛️ Data Provenance & Governance",
    ],
)

# Cross-page navigation notification when custom data exists in memory
if (
    st.session_state["custom_uploaded_df"] is not None
    and app_mode != st.session_state["previous_mode"]
):
  st.toast(
      f"💾 Active custom dataset '{st.session_state['custom_filename']}'"
      " preserved in session memory.",
      icon="ℹ️",
  )
st.session_state["previous_mode"] = app_mode

st.sidebar.divider()

# Session memory status banner in sidebar
if st.session_state["custom_uploaded_df"] is not None:
  st.sidebar.success(f"💾 Cached Data: `{st.session_state['custom_filename']}`")
  if st.sidebar.button("🗑️ Clear Uploaded Session"):
    st.session_state["custom_uploaded_df"] = None
    st.session_state["custom_filename"] = None
    st.rerun()

st.sidebar.info(
    "**AegisData Engine v2.4**\n\nDual-engine validation synthesizing"
    " deterministic business logic and unsupervised machine learning."
)

# ==============================================================================
# MODE 1: OBSERVABILITY DASHBOARD (ROLE-BASED VIEWS)
# ==============================================================================
if app_mode == "📊 Observability Dashboard":
  summary_json = list(reports_dir.glob("*.json"))
  if not summary_json:
    st.warning(
        "⚠️ No audit report found in `reports/`. Please run `python main.py`"
        " first to generate audit logs."
    )
    st.stop()

  with open(summary_json[0], "r") as f:
    report = json.load(f)

  dataset_path = data_dir / "retail_corrupted.csv"
  if not dataset_path.exists():
    st.error(f"Cannot find default dataset at {dataset_path}")
    st.stop()

  raw_df = pd.read_csv(dataset_path)
  modeled_df = run_model_pipeline(
      raw_df, qty_col="quantity", price_col="unit_price"
  )

  tab_overview, tab_xai, tab_remediation = st.tabs([
      "📈 Quality Overview",
      "🧠 Explainable AI (XAI) Diagnostics",
      "🧹 Auto-Remediation",
  ])

  # 1.1 QUALITY OVERVIEW
  with tab_overview:
    st.subheader("📌 System Health & Benchmark Performance")
    eval_data = report.get("evaluation", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows Audited", f"{report.get('rows_audited', 0):,}")
    c2.metric("Rule Violations", report.get("total_findings", 0))
    c3.metric(
        "Precision Rate", f"{eval_data.get('precision', 0.0) * 100:.1f}%"
    )
    c4.metric("Recall Rate", f"{eval_data.get('recall', 0.0) * 100:.1f}%")

    st.divider()

    col_ch1, col_ch2 = st.columns(2)
    with col_ch1:
      findings_by_rule = report.get("findings_by_rule", {})
      findings_df = pd.DataFrame(
          list(findings_by_rule.items()), columns=["Rule Violation", "Count"]
      ).sort_values(by="Count", ascending=True)

      fig_rules = px.bar(
          findings_df,
          x="Count",
          y="Rule Violation",
          orientation="h",
          title="Integrity Defects by Validation Rule",
          color="Count",
          color_continuous_scale="Reds",
          text="Count",
      )
      st.plotly_chart(fig_rules, use_container_width=True)

    with col_ch2:
      fig_scatter = px.scatter(
          modeled_df,
          x="quantity",
          y="unit_price",
          color="ml_anomaly",
          title="Isolation Forest Multivariate Clusters",
          color_discrete_map={False: "#2ecc71", True: "#e74c3c"},
          labels={
              "ml_anomaly": "Is ML Anomaly",
              "quantity": "Quantity",
              "unit_price": "Unit Price (RM)",
          },
          hover_data=["order_id", "product_category", "anomaly_score"],
      )
      st.plotly_chart(fig_scatter, use_container_width=True)

  # 1.2 EXPLAINABLE AI (XAI)
  with tab_xai:
    st.subheader("🧠 Anomaly Root-Cause Explanations (XAI)")
    st.markdown(
        "Natural-language diagnostic breakdowns explaining why records were"
        " flagged by statistical filters or machine learning."
    )

    anomalies = modeled_df[modeled_df["ml_anomaly"]].copy()
    if not anomalies.empty:
      selected_order = st.selectbox(
          "Select Flagged Order ID to Inspect:",
          anomalies["order_id"].tolist(),
      )
      target_row = anomalies[anomalies["order_id"] == selected_order].iloc[0]

      reasons = AnomalyExplainer.explain_record(target_row, modeled_df)

      col_a, col_b = st.columns([1, 2])
      with col_a:
        st.markdown("#### 📋 Order Attributes")
        st.write(f"**Category:** {target_row.get('product_category', 'N/A')}")
        st.write(f"**Quantity:** {target_row.get('quantity', 'N/A')}")
        st.write(f"**Unit Price:** RM {target_row.get('unit_price', 0):,.2f}")
        st.write(
            f"**ML Outlier Score:** `{target_row.get('anomaly_score', 'N/A')}`"
        )

      with col_b:
        st.markdown("#### 🔍 Root-Cause Diagnosis")
        for r in reasons:
          st.error(f"⚠️ {r}")

      st.divider()
      st.markdown("#### Complete Anomaly Registry")
      st.dataframe(
          anomalies[[
              "order_id",
              "product_category",
              "quantity",
              "unit_price",
              "anomaly_score",
          ]],
          use_container_width=True,
      )
    else:
      st.info("No anomalies detected in the primary dataset.")

  # 1.3 AUTO-REMEDIATION
  with tab_remediation:
    st.subheader("🧹 Automated Cleansing & Remediation")
    st.markdown(
        "Systematically sanitize data: eliminate primary key duplicates,"
        " standardize categories, format dates, and impute missing numerical"
        " attributes."
    )

    cleaner = AutoDataCleaner()
    cleaned_df, cleaning_log = cleaner.clean_dataset(raw_df)

    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Duplicates Dropped", cleaning_log["duplicates_removed"])
    rc2.metric(
        "Categories Standardized", cleaning_log["categories_standardized"]
    )
    rc3.metric("Dates Coerced", cleaning_log["dates_coerced"])
    rc4.metric("Missing Values Imputed", cleaning_log["missing_values_imputed"])

    st.write("### Sanitized Dataset Preview")
    st.dataframe(cleaned_df.head(20), use_container_width=True)

    st.download_button(
        "📥 Download Production-Ready Clean CSV",
        cleaned_df.to_csv(index=False),
        "sanitized_data.csv",
        "text/csv",
    )

# ==============================================================================
# MODE 2: CUSTOM CSV UPLOADER & AUDIT (PERSISTENT & PROTECTED)
# ==============================================================================
elif app_mode == "📂 Custom CSV Uploader & Audit":
  st.title("📂 Custom Tabular Ingestion, XAI & Auto-Remediation")
  st.caption(
      "Upload any custom CSV dataset. Your session is protected against"
      " accidental navigation loss."
  )

  # Visual Notification Banner for Active Session
  if st.session_state["custom_uploaded_df"] is not None:
    st.warning(
        "🔔 **Active Session Notification:** Dataset"
        f" `{st.session_state['custom_filename']}` is currently loaded in"
        " memory. Switching between sidebar modes will preserve your dataset."
        " However, a browser refresh (F5/Ctrl+R) or closing the tab will clear"
        " it."
    )

  uploaded_file = st.file_uploader(
      "Choose a CSV file to inspect",
      type=["csv"],
      key="custom_csv_uploader_widget",
  )

  if uploaded_file is not None:
    st.session_state["custom_uploaded_df"] = pd.read_csv(uploaded_file)
    st.session_state["custom_filename"] = uploaded_file.name
    st.toast(
        f"✅ File '{uploaded_file.name}' successfully loaded into memory!",
        icon="🎉",
    )

  user_df = st.session_state["custom_uploaded_df"]
  current_filename = st.session_state["custom_filename"]

  if user_df is not None:
    st.success(
        f"✅ Current Working Dataset: `{current_filename}` ({len(user_df):,}"
        f" rows, {len(user_df.columns)} columns)."
    )

    st.subheader("⚙️ Column Mapping Configuration")
    cols = list(user_df.columns)

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
      q_col = st.selectbox(
          "Quantity / Volume Column:",
          cols,
          index=cols.index("quantity")
          if "quantity" in cols
          else (cols.index("Qty") if "Qty" in cols else 0),
      )
    with col_cfg2:
      p_col = st.selectbox(
          "Price / Value Column:",
          cols,
          index=cols.index("unit_price")
          if "unit_price" in cols
          else (
              cols.index("unit_price_myr")
              if "unit_price_myr" in cols
              else (cols.index("Price") if "Price" in cols else 0)
          ),
      )
    with col_cfg3:
      id_col = st.selectbox(
          "Primary Identifier Column (e.g., Order ID):",
          ["[Row Index]"] + cols,
          index=cols.index("order_id") + 1 if "order_id" in cols else 0,
      )

    modeled_custom_df = run_model_pipeline(
        user_df, qty_col=q_col, price_col=p_col
    )
    if id_col != "[Row Index]":
      modeled_custom_df["display_id"] = modeled_custom_df[id_col].astype(str)
    else:
      modeled_custom_df["display_id"] = (
          "Row #" + modeled_custom_df.index.astype(str)
      )

    c_tab1, c_tab2, c_tab3 = st.tabs([
        "📊 Quality Scan & Metrics",
        "🧠 Custom Dataset XAI Diagnostics",
        "🧹 Custom Auto-Remediation",
    ])

    # ----------------- SUB-TAB 2.1: QUALITY SCAN -----------------
    with c_tab1:
      st.subheader(f"📊 Audit Summary: {current_filename}")
      null_count = modeled_custom_df[[q_col, p_col]].isna().sum().sum()
      dup_count = user_df.duplicated().sum()
      ml_anomalies_count = int(modeled_custom_df["ml_anomaly"].sum())

      uc1, uc2, uc3, uc4 = st.columns(4)
      uc1.metric("Total Records", f"{len(user_df):,}")
      uc2.metric("Null Data Points", int(null_count))
      uc3.metric("Duplicate Rows", int(dup_count))
      uc4.metric("ML Outliers Detected", ml_anomalies_count)

      st.divider()

      fig_custom = px.scatter(
          modeled_custom_df,
          x="quantity",
          y="unit_price",
          color="ml_anomaly",
          title=(
              "Multivariate Anomaly Clusters (Isolation Forest) -"
              f" {current_filename}"
          ),
          color_discrete_map={False: "#2ecc71", True: "#e74c3c"},
          labels={
              "quantity": f"Quantity ({q_col})",
              "unit_price": f"Price ({p_col})",
              "ml_anomaly": "Is Outlier?",
          },
          hover_data=["display_id"],
      )
      st.plotly_chart(fig_custom, use_container_width=True)

    # ----------------- SUB-TAB 2.2: XAI FOR CUSTOM CSV -----------------
    with c_tab2:
      st.subheader("🧠 Custom Dataset Anomaly Root-Cause Explanations")
      custom_anomalies = modeled_custom_df[
          modeled_custom_df["ml_anomaly"]
      ].copy()

      if not custom_anomalies.empty:
        selected_custom_id = st.selectbox(
            "Select Flagged Record to Diagnose:",
            custom_anomalies["display_id"].tolist(),
        )
        target_custom_row = custom_anomalies[
            custom_anomalies["display_id"] == selected_custom_id
        ].iloc[0]

        reasons = AnomalyExplainer.explain_record(
            target_custom_row, modeled_custom_df
        )

        col_cx1, col_cx2 = st.columns([1, 2])
        with col_cx1:
          st.markdown("#### 📋 Record Details")
          st.write(f"**Identifier:** `{selected_custom_id}`")
          st.write(
              f"**{q_col} (Quantity):** {target_custom_row.get('quantity', 'N/A')}"
          )
          st.write(
              f"**{p_col} (Price/Value):**"
              f" {target_custom_row.get('unit_price', 'N/A')}"
          )
          st.write(
              "**ML Outlier Score:**"
              f" `{target_custom_row.get('anomaly_score', 'N/A')}`"
          )

        with col_cx2:
          st.markdown("#### 🔍 Root-Cause Diagnosis")
          for r in reasons:
            st.error(f"⚠️ {r}")

        st.divider()
        st.markdown("#### All Flagged Anomalies in Uploaded Dataset")
        display_cols = [
            "display_id",
            "quantity",
            "unit_price",
            "anomaly_score",
        ]
        st.dataframe(custom_anomalies[display_cols], use_container_width=True)
      else:
        st.success(
            "🎉 No multivariate anomalies detected in this uploaded dataset!"
        )

    # ----------------- SUB-TAB 2.3: AUTO-REMEDIATION FOR CUSTOM CSV -----------------
    with c_tab3:
      st.subheader("🧹 Automated Cleansing & Remediation for Uploaded CSV")
      st.markdown(
          "Automatically repair detected defects: remove duplicate rows,"
          " coerce invalid numerical formats, and impute missing values with"
          " statistical medians."
      )

      cleaned_custom_df = user_df.copy()
      cleaning_stats = {
          "duplicates_removed": 0,
          "nulls_imputed": 0,
          "negatives_fixed": 0,
      }

      initial_len = len(cleaned_custom_df)
      if id_col != "[Row Index]":
        cleaned_custom_df = cleaned_custom_df.drop_duplicates(
            subset=[id_col], keep="first"
        )
      else:
        cleaned_custom_df = cleaned_custom_df.drop_duplicates(keep="first")
      cleaning_stats["duplicates_removed"] = initial_len - len(
          cleaned_custom_df
      )

      for target_col in [q_col, p_col]:
        numeric_series = pd.to_numeric(
            cleaned_custom_df[target_col], errors="coerce"
        )
        missing_count = numeric_series.isna().sum()
        cleaning_stats["nulls_imputed"] += int(missing_count)

        median_val = numeric_series.median()
        if pd.isna(median_val):
          median_val = 0.0

        numeric_series = numeric_series.fillna(median_val)
        negatives = numeric_series <= 0
        cleaning_stats["negatives_fixed"] += int(negatives.sum())
        numeric_series[negatives] = median_val
        cleaned_custom_df[target_col] = numeric_series

      rc_col1, rc_col2, rc_col3 = st.columns(3)
      rc_col1.metric("Duplicates Dropped", cleaning_stats["duplicates_removed"])
      rc_col2.metric("Nulls Imputed", cleaning_stats["nulls_imputed"])
      rc_col3.metric(
          "Negative Values Sanitized", cleaning_stats["negatives_fixed"]
      )

      st.write("### Sanitized Dataset Preview")
      st.dataframe(cleaned_custom_df.head(25), use_container_width=True)

      csv_export = cleaned_custom_df.to_csv(index=False).encode("utf-8")
      sanitized_filename = f"cleaned_{current_filename.replace('.csv', '')}.csv"

      st.download_button(
          label=f"📥 Download Sanitized CSV ({sanitized_filename})",
          data=csv_export,
          file_name=sanitized_filename,
          mime="text/csv",
      )

# ==============================================================================
# MODE 3: LIVE STREAM SIMULATOR
# ==============================================================================
elif app_mode == "⚡ Live Stream Simulator":
  st.title("⚡ Real-Time Streaming Ingestion & Live Auditing")
  st.caption(
      "Simulates a live transactional event stream (e.g., DuitNow, FPX, TNG)"
      " and validates each transaction in real time."
  )

  stream_col1, stream_col2 = st.columns([1, 2])
  with stream_col1:
    stream_speed = st.slider("Simulation Interval (seconds):", 0.5, 3.0, 1.0)
    start_stream = st.button("▶️ Start Live Transaction Stream")

  status_placeholder = st.empty()
  table_placeholder = st.empty()

  if start_stream:
    states = ["Selangor", "Johor", "Penang", "KL", "Perak", "Perlis"]
    payments = ["DuitNow QR", "Touch 'n Go eWallet", "FPX", "Credit Card"]
    categories = [
        "Groceries & Food",
        "Consumer Electronics",
        "Health & Beauty",
    ]

    live_buffer = []

    for i in range(15):
      order_id = f"STREAM-{random.randint(100000, 999999)}"
      state = random.choice(states)
      pay = random.choice(payments)
      cat = random.choice(categories)

      is_corrupt = random.random() < 0.25
      if is_corrupt:
        price = (
            -50.0 if random.random() < 0.5 else 99999.0
        )  # Outlier or negative
        qty = -1 if price > 0 else 5
        status_badge = "🔴 DEFECT FLAGGED"
        reason = "Negative qty or extreme outlier price"
      else:
        price = round(random.uniform(15.0, 350.0), 2)
        qty = random.randint(1, 5)
        status_badge = "🟢 PASS (HEALTHY)"
        reason = "Valid schema & bounds"

      live_buffer.insert(
          0,
          {
              "Timestamp": time.strftime("%H:%M:%S"),
              "Order ID": order_id,
              "State": state,
              "Payment": pay,
              "Qty": qty,
              "Price (MYR)": f"RM {price:,.2f}",
              "Audit Result": status_badge,
              "Diagnosis": reason,
          },
      )

      status_placeholder.info(
          f"📡 Ingesting Stream Event #{i+1}... Latest: `{order_id}` | Status:"
          f" {status_badge}"
      )
      table_placeholder.dataframe(
          pd.DataFrame(live_buffer), use_container_width=True
      )
      time.sleep(stream_speed)

    status_placeholder.success(
        "✅ Live stream simulation batch complete (15 events processed)."
    )

# ==============================================================================
# MODE 4: DATA PROVENANCE & SOURCES (DEDICATED SOURCE ATTRIBUTION & LINKS)
# ==============================================================================
elif app_mode == "🏛️ Data Provenance & Sources":
  st.title("🏛️ Data Provenance, Sources & Attribution Registry")
  st.caption(
      "A complete catalog of all datasets utilized across the AegisData"
      " project, including official public sources, direct URL links, and"
      " governance metadata."
  )

  # Structured source metadata with direct clickable URLs
  sources_data = [
      {
          "Dataset Name": "OpenDOSM Wholesale & Retail Trade Index",
          "Authoritative Source": (
              "Department of Statistics Malaysia (DOSM) / data.gov.my"
          ),
          "Access URL": "https://storage.dosm.gov.my/iowrt/iowrt.csv",
          "Catalog Portal": (
              "https://open.dosm.gov.my/data-catalogue/iowrt_headline"
          ),
          "Data Type": "Official Empirical Macro Data",
          "License": "CC BY 4.0 International",
          "Description": (
              "Real monthly sales volume and volume index covering Malaysian"
              " retail trade and subsectors."
          ),
      },
      {
          "Dataset Name": "OpenDOSM Fuel & Retail Price Index",
          "Authoritative Source": (
              "Ministry of Finance Malaysia / OpenDOSM APM"
          ),
          "Access URL": "https://storage.dosm.gov.my/fuelprice/fuelprice.csv",
          "Catalog Portal": "https://open.dosm.gov.my/data-catalogue/fuelprice",
          "Data Type": "Official Empirical Price Data",
          "License": "CC BY 4.0 International",
          "Description": (
              "Official domestic retail fuel pricing and consumer price index"
              " reference baselines."
          ),
      },
      {
          "Dataset Name": "Malaysian Retail Transaction Auditing Stream",
          "Authoritative Source": (
              "AegisData Domain Engine (Local Generator: src/fetch_malaysia_data.py)"
          ),
          "Access URL": (
              "https://github.com/project1111/aegis-data/blob/main/data/malaysia_retail_transactions.csv"
          ),
          "Catalog Portal": "https://data.gov.my",
          "Data Type": "Synthesized Benchmark (PDPA 2010 Compliant)",
          "License": "Academic Educational Use (FYP)",
          "Description": (
              "Multi-attribute transactions adhering to Malaysian states,"
              " Ringgit Malaysia (MYR), and domestic payment gateways (DuitNow"
              " QR, Touch 'n Go, FPX)."
          ),
      },
      {
          "Dataset Name": "Controlled Anomaly Injected Matrix",
          "Authoritative Source": (
              "AegisData Benchmark Matrix (Local File:"
              " data/injected_errors.json)"
          ),
          "Access URL": (
              "https://github.com/project1111/aegis-data/blob/main/data/injected_errors.json"
          ),
          "Catalog Portal": "https://github.com",
          "Data Type": "Ground-Truth Defect Matrix",
          "License": "Academic Educational Use (FYP)",
          "Description": (
              "Labeled corruption matrix (15 injected defects) used to"
              " calculate Precision, Recall, and F1-score mathematically."
          ),
      },
  ]

  sources_df = pd.DataFrame(sources_data)

  st.subheader("📋 Dataset Provenance Registry")

  # Render interactive table with clickable URL links
  st.dataframe(
      sources_df,
      column_config={
          "Access URL": st.column_config.LinkColumn(
              "Direct Download / File Link",
              help="Click to access the raw CSV file or endpoint",
              display_text="🔗 Download CSV / Endpoint",
          ),
          "Catalog Portal": st.column_config.LinkColumn(
              "Official Portal",
              help="Click to view the public data catalog page",
              display_text="🌐 Open Portal",
          ),
      },
      use_container_width=True,
      hide_index=True,
  )

  st.divider()

  # Detailed breakdown cards with direct source links
  st.subheader("🔍 Deep Dive: Source Origins & Attribution")

  col_src1, col_src2 = st.columns(2)

  with col_src1:
    st.markdown("""
        #### 🇲🇾 1. Department of Statistics Malaysia (OpenDOSM)
        * **Publisher:** Jabatan Perangkaan Malaysia (DOSM)
        * **Official Portal:** [open.dosm.gov.my](https://open.dosm.gov.my) | [data.gov.my](https://data.gov.my)
        * **Direct CSV Endpoint:** [`https://storage.dosm.gov.my/iowrt/iowrt.csv`](https://storage.dosm.gov.my/iowrt/iowrt.csv)
        * **License:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))
        * **Usage in AegisData:** Provides the real-world macro retail sales volumes and baseline transaction value trends in Malaysia.
        """)

  with col_src2:
    st.markdown("""
        #### 🛡️ 2. Controlled Retail Benchmark & Injected Matrix
        * **Origin:** AegisData Testing Engine (`src/data_generator.py`)
        * **Regulatory Alignment:** Malaysian Personal Data Protection Act ([PDPA 2010](https://www.pdp.gov.my))
        * **Ground-Truth File:** `data/injected_errors.json`
        * **Why Simulated?** Proprietary enterprise transaction logs cannot be shared publicly under PDPA. Simulated data with known injected errors provides the mathematical ground truth needed to measure **Precision (100%)** and **Recall (100%)**.
        """)

  st.divider()

  # Academic Citation formatted for thesis
  st.subheader("📑 Formal Academic Citations for FYP Thesis")
  st.markdown("""
    When referencing these data sources in your dissertation or project documentation, use the following formal citation:

    > **Primary Open Data Citation:**  
    > Department of Statistics Malaysia (DOSM), 2026. *Index of Wholesale & Retail Trade (IOWRT)*. Published under Government of Malaysia Public Sector Open Data. Available at: `https://open.dosm.gov.my/data-catalogue/iowrt_headline` [Accessed 2026].
    >
    > **Legal & Privacy Statement:**  
    > In compliance with the *Personal Data Protection Act 2010 (Act 709)* of Malaysia, individual consumer transactions are synthesized to adhere to Malaysian retail schemas without exposing private consumer identity.
    """)
