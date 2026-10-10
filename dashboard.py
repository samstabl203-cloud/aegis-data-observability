"""AegisData: Enterprise Data Observability & ML Engine.

Enhanced User-Friendly Edition with Modern UI, Tooltips, Sample Pickers, and
Interactive Filters.
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
    page_title="AegisData | Enterprise Observability & ML Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------- CUSTOM CSS FOR POLISHED MODERN UI -----------------
st.markdown(
    """
    <style>
    /* Global Typography & Background Adjustments */
    .main {
        background-color: #fcfcfd;
    }
    
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #eef0f4;
        padding: 16px 20px;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    }
    
    /* Badges */
    .badge-clean {
        background-color: #e6f9ed;
        color: #1a7f37;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .badge-warning {
        background-color: #fff8e6;
        color: #9a6700;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .badge-critical {
        background-color: #ffebe9;
        color: #cf222e;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.82rem;
    }

    /* Styled Container Boxes */
    .insight-card {
        background: #ffffff;
        border-radius: 10px;
        border-left: 4px solid #1f6feb;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ----------------- ACCIDENTAL DATA LOSS GUARD -----------------
components.html(
    """
    <script>
    window.addEventListener('beforeunload', function (e) {
        e.preventDefault();
        e.returnValue = 'Warning: Active session in progress. Leaving or reloading will reset temporary in-memory datasets.';
        return e.returnValue;
    });
    </script>
    """,
    height=0,
    width=0,
)

# ----------------- PATH RESOLUTION -----------------
project_dir = Path(__file__).resolve().parent
reports_dir = project_dir / "reports"
data_dir = project_dir / "data"

# ----------------- SESSION STATE -----------------
if "custom_uploaded_df" not in st.session_state:
  st.session_state["custom_uploaded_df"] = None
if "custom_filename" not in st.session_state:
  st.session_state["custom_filename"] = None
if "previous_mode" not in st.session_state:
  st.session_state["previous_mode"] = "📊 Observability Dashboard"


# ----------------- MODEL PIPELINE -----------------
def run_model_pipeline(df: pd.DataFrame, qty_col: str, price_col: str):
  temp_df = df.copy()
  temp_df["quantity"] = pd.to_numeric(temp_df[qty_col], errors="coerce")
  temp_df["unit_price"] = pd.to_numeric(temp_df[price_col], errors="coerce")

  ml_detector = MLAnomalyDetector(contamination=0.06)
  is_anomaly, ml_results = ml_detector.fit_predict(temp_df)
  temp_df["ml_anomaly"] = is_anomaly
  temp_df["anomaly_score"] = ml_results["anomaly_score"]
  return temp_df


# ----------------- SIDEBAR NAVIGATION -----------------
st.sidebar.image(
    "https://img.icons8.com/fluency/96/shield.png", width=64
)  # Clean aesthetic icon
st.sidebar.title("AegisData")
st.sidebar.caption("Data Observability & Anomaly Assurance")

PAGE_OBSERVABILITY = "📊 Observability Dashboard"
PAGE_CUSTOM_CSV = "📂 Custom CSV Uploader & Audit"
PAGE_STREAM_SIM = "⚡ Live Stream Simulator"
PAGE_PROVENANCE = "🏛️ Data Provenance & Sources"

app_mode = st.sidebar.radio(
    "Select Viewpoint:",
    [PAGE_OBSERVABILITY, PAGE_CUSTOM_CSV, PAGE_STREAM_SIM, PAGE_PROVENANCE],
)

# Session persistence toast
if (
    st.session_state["custom_uploaded_df"] is not None
    and app_mode != st.session_state["previous_mode"]
):
  st.toast(
      f"💾 Working dataset '{st.session_state['custom_filename']}' preserved.",
      icon="🛡️",
  )
st.session_state["previous_mode"] = app_mode

st.sidebar.divider()

# Session memory status indicator
if st.session_state["custom_uploaded_df"] is not None:
  st.sidebar.info(f"📁 **Active:** `{st.session_state['custom_filename']}`")
  if st.sidebar.button("🗑️ Reset Uploaded Session", use_container_width=True):
    st.session_state["custom_uploaded_df"] = None
    st.session_state["custom_filename"] = None
    st.rerun()

st.sidebar.markdown(
    """
---
💡 **Platform Quick Guide:**
- **Observability**: Benchmark performance & XAI.
- **Custom Audit**: Upload your own CSV or test samples.
- **Live Simulator**: Real-time streaming feed.
- **Data Provenance**: Official OpenDOSM sources.
"""
)

# ==============================================================================
# MODE 1: OBSERVABILITY DASHBOARD
# ==============================================================================
if app_mode == PAGE_OBSERVABILITY:
  st.title("📊 Enterprise Observability & Benchmark Dashboard")
  st.markdown(
      "Comprehensive data health monitoring synthesizing **deterministic"
      " business rules** and **unsupervised Isolation Forest modeling**."
  )

  summary_json = list(reports_dir.glob("*.json"))
  if not summary_json:
    st.warning(
        "⚠️ No pre-computed audit reports found in `reports/`. Please run"
        " `python main.py` first."
    )
    st.stop()

  with open(summary_json[0], "r") as f:
    report = json.load(f)

  dataset_path = data_dir / "retail_corrupted.csv"
  if not dataset_path.exists():
    st.error("Default corrupted dataset missing.")
    st.stop()

  raw_df = pd.read_csv(dataset_path)
  modeled_df = run_model_pipeline(
      raw_df, qty_col="quantity", price_col="unit_price"
  )

  tab_overview, tab_xai, tab_remediation = st.tabs([
      "📈 Executive Overview",
      "🧠 Explainable AI (XAI)",
      "🧹 Auto-Remediation",
  ])

  # TAB 1: EXECUTIVE OVERVIEW
  with tab_overview:
    eval_data = report.get("evaluation", {})

    st.markdown("### 📌 Executive Health & Performance Scorecard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Audited Records",
        f"{report.get('rows_audited', 0):,}",
        help="Total individual transactions validated in this batch run.",
    )
    c2.metric(
        "Total Defects Flagged",
        report.get("total_findings", 0),
        delta=f"{report.get('rows_with_findings', 0)} rows affected",
        delta_color="inverse",
        help="Number of discrete violations across all rules.",
    )
    c3.metric(
        "Detection Recall Rate",
        f"{eval_data.get('recall', 0.0) * 100:.1f}%",
        help="Proportion of actual injected errors successfully caught (100% = 0 missed defects).",
    )
    c4.metric(
        "Benchmark F1-Score",
        f"{eval_data.get('f1_score', 0.0) * 100:.1f}%",
        help="Harmonic balance between precision (avoiding false alarms) and recall.",
    )

    st.divider()

    chart_left, chart_right = st.columns(2)
    with chart_left:
      findings_by_rule = report.get("findings_by_rule", {})
      findings_df = pd.DataFrame(
          list(findings_by_rule.items()), columns=["Validation Rule", "Count"]
      ).sort_values(by="Count", ascending=True)

      fig_rules = px.bar(
          findings_df,
          x="Count",
          y="Validation Rule",
          orientation="h",
          title="Detected Violations by Rule Type",
          color="Count",
          color_continuous_scale="Reds",
          text="Count",
      )
      fig_rules.update_layout(
          plot_bgcolor="rgba(0,0,0,0)",
          paper_bgcolor="rgba(0,0,0,0)",
          height=380,
      )
      st.plotly_chart(fig_rules, use_container_width=True)

    with chart_right:
      fig_scatter = px.scatter(
          modeled_df,
          x="quantity",
          y="unit_price",
          color="ml_anomaly",
          title="Isolation Forest: Multivariate Anomaly Clusters",
          color_discrete_map={False: "#2ecc71", True: "#e74c3c"},
          labels={
              "ml_anomaly": "Is ML Anomaly",
              "quantity": "Quantity",
              "unit_price": "Unit Price (RM)",
          },
          hover_data=["order_id", "product_category", "anomaly_score"],
      )
      fig_scatter.update_layout(
          plot_bgcolor="rgba(0,0,0,0)",
          paper_bgcolor="rgba(0,0,0,0)",
          height=380,
      )
      st.plotly_chart(fig_scatter, use_container_width=True)

  # TAB 2: EXPLAINABLE AI (XAI)
  with tab_xai:
    st.markdown("### 🧠 Explainable AI Root-Cause Diagnostic Inspector")
    st.caption(
        "Examine diagnostic explanations detailing exactly why records were"
        " flagged as outliers."
    )

    anomalies = modeled_df[modeled_df["ml_anomaly"]].copy()
    if not anomalies.empty:
      col_sel, col_blank = st.columns([1, 2])
      with col_sel:
        selected_order = st.selectbox(
            "Select Flagged Order ID to Inspect:",
            anomalies["order_id"].tolist(),
        )

      target_row = anomalies[anomalies["order_id"] == selected_order].iloc[0]
      reasons = AnomalyExplainer.explain_record(target_row, modeled_df)

      box_left, box_right = st.columns([1, 2])
      with box_left:
        st.markdown("##### 📋 Offending Record Details")
        st.write(f"**Order ID:** `{target_row.get('order_id', 'N/A')}`")
        st.write(f"**Category:** {target_row.get('product_category', 'N/A')}")
        st.write(f"**Quantity:** {target_row.get('quantity', 'N/A')}")
        st.write(f"**Unit Price:** RM {target_row.get('unit_price', 0):,.2f}")
        st.write(
            f"**Isolation Forest Score:** `{target_row.get('anomaly_score', 'N/A')}`"
        )

      with box_right:
        st.markdown("##### 🔍 Root-Cause Diagnosis")
        for r in reasons:
          st.error(f"⚠️ {r}")

      st.divider()
      st.markdown("##### Complete Flagged Anomaly Registry")
      st.dataframe(
          anomalies[[
              "order_id",
              "product_category",
              "quantity",
              "unit_price",
              "anomaly_score",
          ]],
          use_container_width=True,
          hide_index=True,
      )
    else:
      st.success("No anomalies detected in the inspected dataset.")

  # TAB 3: AUTO-REMEDIATION
  with tab_remediation:
    st.markdown("### 🧹 Automated Remediation & Data Sanitization")
    st.caption(
        "One-click pipeline to resolve schema violations: drop duplicate keys,"
        " coerce dates, standardize strings, and impute missing values."
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

    st.markdown("##### Sanitized Production Preview (Clean Output)")
    st.dataframe(cleaned_df.head(15), use_container_width=True)

    st.download_button(
        "📥 Download Sanitized Clean CSV",
        cleaned_df.to_csv(index=False),
        "sanitized_data.csv",
        "text/csv",
        use_container_width=True,
    )

# ==============================================================================
# MODE 2: CUSTOM CSV UPLOADER & AUDIT (ENHANCED USER-FRIENDLY UX)
# ==============================================================================
elif app_mode == PAGE_CUSTOM_CSV:
  st.title("📂 Custom Tabular Ingestion & Dynamic Audit")
  st.markdown(
      "Inspect any custom dataset with automatic column detection, interactive"
      " mapping, and real-time anomaly diagnostics."
  )

  # Sample Dataset Picker for easy evaluation
  st.markdown("#### ⚡ Quick Demo or File Upload")
  demo_choice = st.radio(
      "Choose Data Input Method:",
      [
          "Upload My Own CSV File",
          "Load Pre-packaged Sample (Malaysian Retail Transactions)",
          "Load Pre-packaged Sample (Corrupted Retail Benchmark)",
      ],
      horizontal=True,
  )

  loaded_df = None
  selected_filename = ""

  if demo_choice == "Upload My Own CSV File":
    uploaded_file = st.file_uploader(
        "Drag and drop your CSV file here:",
        type=["csv"],
        help="Upload standard comma-delimited tabular CSV files.",
    )
    if uploaded_file is not None:
      loaded_df = pd.read_csv(uploaded_file)
      selected_filename = uploaded_file.name
  elif (
      demo_choice
      == "Load Pre-packaged Sample (Malaysian Retail Transactions)"
  ):
    sample_path = data_dir / "malaysia_retail_transactions.csv"
    if sample_path.exists():
      loaded_df = pd.read_csv(sample_path)
      selected_filename = "malaysia_retail_transactions.csv"
    else:
      st.info("Sample not found locally. Run fetch script to generate.")
  else:
    sample_path = data_dir / "retail_corrupted.csv"
    if sample_path.exists():
      loaded_df = pd.read_csv(sample_path)
      selected_filename = "retail_corrupted.csv"

  if loaded_df is not None:
    st.session_state["custom_uploaded_df"] = loaded_df
    st.session_state["custom_filename"] = selected_filename

  user_df = st.session_state["custom_uploaded_df"]
  current_filename = st.session_state["custom_filename"]

  if user_df is not None:
    st.success(
        f"✅ Loaded: `{current_filename}` — {len(user_df):,} records,"
        f" {len(user_df.columns)} features."
    )

    with st.expander("🛠️ Column Mapping Settings", expanded=True):
      cols = list(user_df.columns)
      c_map1, c_map2, c_map3 = st.columns(3)

      # Smart automatic column guessing
      default_qty_idx = next(
          (
              i
              for i, c in enumerate(cols)
              if "qty" in c.lower() or "quantity" in c.lower()
          ),
          0,
      )
      default_price_idx = next(
          (
              i
              for i, c in enumerate(cols)
              if "price" in c.lower()
              or "amount" in c.lower()
              or "myr" in c.lower()
          ),
          0,
      )
      default_id_idx = next(
          (i for i, c in enumerate(cols) if "id" in c.lower()), 0
      )

      with c_map1:
        q_col = st.selectbox(
            "Quantity Column:",
            cols,
            index=default_qty_idx,
            help="Column specifying order transaction volume.",
        )
      with c_map2:
        p_col = st.selectbox(
            "Price Column:",
            cols,
            index=default_price_idx,
            help="Column specifying monetary or unit value.",
        )
      with c_map3:
        id_col = st.selectbox(
            "Unique ID Column:",
            ["[Row Index]"] + cols,
            index=default_id_idx + 1,
            help="Primary key identifier for rows.",
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
        "📊 Data Health & Scan",
        "🧠 Custom XAI Root-Cause",
        "🧹 Auto-Clean & Download",
    ])

    with c_tab1:
      uc1, uc2, uc3, uc4 = st.columns(4)
      nulls = modeled_custom_df[[q_col, p_col]].isna().sum().sum()
      dups = user_df.duplicated().sum()
      anoms = int(modeled_custom_df["ml_anomaly"].sum())

      uc1.metric("Inspected Records", f"{len(user_df):,}")
      uc2.metric(
          "Missing Entries",
          int(nulls),
          delta="Imputable" if nulls > 0 else None,
      )
      uc3.metric(
          "Duplicate Rows", int(dups), delta="Removable" if dups > 0 else None
      )
      uc4.metric("ML Outliers Detected", anoms, delta_color="inverse")

      fig_custom = px.scatter(
          modeled_custom_df,
          x="quantity",
          y="unit_price",
          color="ml_anomaly",
          title=f"Multivariate Anomaly Clusters: {current_filename}",
          color_discrete_map={False: "#2ecc71", True: "#e74c3c"},
          labels={
              "quantity": f"Quantity ({q_col})",
              "unit_price": f"Price ({p_col})",
              "ml_anomaly": "Flagged Outlier",
          },
          hover_data=["display_id"],
      )
      st.plotly_chart(fig_custom, use_container_width=True)

    with c_tab2:
      custom_anomalies = modeled_custom_df[
          modeled_custom_df["ml_anomaly"]
      ].copy()
      if not custom_anomalies.empty:
        sel_custom_id = st.selectbox(
            "Choose Record ID to Diagnose:",
            custom_anomalies["display_id"].tolist(),
        )
        t_row = custom_anomalies[
            custom_anomalies["display_id"] == sel_custom_id
        ].iloc[0]
        reasons = AnomalyExplainer.explain_record(t_row, modeled_custom_df)

        x_col1, x_col2 = st.columns([1, 2])
        with x_col1:
          st.write(f"**ID:** `{sel_custom_id}`")
          st.write(f"**{q_col}:** {t_row.get('quantity', 'N/A')}")
          st.write(f"**{p_col}:** {t_row.get('unit_price', 'N/A')}")
          st.write(f"**Score:** `{t_row.get('anomaly_score', 'N/A')}`")
        with x_col2:
          for r in reasons:
            st.error(f"⚠️ {r}")

        st.dataframe(
            custom_anomalies[[
                "display_id",
                "quantity",
                "unit_price",
                "anomaly_score",
            ]],
            use_container_width=True,
            hide_index=True,
        )
      else:
        st.success("🎉 No multi-attribute anomalies found in this dataset!")

    with c_tab3:
      st.markdown("##### 🧹 Automated Remediation for Custom Dataset")
      cleaned_custom_df = user_df.copy()
      if id_col != "[Row Index]":
        cleaned_custom_df = cleaned_custom_df.drop_duplicates(
            subset=[id_col], keep="first"
        )
      else:
        cleaned_custom_df = cleaned_custom_df.drop_duplicates(keep="first")

      for target_c in [q_col, p_col]:
        numeric_s = pd.to_numeric(
            cleaned_custom_df[target_c], errors="coerce"
        )
        med_val = numeric_s.median() if pd.notna(numeric_s.median()) else 0.0
        numeric_s = numeric_s.fillna(med_val)
        numeric_s[numeric_s <= 0] = med_val
        cleaned_custom_df[target_c] = numeric_s

      st.dataframe(cleaned_custom_df.head(15), use_container_width=True)
      st.download_button(
          label=f"📥 Download Sanitized {current_filename}",
          data=cleaned_custom_df.to_csv(index=False),
          file_name=f"sanitized_{current_filename}",
          mime="text/csv",
          use_container_width=True,
      )

# ==============================================================================
# MODE 3: LIVE STREAM SIMULATOR
# ==============================================================================
elif app_mode == PAGE_STREAM_SIM:
  st.title("⚡ Live Streaming Transaction Ingestion Simulator")
  st.markdown(
      "Simulates high-velocity e-commerce events (e.g., DuitNow, FPX, TNG"
      " eWallet) with **real-time micro-batch auditing**."
  )

  sim_cfg1, sim_cfg2 = st.columns([1, 3])
  with sim_cfg1:
    stream_speed = st.slider("Emission Interval (seconds):", 0.5, 2.5, 1.0)
    start_stream = st.button(
        "▶️ Launch Streaming Feed",
        type="primary",
        use_container_width=True,
    )

  status_box = st.empty()
  stream_table = st.empty()

  if start_stream:
    states = ["Selangor", "Johor", "Penang", "KL", "Perak", "Perlis"]
    payments = ["DuitNow QR", "Touch 'n Go eWallet", "FPX", "Credit Card"]
    categories = [
        "Groceries & Food",
        "Consumer Electronics",
        "Health & Beauty",
    ]
    live_buffer = []

    for i in range(12):
      order_id = f"STREAM-{random.randint(100000, 999999)}"
      state = random.choice(states)
      pay = random.choice(payments)
      cat = random.choice(categories)

      is_corrupt = random.random() < 0.25
      if is_corrupt:
        price = -50.0 if random.random() < 0.5 else 99999.0
        qty = -1 if price > 0 else 5
        status_badge = "🔴 FLAGGED DEFECT"
        reason = "Invalid bounds or extreme outlier"
      else:
        price = round(random.uniform(15.0, 350.0), 2)
        qty = random.randint(1, 5)
        status_badge = "🟢 CLEAN PASS"
        reason = "Within standard distribution"

      live_buffer.insert(
          0,
          {
              "Time": time.strftime("%H:%M:%S"),
              "Event ID": order_id,
              "Region / State": state,
              "Channel": pay,
              "Units": qty,
              "Amount (MYR)": f"RM {price:,.2f}",
              "Health Status": status_badge,
              "Diagnostic Reason": reason,
          },
      )

      status_box.info(
          f"📡 Processing Event #{i+1}... Latest: `{order_id}` | Status:"
          f" **{status_badge}**"
      )
      stream_table.dataframe(
          pd.DataFrame(live_buffer), use_container_width=True, hide_index=True
      )
      time.sleep(stream_speed)

    status_box.success("✅ Micro-batch stream sequence completed.")

# ==============================================================================
# MODE 4: DATA PROVENANCE & SOURCES
# ==============================================================================
elif app_mode == PAGE_PROVENANCE:
  st.title("🏛️ Data Provenance & Official Source Registry")
  st.markdown(
      "Authoritative catalog of official government portals, empirical macro"
      " indices, and direct dataset download links."
  )

  sources_data = [
      {
          "Dataset Identifier": "OpenDOSM Wholesale & Retail Trade Index",
          "Authoritative Publisher": (
              "Department of Statistics Malaysia (DOSM) / data.gov.my"
          ),
          "Direct Access Link": "https://storage.dosm.gov.my/iowrt/iowrt.csv",
          "Catalog Portal": (
              "https://open.dosm.gov.my/data-catalogue/iowrt_headline"
          ),
          "Dataset Classification": "Official Macro Empirical Data",
          "Licensing Terms": "CC BY 4.0 International",
          "Operational Scope": (
              "Official volume and sales value indices across Malaysian retail"
              " trade subsectors."
          ),
      },
      {
          "Dataset Identifier": "OpenDOSM Fuel & Retail Price Index",
          "Authoritative Publisher": (
              "Ministry of Finance Malaysia / OpenDOSM APM"
          ),
          "Direct Access Link": (
              "https://storage.dosm.gov.my/fuelprice/fuelprice.csv"
          ),
          "Catalog Portal": "https://open.dosm.gov.my/data-catalogue/fuelprice",
          "Dataset Classification": "Official Empirical Price Series",
          "Licensing Terms": "CC BY 4.0 International",
          "Operational Scope": (
              "Weekly domestic retail prices for petroleum products and"
              " consumer price index reference."
          ),
      },
      {
          "Dataset Identifier": "Malaysian Retail Transaction Auditing Stream",
          "Authoritative Publisher": (
              "AegisData Domain Engine (Local: src/fetch_malaysia_data.py)"
          ),
          "Direct Access Link": "https://data.gov.my",
          "Catalog Portal": "https://data.gov.my",
          "Dataset Classification": "Synthesized Benchmark (PDPA Compliant)",
          "Licensing Terms": "Educational Use (FYP)",
          "Operational Scope": (
              "Multi-attribute transactions reflecting Malaysian regional"
              " states and domestic e-payments (DuitNow, TNG, FPX)."
          ),
      },
      {
          "Dataset Identifier": "Controlled Anomaly Injected Matrix",
          "Authoritative Publisher": (
              "AegisData Testing Suite (Local: data/injected_errors.json)"
          ),
          "Direct Access Link": (
              "https://raw.githubusercontent.com/project1111/aegis-data/main/data/injected_errors.json"
          ),
          "Catalog Portal": "https://github.com",
          "Dataset Classification": "Ground-Truth Defect Matrix",
          "Licensing Terms": "Educational Use (FYP)",
          "Operational Scope": (
              "Injected defect labels enabling mathematical precision, recall,"
              " and F1-score evaluation."
          ),
      },
  ]

  st.dataframe(
      pd.DataFrame(sources_data),
      column_config={
          "Direct Access Link": st.column_config.LinkColumn(
              "Direct CSV / API Endpoint",
              help="Click to access raw data stream",
              display_text="🔗 Download Endpoint",
          ),
          "Catalog Portal": st.column_config.LinkColumn(
              "Government Catalog",
              help="Click to inspect metadata",
              display_text="🌐 Open Portal",
          ),
      },
      use_container_width=True,
      hide_index=True,
  )

  st.divider()

  prov_col1, prov_col2 = st.columns(2)
  with prov_col1:
    st.markdown("""
        ##### 🇲🇾 Department of Statistics Malaysia (OpenDOSM)
        - **Official API Source:** [`storage.dosm.gov.my/iowrt/iowrt.csv`](https://storage.dosm.gov.my/iowrt/iowrt.csv)
        - **Official Data Portal:** [open.dosm.gov.my](https://open.dosm.gov.my)
        - **License:** Creative Commons Attribution 4.0 ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))
        """)
  with prov_col2:
    st.markdown("""
        ##### 🛡️ Controlled Anomaly Benchmark & Privacy Compliance
        - **Compliance Standard:** Personal Data Protection Act ([PDPA 2010](https://www.pdp.gov.my))
        - **Controlled Answer Key:** `data/injected_errors.json`
        - **Benchmarking Precision:** Measured against known synthetic corruptions to prevent ungrounded hallucination.
        """)
