from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import plotly.io as pio
import streamlit as st
from dotenv import load_dotenv

from agent_workflow import analyze
from dataset_manager import Dataset, load_dataset, schema_for_agent


def render_chart(path: str) -> None:
    """Render interactive Plotly artifacts and retain PNG compatibility."""
    chart_path = Path(path)
    if chart_path.name.endswith(".plotly.json"):
        figure = pio.from_json(chart_path.read_text(encoding="utf-8"))
        figure.update_layout(
            template="plotly_white",
            margin=dict(l=30, r=25, t=55, b=30),
            font=dict(family="DM Sans", color="#17202a"),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
        )
        st.plotly_chart(figure, width="stretch", config={"displaylogo": False, "responsive": True})
    else:
        st.image(path, width="stretch")


load_dotenv()
st.set_page_config(page_title="Northstar Analytics", page_icon="✦", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@700;800&display=swap');
:root { --ink:#17202a; --muted:#667085; --brand:#5b4cf0; --panel:#ffffff; }
.stApp { background: radial-gradient(circle at 85% 0%, #e9e6ff 0, transparent 26rem), #f6f7fb; color:var(--ink); }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
h1,h2,h3 { font-family:'Manrope',sans-serif!important; letter-spacing:-.035em; }
[data-testid="stMainBlockContainer"] { max-width:1180px; padding-top:2.4rem; padding-bottom:8rem; }
[data-testid="stAppViewContainer"] main,
[data-testid="stAppViewContainer"] main p,
[data-testid="stAppViewContainer"] main label,
[data-testid="stAppViewContainer"] main h1,
[data-testid="stAppViewContainer"] main h2,
[data-testid="stAppViewContainer"] main h3,
[data-testid="stAppViewContainer"] main .stMarkdown { color:#17202a!important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label { color:#f8fafc!important; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#1d1f28 0%,#292b38 100%)!important; border-right:1px solid #383b4b; }
[data-testid="stSidebar"][aria-expanded="true"] { width:340px!important; min-width:340px!important; max-width:340px!important; }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top:1rem; }
.brand { display:flex; align-items:center; gap:.7rem; margin:.15rem 0 1.2rem; }
.brand-mark { width:36px; height:36px; border-radius:11px; display:grid; place-items:center; color:white; font-weight:800; background:linear-gradient(135deg,#7567ff,#a855f7); box-shadow:0 8px 24px rgba(117,103,255,.35); }
.brand-name { color:#fff; font-family:'Manrope',sans-serif; font-weight:800; font-size:1.08rem; }
.brand-sub { color:#9da4b6; font-size:.73rem; }
.session-pill { display:inline-flex; align-items:center; gap:.45rem; background:#eeecff; color:#5145cd; border:1px solid #dcd7ff; border-radius:99px; padding:.38rem .72rem; font-size:.78rem; font-weight:700; }
.hero { padding:1.2rem 0 1.1rem; }
.eyebrow { color:#5b4cf0; font-size:.78rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
.hero h1 { font-size:3.25rem; line-height:1.04; margin:.35rem 0 .7rem; background:linear-gradient(110deg,#17202a 12%,#4f46e5 70%,#9333ea); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.hero p { color:#667085; max-width:720px; font-size:1.05rem; }
.metric-card { background:linear-gradient(145deg,#fff,#fafaff); border:1px solid #e4e3f1; border-radius:18px; padding:1.05rem 1.15rem; box-shadow:0 10px 32px rgba(37,35,82,.07); transition:transform .2s ease,box-shadow .2s ease; }
.metric-card:hover { transform:translateY(-2px); box-shadow:0 14px 38px rgba(37,35,82,.11); }
.metric-label { color:#667085; font-size:.78rem; text-transform:uppercase; letter-spacing:.07em; }
.metric-value { color:#17202a; font-size:1.35rem; font-weight:700; margin-top:.25rem; }
[data-testid="stFileUploader"] { background:#fff; border-radius:16px; padding:.5rem; border:1px solid #e9eaf0; }
[data-testid="stChatMessage"] { background:rgba(255,255,255,.92); color:#17202a!important; border:1px solid #e0e2ec; border-radius:20px; padding:.55rem .8rem; box-shadow:0 10px 28px rgba(25,30,60,.055); backdrop-filter:blur(12px); }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { background:linear-gradient(135deg,#f1efff,#faf9ff); border-color:#ddd8ff; }
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div { color:#17202a!important; }
[data-testid="stAlert"] p { color:inherit!important; }
[data-testid="stExpander"] summary p { color:#17202a!important; font-weight:600; }
[data-testid="stExpander"] { background:rgba(255,255,255,.78); border:1px solid #e1e3ec!important; border-radius:14px!important; overflow:hidden; }
[data-testid="stDataFrame"], [data-testid="stPlotlyChart"], [data-testid="stImage"] { background:#fff; border:1px solid #e1e3ec; border-radius:16px; overflow:hidden; box-shadow:0 8px 26px rgba(25,30,60,.055); }
[data-testid="stWidgetLabel"] p { color:#344054!important; font-weight:600; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color:#f8fafc!important; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] > [data-testid="stWidgetLabel"] p { color:#17202a!important; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] button,
[data-testid="stSidebar"] [data-testid="stFileUploader"] button span,
[data-testid="stSidebar"] [data-testid="stFileUploader"] button p { color:#17202a!important; }
[data-testid="stSidebar"] .stButton button { background:#343746!important; color:#f8fafc!important; border:1px solid #4b4f62!important; }
[data-testid="stSidebar"] .stButton button:hover { background:#414557!important; border-color:#7165e8!important; }
[data-testid="stSidebar"] .stButton button[kind="primary"] { background:linear-gradient(135deg,#6658ed,#8b5cf6)!important; border:0!important; }
[data-testid="stSidebar"] .stButton button p,
[data-testid="stSidebar"] .stButton button span { color:#f8fafc!important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] input { background:#303340!important; color:#f8fafc!important; border-color:#494d60!important; }
[data-testid="stSidebar"] hr { border-color:#3b3e4d!important; }
[data-testid="stAppViewContainer"] main [data-baseweb="select"] > div { background:#fff!important; color:#17202a!important; }
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] input {
  background:#ffffff!important;
  color:#111827!important;
  caret-color:#4f46e5!important;
  -webkit-text-fill-color:#111827!important;
}
[data-testid="stBottomBlockContainer"] { background:linear-gradient(180deg,rgba(246,247,251,0),rgba(246,247,251,.92) 30%,#f6f7fb 68%)!important; padding-top:2.25rem!important; backdrop-filter:blur(8px); }
[data-testid="stChatInput"] {
  background:#ffffff!important;
  border:1px solid #cfd3df!important;
  border-radius:20px!important;
  box-shadow:0 16px 46px rgba(17,19,29,.24),0 0 0 1px rgba(117,103,255,.10)!important;
  padding:.35rem .45rem .35rem .75rem!important;
  transition:border-color .2s ease,box-shadow .2s ease,transform .2s ease;
}
[data-testid="stChatInput"]:focus-within { border-color:#8b7fff!important; box-shadow:0 18px 52px rgba(17,19,29,.28),0 0 0 4px rgba(117,103,255,.16)!important; transform:translateY(-1px); }
[data-testid="stChatInput"] textarea { min-height:3rem!important; font-size:1rem!important; line-height:1.5!important; padding-top:.78rem!important; }
[data-testid="stChatInput"] button { width:42px!important; height:42px!important; border-radius:13px!important; background:linear-gradient(135deg,#6c5df2,#9b5cf4)!important; color:#fff!important; border:0!important; box-shadow:0 8px 20px rgba(108,93,242,.35)!important; }
[data-testid="stChatInput"] button:hover { transform:translateY(-1px) scale(1.03); filter:brightness(1.08); }
[data-testid="stChatInput"] button svg { fill:#fff!important; color:#fff!important; }
[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInput"] input::placeholder {
  color:#667085!important;
  -webkit-text-fill-color:#667085!important;
  opacity:1!important;
}
[data-testid="stElementToolbar"] {
  background:#ffffff!important;
  border:1px solid #d9dee8!important;
  border-radius:10px!important;
  box-shadow:0 4px 14px rgba(23,32,42,.10)!important;
}
[data-testid="stElementToolbar"] button,
button[aria-label="Fullscreen"],
button[aria-label="Download as CSV"],
button[aria-label="Show/hide columns"],
button[aria-label="Search"] {
  background:#ffffff!important;
  color:#17202a!important;
}
[data-testid="stElementToolbar"] button:hover,
button[aria-label="Fullscreen"]:hover,
button[aria-label="Download as CSV"]:hover,
button[aria-label="Show/hide columns"]:hover,
button[aria-label="Search"]:hover { background:#eeecff!important; }
[data-testid="stElementToolbar"] svg,
button[aria-label="Fullscreen"] svg,
button[aria-label="Download as CSV"] svg,
button[aria-label="Show/hide columns"] svg,
button[aria-label="Search"] svg {
  color:#17202a!important;
  fill:#17202a!important;
  stroke:#17202a!important;
}
.stButton>button { border-radius:13px; font-weight:700; transition:transform .18s ease,box-shadow .18s ease,background .18s ease; }
.stButton>button:hover { transform:translateY(-1px); box-shadow:0 8px 20px rgba(49,46,129,.14); }
@media (max-width:760px) {
  .hero h1 { font-size:2.35rem; }
  [data-testid="stMainBlockContainer"] { padding-left:1rem; padding-right:1rem; }
}
</style>
<div class="hero">
  <div class="eyebrow">Conversational analytics workspace</div>
  <h1>Ask your data. See the evidence.</h1>
  <p>Bring several datasets into one workspace, choose the sources for each question, and receive a clear explanation supported by computed tables and charts.</p>
</div>
""", unsafe_allow_html=True)

if "sessions" not in st.session_state:
    # Preserve work from the earlier single-workspace version when hot-reloading.
    first_id = f"session_{uuid4().hex[:8]}"
    st.session_state.sessions = {
        first_id: {
            "name": "Analysis 1",
            "datasets": st.session_state.pop("datasets", {}),
            "history": st.session_state.pop("history", []),
        }
    }
    st.session_state.active_session = first_id
if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = True


def create_session() -> str:
    session_id = f"session_{uuid4().hex[:8]}"
    number = len(st.session_state.sessions) + 1
    st.session_state.sessions[session_id] = {
        "name": f"Analysis {number}",
        "datasets": {},
        "history": [],
    }
    return session_id


sessions = st.session_state.sessions
if st.session_state.active_session not in sessions:
    st.session_state.active_session = next(iter(sessions))

if not st.session_state.sidebar_open:
    st.markdown('<style>[data-testid="stSidebar"]{display:none!important}</style>', unsafe_allow_html=True)
    if st.button("☰ Open workspace", key="open_workspace_panel", type="primary"):
        st.session_state.sidebar_open = True
        st.rerun()

with st.sidebar:
    st.markdown("""
    <div class="brand">
      <div class="brand-mark">N</div>
      <div><div class="brand-name">Northstar</div><div class="brand-sub">ANALYTICS STUDIO</div></div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("← Hide panel", key="hide_workspace_panel", width="stretch"):
        st.session_state.sidebar_open = False
        st.rerun()

    if st.button("＋ New analysis", type="primary", width="stretch"):
        st.session_state.active_session = create_session()
        st.rerun()

    selected_session = st.selectbox(
        "Your sessions",
        options=list(sessions),
        index=list(sessions).index(st.session_state.active_session),
        format_func=lambda item: sessions[item]["name"],
    )
    if selected_session != st.session_state.active_session:
        st.session_state.active_session = selected_session
        st.rerun()

    active_id = st.session_state.active_session
    active = sessions[active_id]
    session_actions = st.columns(2)
    if session_actions[0].button("Rename", width="stretch"):
        st.session_state.show_rename = not st.session_state.get("show_rename", False)
    if session_actions[1].button("Delete", width="stretch", disabled=len(sessions) == 1):
        del sessions[active_id]
        st.session_state.active_session = next(iter(sessions))
        st.rerun()
    if st.session_state.get("show_rename", False):
        new_name = st.text_input("Session name", value=active["name"], max_chars=40)
        if new_name.strip() and new_name.strip() != active["name"]:
            active["name"] = new_name.strip()

    st.divider()
    st.markdown("#### Data sources")
    if not os.getenv("GROQ_API_KEY"):
        key = st.text_input("Groq API key", type="password", help="Kept only for this app process.")
        if key:
            os.environ["GROQ_API_KEY"] = key
    uploaded = st.file_uploader(
        "Add datasets",
        type=["csv", "xlsx", "xls", "json", "parquet"],
        accept_multiple_files=True,
        help="CSV, Excel, JSON, or Parquet. Up to 25 MB and 500,000 rows per file.",
        key=f"uploads_{active_id}",
    )
    if uploaded:
        for item in uploaded:
            try:
                item.seek(0)
                dataset = load_dataset(item.name, item)
                active["datasets"][dataset.dataset_id] = dataset
            except Exception as exc:
                st.error(str(exc))

    st.caption("Rows stay in the local Python runtime. Only schema metadata and computed results are shared with the model.")
    if active["datasets"] and st.button("Clear session data", width="stretch"):
        active["datasets"] = {}
        active["history"] = []
        st.rerun()

active_id = st.session_state.active_session
active = sessions[active_id]
datasets: dict[str, Dataset] = active["datasets"]
st.markdown(f'<div class="session-pill">● {active["name"]}</div>', unsafe_allow_html=True)
if not datasets:
    st.markdown("""
    <div style="margin-top:1rem;background:#fff;border:1px dashed #c9c5ee;border-radius:20px;padding:2.4rem;text-align:center;box-shadow:0 12px 40px rgba(32,35,54,.04)">
      <div style="font-size:2rem;margin-bottom:.5rem">↗</div>
      <div style="font-family:Manrope,sans-serif;font-size:1.15rem;font-weight:800;color:#17202a">Add your first data source</div>
      <div style="color:#667085;margin-top:.35rem">Open the data panel and choose one or more files for this session.</div>
    </div>
    """, unsafe_allow_html=True)
    if not st.session_state.sidebar_open:
        empty_left, empty_action, empty_right = st.columns([1, 1.5, 1])
        with empty_action:
            if st.button("↗ Open data panel", type="primary", width="stretch", key="empty_upload_action"):
                st.session_state.sidebar_open = True
                st.rerun()
    st.stop()

total_rows = sum(len(item.frame) for item in datasets.values())
total_columns = sum(len(item.frame.columns) for item in datasets.values())
m1, m2, m3 = st.columns(3)
for column, label, value in [
    (m1, "Datasets", len(datasets)), (m2, "Rows available", f"{total_rows:,}"), (m3, "Columns mapped", f"{total_columns:,}")
]:
    column.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

st.subheader("Choose evidence")
selected_ids = st.multiselect(
    "Datasets for this question",
    options=list(datasets),
    default=list(datasets),
    format_func=lambda item: datasets[item].label,
    label_visibility="collapsed",
)

with st.expander("Inspect dataset schemas"):
    for dataset_id in selected_ids:
        dataset = datasets[dataset_id]
        schema = schema_for_agent(dataset)
        st.markdown(f"**{dataset.name}** — {schema['row_count']:,} rows")
        st.dataframe(
            [{"Column": c["name"], "Type": c["dtype"], "Missing": c["missing_count"], "Unique": c["unique_count"]} for c in schema["columns"]],
            hide_index=True,
            width="stretch",
        )

for entry in active["history"]:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(entry["answer"])
        for chart in entry["charts"]:
            render_chart(chart)
        for table in entry["tables"]:
            st.markdown(f"**{table['name']}**")
            st.dataframe(table["rows"], width="stretch", hide_index=True)

question = st.chat_input("Ask a question about the selected datasets…")
if question:
    if not selected_ids:
        st.warning("Choose at least one dataset before asking a question.")
        st.stop()
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        status = st.status("Understanding the question and planning the analysis…", expanded=True)
        try:
            status.write("Generating safe Python analysis from dataset schemas")
            response = analyze(question, [datasets[item] for item in selected_ids])
            status.write(f"Analysis completed in {response.attempts} code attempt(s)")
            status.update(label="Analysis complete", state="complete", expanded=False)
            st.markdown(response.answer)
            for chart in response.execution.chart_paths:
                render_chart(chart)
            for table in response.execution.tables:
                st.markdown(f"**{table['name']}**")
                st.dataframe(table["rows"], width="stretch", hide_index=True)
            with st.expander("View executed Python"):
                st.code(response.code, language="python")
            active["history"].append({
                "question": question,
                "answer": response.answer,
                "charts": response.execution.chart_paths,
                "tables": response.execution.tables,
            })
        except Exception as exc:
            status.update(label="Analysis could not be completed", state="error", expanded=False)
            st.error(str(exc))
            st.caption("Try narrowing the question, choosing fewer datasets, or checking that the required columns contain usable values.")
