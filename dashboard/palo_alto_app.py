import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Palo Alto Networks | Employee Engagement Intelligence",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0a0c10; color: #e8eaf0; }

[data-testid="stSidebar"] { background: #10131a !important; border-right: 1px solid #1c2030; }
[data-testid="stSidebar"] * { color: #c8cad4 !important; }
[data-testid="stSidebar"] label { color: #6b7280 !important; font-size: 11px !important;
    letter-spacing: 0.08em; text-transform: uppercase; }

.brand-header { padding: 24px 0 18px 0; border-bottom: 1px solid #1c2030; margin-bottom: 24px; }
.brand-title { font-family: 'Syne', sans-serif; font-size: 24px; font-weight: 800;
    color: #f5f6fa; letter-spacing: -0.4px; }
.brand-sub { font-size: 11px; color: #4b5268; letter-spacing: 0.12em;
    text-transform: uppercase; margin-top: 3px; }

.kpi-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 24px; }
.kpi-card { background: #10131a; border: 1px solid #1c2030; border-radius: 10px;
    padding: 18px 18px; position: relative; overflow: hidden; }
.kpi-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; }
.kpi-card.blue::before   { background: linear-gradient(90deg,#3b82f6,#60a5fa); }
.kpi-card.green::before  { background: linear-gradient(90deg,#10b981,#34d399); }
.kpi-card.amber::before  { background: linear-gradient(90deg,#f59e0b,#fbbf24); }
.kpi-card.rose::before   { background: linear-gradient(90deg,#f43f5e,#fb7185); }
.kpi-card.purple::before { background: linear-gradient(90deg,#8b5cf6,#a78bfa); }
.kpi-label { font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase;
    color: #4b5268; margin-bottom: 6px; }
.kpi-value { font-family: 'Syne', sans-serif; font-size: 26px; font-weight: 700; color: #f5f6fa; }
.kpi-delta { font-size: 11px; color: #6b7280; margin-top: 4px; }

.section-title { font-family: 'Syne', sans-serif; font-size: 14px; font-weight: 700;
    color: #f5f6fa; letter-spacing: 0.02em; margin: 20px 0 12px 0;
    display: flex; align-items: center; gap: 8px; }
.dot { width: 6px; height: 6px; border-radius: 50%;
    background: #3b82f6; display: inline-block; }

.alert-high   { background:#2d1216; color:#f87171; border:1px solid #7f1d1d;
    padding:10px 14px; border-radius:8px; margin:6px 0; font-size:13px; }
.alert-medium { background:#2d2007; color:#fbbf24; border:1px solid #78350f;
    padding:10px 14px; border-radius:8px; margin:6px 0; font-size:13px; }
.alert-low    { background:#052e16; color:#4ade80; border:1px solid #14532d;
    padding:10px 14px; border-radius:8px; margin:6px 0; font-size:13px; }

.stTabs [data-baseweb="tab-list"] { background:#10131a; border-radius:10px; padding:4px; gap:2px; }
.stTabs [data-baseweb="tab"] { background:transparent; color:#6b7280; border-radius:8px;
    padding:8px 20px; font-size:13px; font-family:'DM Sans',sans-serif; }
.stTabs [aria-selected="true"] { background:#1c2030 !important; color:#f5f6fa !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0a0c10; }
::-webkit-scrollbar-thumb { background: #1c2030; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Plot theme ─────────────────────────────────────────────────────────────────
BG    = "#10131a"
GRID  = "#1c2030"
FONT  = "#9ca3af"
AXIS  = "#374151"
PAL   = ["#3b82f6","#10b981","#f59e0b","#f43f5e","#8b5cf6","#06b6d4","#ec4899","#84cc16"]

def dark(fig, height=340, legend=False):
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="DM Sans", color=FONT, size=11),
        height=height,
        margin=dict(l=16, r=16, t=36, b=16),
        showlegend=legend,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis=dict(gridcolor=GRID, linecolor=AXIS, tickfont=dict(size=10)),
        yaxis=dict(gridcolor=GRID, linecolor=AXIS, tickfont=dict(size=10)),
    )
    return fig

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load():
    df = pd.read_csv("data/Palo Alto Networks.csv")
    df.columns = df.columns.str.lower()

    # Engagement index
    eng_cols = ["jobinvolvement","jobsatisfaction","environmentsatisfaction","relationshipsatisfaction"]
    df["engagement_index"] = df[eng_cols].mean(axis=1)

    # Burnout score & risk
    df["burnout_score"] = 0
    df.loc[df["overtime"] == "Yes",          "burnout_score"] += 1
    df.loc[df["worklifebalance"] <= 2,       "burnout_score"] += 1
    df["burnout_risk"] = df["burnout_score"].apply(
        lambda s: "Low" if s == 0 else ("Medium" if s == 1 else "High"))

    # Work-life balance index
    df["wlb_index"] = df["worklifebalance"].mean()

    # Satisfaction stability (std across satisfaction dims)
    df["satisfaction_stability"] = df[eng_cols].std(axis=1)

    # Workload stress indicator
    df["workload_stress"] = (
        (df["overtime"] == "Yes").astype(int) +
        (df["businesstravel"] == "Travel_Frequently").astype(int)
    )

    # Tenure group
    df["tenure_group"] = pd.cut(df["yearsatcompany"],
        bins=[0,2,5,10,40], labels=["0-2 yrs","3-5 yrs","6-10 yrs","10+ yrs"])

    # Commute type
    median_dist = df["distancefromhome"].median()
    df["commute_type"] = np.where(df["distancefromhome"] > median_dist, "Long", "Short")

    return df

df = load()

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 👥 Palo Alto Networks")
    st.markdown("---")

    st.markdown("**DEPARTMENT**")
    depts = ["All"] + sorted(df["department"].unique().tolist())
    sel_dept = st.selectbox("", depts, key="dept")

    st.markdown("**JOB ROLE**")
    roles = ["All"] + sorted(df["jobrole"].unique().tolist())
    sel_role = st.selectbox("", roles, key="role")

    st.markdown("**OVERTIME**")
    ot_opts = ["All", "Yes (Overtime Only)", "No (No Overtime)"]
    sel_ot = st.selectbox("", ot_opts, key="ot")

    st.markdown("**ENGAGEMENT THRESHOLD**")
    eng_thresh = st.slider("Flag employees below:", 1.0, 4.0, 2.5, step=0.1)

    st.markdown("**TENURE RANGE (years)**")
    tenure_range = st.slider("", 0, 40, (0, 40))

    st.markdown("---")
    st.caption("Palo Alto Networks\nEmployee Engagement Intelligence v1.0")

# ── Apply filters ──────────────────────────────────────────────────────────────
df_f = df.copy()
if sel_dept != "All":
    df_f = df_f[df_f["department"] == sel_dept]
if sel_role != "All":
    df_f = df_f[df_f["jobrole"] == sel_role]
if sel_ot == "Yes (Overtime Only)":
    df_f = df_f[df_f["overtime"] == "Yes"]
elif sel_ot == "No (No Overtime)":
    df_f = df_f[df_f["overtime"] == "No"]
df_f = df_f[
    (df_f["yearsatcompany"] >= tenure_range[0]) &
    (df_f["yearsatcompany"] <= tenure_range[1])
]

# ── KPI calculations ───────────────────────────────────────────────────────────
total_emp       = len(df_f)
attrition_rate  = df_f["attrition"].mean() * 100
avg_engagement  = df_f["engagement_index"].mean()
avg_burnout     = df_f["burnout_score"].mean()
high_risk_pct   = (df_f["burnout_risk"] == "High").mean() * 100
avg_wlb         = df_f["worklifebalance"].mean()
ot_attrition    = df_f[df_f["overtime"]=="Yes"]["attrition"].mean()*100
no_ot_attrition = df_f[df_f["overtime"]=="No"]["attrition"].mean()*100
low_eng_count   = (df_f["engagement_index"] < eng_thresh).sum()

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="brand-header">
  <div>
    <div class="brand-title">👥 Palo Alto Networks — Employee Engagement Intelligence</div>
    <div class="brand-sub">Workforce Analytics | Burnout Risk | Attrition Intelligence Dashboard</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI CARDS ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card blue">
    <div class="kpi-label">Avg Engagement Index</div>
    <div class="kpi-value">{avg_engagement:.2f}</div>
    <div class="kpi-delta">Out of 4.00 max score</div>
  </div>
  <div class="kpi-card rose">
    <div class="kpi-label">Attrition Rate</div>
    <div class="kpi-value">{attrition_rate:.1f}%</div>
    <div class="kpi-delta">{int(df_f['attrition'].sum())} of {total_emp} employees</div>
  </div>
  <div class="kpi-card amber">
    <div class="kpi-label">High Burnout Risk</div>
    <div class="kpi-value">{high_risk_pct:.1f}%</div>
    <div class="kpi-delta">{int((df_f['burnout_risk']=='High').sum())} employees flagged</div>
  </div>
  <div class="kpi-card green">
    <div class="kpi-label">Avg Work-Life Balance</div>
    <div class="kpi-value">{avg_wlb:.2f}</div>
    <div class="kpi-delta">Out of 4.00 scale</div>
  </div>
  <div class="kpi-card purple">
    <div class="kpi-label">Low Engagement Alerts</div>
    <div class="kpi-value">{low_eng_count}</div>
    <div class="kpi-delta">Below {eng_thresh:.1f} threshold</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Engagement Health",
    "🔥 Burnout Risk",
    "🏢 Role & Career Stage",
    "⚡ Manager Action Panel"
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — ENGAGEMENT HEALTH OVERVIEW
# ══════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown('<div class="section-title"><span class="dot"></span>Engagement Index Distribution</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Histogram(
            x=df_f["engagement_index"], nbinsx=16,
            marker=dict(color="#3b82f6", line=dict(color=BG, width=1)),
        ))
        fig.add_vline(x=avg_engagement, line_dash="dash", line_color="#f59e0b",
                      annotation_text=f"Avg: {avg_engagement:.2f}",
                      annotation_font_color="#f59e0b", annotation_font_size=11)
        fig.add_vline(x=eng_thresh, line_dash="dot", line_color="#f43f5e",
                      annotation_text=f"Threshold: {eng_thresh}",
                      annotation_font_color="#f43f5e", annotation_font_size=11)
        dark(fig, height=300)
        fig.update_layout(xaxis_title="Engagement Index", yaxis_title="Employee Count")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title"><span class="dot"></span>Satisfaction Dimensions</div>', unsafe_allow_html=True)
        sat_dims = {
            "Job Satisfaction":          df_f["jobsatisfaction"].mean(),
            "Environment Satisfaction":  df_f["environmentsatisfaction"].mean(),
            "Job Involvement":           df_f["jobinvolvement"].mean(),
            "Relationship Satisfaction": df_f["relationshipsatisfaction"].mean(),
            "Work-Life Balance":         df_f["worklifebalance"].mean(),
        }
        sat_df = pd.DataFrame(list(sat_dims.items()), columns=["Dimension","Score"])
        colors = ["#10b981" if s >= 2.8 else "#f59e0b" if s >= 2.4 else "#f43f5e" for s in sat_df["Score"]]
        fig2 = go.Figure(go.Bar(
            x=sat_df["Score"], y=sat_df["Dimension"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=sat_df["Score"].apply(lambda v: f"{v:.2f}"),
            textposition="outside", textfont=dict(size=11, color="#9ca3af"),
        ))
        dark(fig2, height=300)
        fig2.update_layout(xaxis_title="Average Score (1-4)", yaxis_title="",
                           xaxis_range=[0, 4.2])
        fig2.add_vline(x=2.5, line_dash="dash", line_color="#374151")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2, gap="medium")

    with col3:
        st.markdown('<div class="section-title"><span class="dot"></span>Engagement by Department</div>', unsafe_allow_html=True)
        dept_eng = df_f.groupby("department")["engagement_index"].mean().reset_index()
        dept_att = df_f.groupby("department")["attrition"].mean().reset_index()
        dept_df  = dept_eng.merge(dept_att, on="department")
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Bar(
            x=dept_df["department"], y=dept_df["engagement_index"],
            name="Engagement", marker_color="#3b82f6",
            text=dept_df["engagement_index"].apply(lambda v: f"{v:.2f}"),
            textposition="outside", textfont=dict(size=10)
        ), secondary_y=False)
        fig3.add_trace(go.Scatter(
            x=dept_df["department"], y=dept_df["attrition"]*100,
            name="Attrition %", mode="lines+markers",
            line=dict(color="#f43f5e", width=2), marker=dict(size=7)
        ), secondary_y=True)
        fig3.update_layout(paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(family="DM Sans", color=FONT, size=11),
            height=300, margin=dict(l=16,r=16,t=36,b=16),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor=GRID, tickfont=dict(size=10)),
            yaxis=dict(gridcolor=GRID, title="Engagement Index"),
            yaxis2=dict(title="Attrition %", showgrid=False, tickformat=".0f"))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-title"><span class="dot"></span>Engagement vs Attrition — Boxplot</div>', unsafe_allow_html=True)
        df_f["attrition_label"] = df_f["attrition"].map({0:"Retained",1:"Churned"})
        fig4 = go.Figure()
        for label, color in [("Retained","#10b981"),("Churned","#f43f5e")]:
            sub = df_f[df_f["attrition_label"]==label]["engagement_index"]
            fig4.add_trace(go.Box(
                y=sub, name=label, marker_color=color,
                line=dict(color=color), fillcolor="rgba(59,130,246,0.15)" if color =="#3b82f6" else "rgba(244,63,94,0.15)",
                boxmean=True
            ))
        dark(fig4, height=300, legend=True)
        fig4.update_layout(yaxis_title="Engagement Index")
        st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2 — BURNOUT RISK DASHBOARD
# ══════════════════════════════════════════════════════════════════
with tab2:
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown('<div class="section-title"><span class="dot"></span>Burnout Risk Distribution</div>', unsafe_allow_html=True)
        risk_counts = df_f["burnout_risk"].value_counts().reindex(["Low","Medium","High"]).fillna(0)
        colors_risk = {"Low":"#10b981","Medium":"#f59e0b","High":"#f43f5e"}
        fig = go.Figure(go.Bar(
            x=risk_counts.index,
            y=risk_counts.values,
            marker_color=[colors_risk[r] for r in risk_counts.index],
            text=risk_counts.values.astype(int),
            textposition="outside", textfont=dict(size=12, color="#f5f6fa"),
        ))
        dark(fig, height=300)
        fig.update_layout(xaxis_title="Risk Level", yaxis_title="Employee Count")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title"><span class="dot"></span>Overtime Impact on Attrition</div>', unsafe_allow_html=True)
        ot_df = df_f.groupby("overtime")["attrition"].mean().reset_index()
        ot_df["attrition_pct"] = ot_df["attrition"] * 100
        bar_colors = ["#10b981" if v < 15 else "#f59e0b" if v < 25 else "#f43f5e"
                      for v in ot_df["attrition_pct"]]
        fig2 = go.Figure(go.Bar(
            x=ot_df["overtime"], y=ot_df["attrition_pct"],
            marker_color=bar_colors,
            text=ot_df["attrition_pct"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside", textfont=dict(size=13, color="#f5f6fa"),
        ))
        dark(fig2, height=300)
        fig2.update_layout(xaxis_title="Overtime Status", yaxis_title="Attrition Rate (%)")
        fig2.add_hline(y=attrition_rate, line_dash="dash", line_color="#6b7280",
                       annotation_text=f"Avg: {attrition_rate:.1f}%",
                       annotation_font_size=10, annotation_font_color="#6b7280")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2, gap="medium")

    with col3:
        st.markdown('<div class="section-title"><span class="dot"></span>Work-Life Balance vs Burnout</div>', unsafe_allow_html=True)
        wlb_df = df_f.groupby("worklifebalance")["burnout_score"].mean().reset_index()
        wlb_df["wlb_label"] = wlb_df["worklifebalance"].map(
            {1:"Poor",2:"Fair",3:"Good",4:"Excellent"})
        fig3 = go.Figure(go.Bar(
            x=wlb_df["wlb_label"], y=wlb_df["burnout_score"],
            marker=dict(
                color=wlb_df["burnout_score"],
                colorscale=[[0,"#10b981"],[0.5,"#f59e0b"],[1,"#f43f5e"]],
            ),
            text=wlb_df["burnout_score"].apply(lambda v: f"{v:.2f}"),
            textposition="outside", textfont=dict(size=11, color="#9ca3af"),
        ))
        dark(fig3, height=300)
        fig3.update_layout(xaxis_title="Work-Life Balance Rating", yaxis_title="Avg Burnout Score")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-title"><span class="dot"></span>Business Travel vs Burnout Score</div>', unsafe_allow_html=True)
        travel_df = df_f.groupby("businesstravel")[["burnout_score","attrition"]].mean().reset_index()
        travel_df["attrition_pct"] = travel_df["attrition"] * 100
        fig4 = make_subplots(specs=[[{"secondary_y": True}]])
        fig4.add_trace(go.Bar(
            x=travel_df["businesstravel"], y=travel_df["burnout_score"],
            name="Avg Burnout", marker_color="#f59e0b",
            text=travel_df["burnout_score"].apply(lambda v: f"{v:.2f}"),
            textposition="outside", textfont=dict(size=10)
        ), secondary_y=False)
        fig4.add_trace(go.Scatter(
            x=travel_df["businesstravel"], y=travel_df["attrition_pct"],
            name="Attrition %", mode="lines+markers",
            line=dict(color="#f43f5e", width=2), marker=dict(size=7)
        ), secondary_y=True)
        fig4.update_layout(paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(family="DM Sans", color=FONT, size=11),
            height=300, margin=dict(l=16,r=16,t=36,b=16),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor=GRID, tickfont=dict(size=10)),
            yaxis=dict(gridcolor=GRID, title="Avg Burnout Score"),
            yaxis2=dict(title="Attrition %", showgrid=False))
        st.plotly_chart(fig4, use_container_width=True)

    # Burnout by department heatmap
    st.markdown('<div class="section-title"><span class="dot"></span>High Burnout Risk % by Department & Overtime</div>', unsafe_allow_html=True)
    heat_df = df_f.groupby(["department","overtime"])["burnout_score"].mean().reset_index()
    heat_pivot = heat_df.pivot(index="department", columns="overtime", values="burnout_score")
    fig5 = go.Figure(go.Heatmap(
        z=heat_pivot.values,
        x=heat_pivot.columns.tolist(),
        y=heat_pivot.index.tolist(),
        colorscale=[[0,"#052e16"],[0.5,"#f59e0b"],[1,"#7f1d1d"]],
        text=[[f"{v:.2f}" for v in row] for row in heat_pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=12, color="white"),
        showscale=True,
        colorbar=dict(tickfont=dict(color=FONT))
    ))
    dark(fig5, height=260)
    fig5.update_layout(xaxis_title="Overtime", yaxis_title="Department")
    st.plotly_chart(fig5, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3 — ROLE & CAREER STAGE
# ══════════════════════════════════════════════════════════════════
with tab3:
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown('<div class="section-title"><span class="dot"></span>Engagement by Job Role</div>', unsafe_allow_html=True)
        role_df = df_f.groupby("jobrole")["engagement_index"].mean().reset_index().sort_values("engagement_index")
        colors_r = ["#f43f5e" if v < eng_thresh else "#10b981" for v in role_df["engagement_index"]]
        fig = go.Figure(go.Bar(
            x=role_df["engagement_index"], y=role_df["jobrole"],
            orientation="h",
            marker=dict(color=colors_r, line=dict(width=0)),
            text=role_df["engagement_index"].apply(lambda v: f"{v:.2f}"),
            textposition="outside", textfont=dict(size=10, color="#9ca3af"),
        ))
        dark(fig, height=360)
        fig.update_layout(xaxis_title="Avg Engagement Index", yaxis_title="",
                          xaxis_range=[0, 4.2])
        fig.add_vline(x=eng_thresh, line_dash="dot", line_color="#f43f5e",
                      annotation_text="Threshold", annotation_font_size=9,
                      annotation_font_color="#f43f5e")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title"><span class="dot"></span>Engagement by Job Level</div>', unsafe_allow_html=True)
        level_df = df_f.groupby("joblevel")[["engagement_index","attrition"]].mean().reset_index()
        level_df["attrition_pct"] = level_df["attrition"] * 100
        level_df["level_label"] = level_df["joblevel"].apply(lambda l: f"Level {l}")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=level_df["level_label"], y=level_df["engagement_index"],
            mode="lines+markers",
            line=dict(color="#3b82f6", width=2),
            marker=dict(size=10, color="#3b82f6"),
            name="Engagement", fill="tozeroy",
            fillcolor="rgba(59,130,246,0.08)"
        ))
        dark(fig2, height=360, legend=True)
        fig2.update_layout(yaxis_title="Avg Engagement Index",
                           yaxis_range=[2.0, 3.2])
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2, gap="medium")

    with col3:
        st.markdown('<div class="section-title"><span class="dot"></span>Tenure vs Engagement Trend</div>', unsafe_allow_html=True)
        ten_df = df_f.groupby("tenure_group", observed=False)["engagement_index"].mean().reset_index()
        fig3 = go.Figure(go.Bar(
            x=ten_df["tenure_group"].astype(str), y=ten_df["engagement_index"],
            marker=dict(
                color=ten_df["engagement_index"],
                colorscale=[[0,"#1e3a5f"],[0.5,"#3b82f6"],[1,"#93c5fd"]],
            ),
            text=ten_df["engagement_index"].apply(lambda v: f"{v:.3f}"),
            textposition="outside", textfont=dict(size=11, color="#9ca3af"),
        ))
        dark(fig3, height=300)
        fig3.update_layout(xaxis_title="Tenure Group", yaxis_title="Avg Engagement",
                           yaxis_range=[2.5, 3.0])
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-title"><span class="dot"></span>Attrition Rate by Tenure Group</div>', unsafe_allow_html=True)
        ten_att = df_f.groupby("tenure_group", observed=False)["attrition"].mean().reset_index()
        ten_att["attrition_pct"] = ten_att["attrition"] * 100
        bar_colors = ["#f43f5e" if v > 20 else "#f59e0b" if v > 12 else "#10b981"
                      for v in ten_att["attrition_pct"]]
        fig4 = go.Figure(go.Bar(
            x=ten_att["tenure_group"].astype(str), y=ten_att["attrition_pct"],
            marker_color=bar_colors,
            text=ten_att["attrition_pct"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside", textfont=dict(size=12, color="#f5f6fa"),
        ))
        dark(fig4, height=300)
        fig4.update_layout(xaxis_title="Tenure Group", yaxis_title="Attrition Rate (%)")
        fig4.add_hline(y=attrition_rate, line_dash="dash", line_color="#6b7280",
                       annotation_text=f"Avg {attrition_rate:.1f}%", annotation_font_size=10)
        st.plotly_chart(fig4, use_container_width=True)

    # Role attrition table
    st.markdown('<div class="section-title"><span class="dot"></span>Role-Level Engagement & Attrition Summary</div>', unsafe_allow_html=True)
    role_summary = df_f.groupby("jobrole").agg(
        employees   = ("attrition","count"),
        avg_engagement = ("engagement_index","mean"),
        avg_burnout = ("burnout_score","mean"),
        attrition_rate = ("attrition","mean"),
    ).reset_index().sort_values("attrition_rate", ascending=False)
    role_summary["attrition_rate"] = (role_summary["attrition_rate"]*100).round(1).astype(str)+"%"
    role_summary["avg_engagement"] = role_summary["avg_engagement"].round(2)
    role_summary["avg_burnout"]    = role_summary["avg_burnout"].round(2)
    role_summary.columns = ["Job Role","Employees","Avg Engagement","Avg Burnout","Attrition Rate"]
    st.dataframe(role_summary.reset_index(drop=True), use_container_width=True, height=300)

# ══════════════════════════════════════════════════════════════════
# TAB 4 — MANAGER ACTION PANEL
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title"><span class="dot"></span>⚠️ Priority Intervention Areas</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        st.metric("Overtime Attrition Rate",   f"{ot_attrition:.1f}%",
                  f"+{ot_attrition-no_ot_attrition:.1f}pp vs non-overtime")
    with col2:
        st.metric("High Burnout Employees",
                  int((df_f["burnout_risk"]=="High").sum()),
                  f"{high_risk_pct:.1f}% of workforce")
    with col3:
        st.metric("Below Engagement Threshold",
                  low_eng_count,
                  f"Threshold: {eng_thresh:.1f}")

    st.markdown("---")

    # Low engagement alerts
    st.markdown('<div class="section-title"><span class="dot"></span>Low-Engagement Employee Alerts</div>', unsafe_allow_html=True)

    low_eng = df_f[df_f["engagement_index"] < eng_thresh].copy()
    low_eng_summary = low_eng.groupby(["department","jobrole"]).agg(
        count        = ("engagement_index","count"),
        avg_eng      = ("engagement_index","mean"),
        avg_burnout  = ("burnout_score","mean"),
        attrition_rt = ("attrition","mean"),
    ).reset_index().sort_values("avg_eng")

    for _, row in low_eng_summary.iterrows():
        level = "🔴 HIGH" if row["attrition_rt"] > 0.25 else "🟡 WATCH" if row["attrition_rt"] > 0.12 else "🟢 LOW"
        css   = "alert-high" if "HIGH" in level else "alert-medium" if "WATCH" in level else "alert-low"
        st.markdown(f"""
        <div class="{css}">
            <strong>{row['department']} — {row['jobrole']}</strong> &nbsp;|&nbsp;
            {int(row['count'])} employees &nbsp;|&nbsp;
            Avg Engagement: <strong>{row['avg_eng']:.2f}</strong> &nbsp;|&nbsp;
            Burnout: {row['avg_burnout']:.2f} &nbsp;|&nbsp;
            Attrition: {row['attrition_rt']*100:.0f}% &nbsp;|&nbsp; {level}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_a, col_b = st.columns(2, gap="medium")

    with col_a:
        st.markdown('<div class="section-title"><span class="dot"></span>Workload Stress by Department</div>', unsafe_allow_html=True)
        stress_df = df_f.groupby("department")[["workload_stress","burnout_score","engagement_index"]].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Workload Stress", x=stress_df["department"],
                             y=stress_df["workload_stress"], marker_color="#f59e0b"))
        fig.add_trace(go.Bar(name="Burnout Score",   x=stress_df["department"],
                             y=stress_df["burnout_score"], marker_color="#f43f5e"))
        fig.update_layout(barmode="group")
        dark(fig, height=300, legend=True)
        fig.update_layout(yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title"><span class="dot"></span>Satisfaction Stability by Role</div>', unsafe_allow_html=True)
        stab_df = df_f.groupby("jobrole")["satisfaction_stability"].mean().reset_index().sort_values("satisfaction_stability", ascending=False)
        fig2 = go.Figure(go.Bar(
            x=stab_df["jobrole"], y=stab_df["satisfaction_stability"],
            marker=dict(
                color=stab_df["satisfaction_stability"],
                colorscale=[[0,"#10b981"],[0.5,"#f59e0b"],[1,"#f43f5e"]],
            ),
            text=stab_df["satisfaction_stability"].apply(lambda v: f"{v:.2f}"),
            textposition="outside", textfont=dict(size=10),
        ))
        dark(fig2, height=300)
        fig2.update_layout(xaxis_tickangle=-35,
                           yaxis_title="Satisfaction Instability (higher = more volatile)")
        st.plotly_chart(fig2, use_container_width=True)

    # Recommended actions
    st.markdown("---")
    st.markdown('<div class="section-title"><span class="dot"></span>Recommended Actions for Management</div>', unsafe_allow_html=True)
    top_risk_dept = df_f.groupby("department")["attrition"].mean().idxmax()
    top_risk_role = df_f.groupby("jobrole")["attrition"].mean().idxmax()

    actions = [
        ("🔴 Critical", f"Overtime workers have {ot_attrition:.0f}% attrition vs {no_ot_attrition:.0f}% — implement mandatory overtime limits or compensation review"),
        ("🔴 Critical", f"{int((df_f['burnout_risk']=='High').sum())} employees are High Burnout Risk — prioritise 1:1 manager check-ins and workload redistribution"),
        ("🟡 High",     f"{top_risk_dept} department has highest attrition — schedule department health review and exit interview analysis"),
        ("🟡 High",     f"{top_risk_role} role has highest attrition — review compensation benchmarks and career progression pathways"),
        ("🟢 Medium",   f"{low_eng_count} employees below engagement threshold {eng_thresh:.1f} — deploy pulse survey and recognition programme"),
        ("🟢 Medium",   "Frequent business travellers show highest burnout scores — consider travel policy review and post-travel recovery days"),
    ]
    for priority, action in actions:
        css = "alert-high" if "Critical" in priority else "alert-medium" if "High" in priority else "alert-low"
        st.markdown(f'<div class="{css}"><strong>{priority}</strong> — {action}</div>', unsafe_allow_html=True)
