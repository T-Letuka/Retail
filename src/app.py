
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="RetailLens",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Colour palette ────────────────────────────────────────
C = {
    'navy':      '#001d3d',
    'accent':    '#003566',
    'highlight': '#d00000',
    'green':     '#80b918',
    'neutral':   '#ede7e3',
    'orange':    '#ff5400',
}

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 4.5rem;
        font-weight: 800;
        color: #1B3A5C;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1rem;
        color: #95A5A6;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #F8F9FA;
        border-left: 4px solid #2E75B6;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1B3A5C;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #95A5A6;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .finding-card {
        background: #EBF5FB;
        border-left: 4px solid #2E75B6;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
    }
    .warning-card {
        background: #FDEDEC;
        border-left: 4px solid #E74C3C;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
    }
    .success-card {
        background: #EAFAF1;
        border-left: 4px solid #2ECC71;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1B3A5C;
        border-bottom: 2px solid #2E75B6;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }
    div[data-testid="stSidebar"] {
        background-color: #1B3A5C;
    }
    div[data-testid="stSidebar"] * {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────
# Use st.cache_data so data loads once and is reused
# across reruns — essential for dashboard performance

@st.cache_data
def load_data():
    """Load all processed datasets."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    master = pd.read_csv(
        os.path.join(base, 'data/processed/master.csv'),
        parse_dates=[
            'order_purchase_timestamp',
            'order_delivered_customer_date',
            'order_estimated_delivery_date'
        ]
    )

    rfm = pd.read_csv(
        os.path.join(base, 'data/processed/rfm_scored.csv'),
        parse_dates=['first_order', 'last_order']
    )

    churn = pd.read_csv(
        os.path.join(base, 'data/processed/churn_features.csv')
    )

    return master, rfm, churn


@st.cache_resource
def load_models():
    """Load serialised models — cached as resource (not reloaded on rerun)."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model  = joblib.load(os.path.join(base, 'outputs/models/churn_model.pkl'))
    scaler = joblib.load(os.path.join(base, 'outputs/models/scaler.pkl'))

    return model, scaler


# ── Load everything ───────────────────────────────────────
try:
    master, rfm, churn_df = load_data()
    model, scaler = load_models()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Make sure you have run all notebooks and that processed files exist.")
    data_loaded = False
    st.stop()


# ── Sidebar navigation ────────────────────────────────────
st.sidebar.markdown("##  Retail")
st.sidebar.markdown("*From Transactions to Intelligence*")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        " Business Overview",
        " Customer Segments",
        " Statistical Findings",
        " Churn Predictor"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset**")
st.sidebar.markdown("Brazilian E-Commerce by Olist")
st.sidebar.markdown("100,000 orders | 2017–2018")
st.sidebar.markdown("---")
st.sidebar.markdown("**Built by**")
st.sidebar.markdown("Tisetso Letuka")
st.sidebar.markdown("*Data Science Portfolio*")


# ============================================================
# PAGE 1 — BUSINESS OVERVIEW
# ============================================================

if page == " Business Overview":

    st.markdown('<p class="main-title"> Business Overview</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Data Analysis  |  '
                'What is happening in this business?</p>',
                unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────
    orders_deduped = master.drop_duplicates(subset='order_id')
    delivered      = orders_deduped[orders_deduped['order_status'] == 'delivered']

    total_revenue  = orders_deduped['payment_value'].sum()
    total_orders   = orders_deduped['order_id'].nunique()
    total_customers= master['customer_unique_id'].nunique()
    mean_review    = orders_deduped['review_score'].mean()
    late_rate      = delivered['was_late'].mean() * 100 if 'was_late' in delivered.columns else (
        (delivered['order_delivered_customer_date'] >
         delivered['order_estimated_delivery_date']).mean() * 100
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Revenue", f"R${total_revenue/1e6:.1f}M")
    with col2:
        st.metric("Total Orders", f"{total_orders:,}")
    with col3:
        st.metric("Unique Customers", f"{total_customers:,}")
    with col4:
        st.metric("Avg Review Score", f"{mean_review:.2f}/5.00")
    with col5:
        st.metric("Late Delivery Rate", f"{late_rate:.1f}%",
                  delta=f"-{100-late_rate:.1f}% on time",
                  delta_color="inverse")

    st.markdown("---")

    # ── Revenue trend ──────────────────────────────────────
    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.markdown('<p class="section-header">Monthly Revenue Trend</p>',
                    unsafe_allow_html=True)

        master['order_month'] = master['order_purchase_timestamp'].dt.to_period('M')

        # Complete months only
        days_per_month = (
            master.groupby('order_month')['order_purchase_timestamp']
            .apply(lambda x: x.dt.day.nunique())
            .reset_index(name='days')
        )
        complete = days_per_month[days_per_month['days'] >= 27]['order_month']

        monthly = (
            master[master['order_month'].isin(complete.values)]
            .groupby('order_month')['payment_value']
            .sum()
            .reset_index()
        )
        monthly['month_dt']  = monthly['order_month'].dt.to_timestamp()
        monthly['mom_growth'] = monthly['payment_value'].pct_change() * 100

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.05
        )

        fig.add_trace(go.Scatter(
            x=monthly['month_dt'],
            y=monthly['payment_value'] / 1e6,
            mode='lines+markers',
            line=dict(color=C['navy'], width=2.5),
            fill='tozeroy',
            fillcolor='rgba(46, 117, 182, 0.1)',
            name='Revenue',
            hovertemplate='%{x|%b %Y}<br>R$%{y:.2f}M<extra></extra>'
        ), row=1, col=1)

        colours_mom = [
            C['accent'] if v >= 0 else C['highlight']
            for v in monthly['mom_growth'].fillna(0)
        ]
        fig.add_trace(go.Bar(
            x=monthly['month_dt'],
            y=monthly['mom_growth'].fillna(0),
            marker_color=colours_mom,
            name='MoM Growth %',
            hovertemplate='%{x|%b %Y}<br>%{y:.1f}%<extra></extra>'
        ), row=2, col=1)

        fig.update_layout(
            height=420,
            showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        fig.update_yaxes(title_text="R$ Millions", row=1, col=1,
                         tickformat=".1f", tickprefix="R$", ticksuffix="M")
        fig.update_yaxes(title_text="MoM %", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<p class="section-header">Key Numbers</p>',
                    unsafe_allow_html=True)

        peak_month = monthly.loc[monthly['payment_value'].idxmax(), 'order_month']
        peak_val   = monthly['payment_value'].max() / 1e6

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Peak Month</div>
            <div class="metric-value">{peak_month}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Peak Revenue</div>
            <div class="metric-value">R${peak_val:.1f}M</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Analysis Period</div>
            <div class="metric-value">20 months</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Max MoM Growth</div>
            <div class="metric-value">{monthly['mom_growth'].max():.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Max MoM Decline</div>
            <div class="metric-value">{monthly['mom_growth'].min():.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Product categories + Delivery ─────────────────────
    col_cat, col_del = st.columns(2)

    with col_cat:
        st.markdown('<p class="section-header">Top Categories by Revenue</p>',
                    unsafe_allow_html=True)

        cat_rev = (
            master.groupby('product_category_name_english')['payment_value']
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig_cat = px.bar(
            cat_rev,
            x='payment_value',
            y='product_category_name_english',
            orientation='h',
            color='payment_value',
            color_continuous_scale=['#D6E4F0', '#1B3A5C'],
            labels={'payment_value': 'Revenue (R$)',
                    'product_category_name_english': ''},
        )
        fig_cat.update_layout(
            height=350,
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        fig_cat.update_xaxes(tickformat=".0f", tickprefix="R$")
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_del:
        st.markdown('<p class="section-header">Delivery Performance</p>',
                    unsafe_allow_html=True)

        delivered_calc = (
            master[master['order_status'] == 'delivered']
            .drop_duplicates(subset='order_id')
            .copy()
        )
        delivered_calc['delay'] = (
            delivered_calc['order_delivered_customer_date'] -
            delivered_calc['order_estimated_delivery_date']
        ).dt.days

        on_time_pct = (delivered_calc['delay'] <= 0).mean() * 100
        late_pct    = 100 - on_time_pct
        mean_delay  = delivered_calc['delay'].mean()

        fig_del = go.Figure(go.Pie(
            values=[on_time_pct, late_pct],
            labels=['On Time / Early', 'Late'],
            hole=0.55,
            marker_colors=[C['accent'], C['highlight']],
            textinfo='label+percent',
            hovertemplate='%{label}: %{percent}<extra></extra>'
        ))
        fig_del.add_annotation(
            text=f"{delivered_calc['order_id'].nunique():,}<br>orders",
            x=0.5, y=0.5,
            font_size=13,
            showarrow=False
        )
        fig_del.update_layout(
            height=350,
            showlegend=False,
            margin=dict(l=0, r=0, t=10, b=10)
        )
        st.plotly_chart(fig_del, use_container_width=True)

        st.markdown(f"""
        <div class="success-card">
            Mean delivery delay: <b>{mean_delay:.1f} days</b>
            (negative = arrives early on average)
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PAGE 2 — CUSTOMER SEGMENTS
# ============================================================

elif page == " Customer Segments":

    st.markdown('<p class="main-title"> Customer Segments</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-title">RFM Segmentation  |  '
                'Who are the customers?</p>',
                unsafe_allow_html=True)

    # ── Segment profiles ──────────────────────────────────
    if 'segment' not in rfm.columns:
        st.warning("Segment column not found in rfm_scored.csv. "
                   "Rerun Notebook 01 Analysis 5.")
        st.stop()

    seg_profile = (
        rfm.groupby('segment')
        .agg(
            customers   = ('customer_unique_id', 'count'),
            avg_recency = ('recency_days', 'mean'),
            avg_orders  = ('total_orders', 'mean'),
            avg_spend   = ('total_spend', 'mean'),
            total_rev   = ('total_spend', 'sum'),
        )
        .reset_index()
    )
    seg_profile['pct_customers'] = (
        seg_profile['customers'] / seg_profile['customers'].sum() * 100
    )
    seg_profile['pct_revenue'] = (
        seg_profile['total_rev'] / seg_profile['total_rev'].sum() * 100
    )

    seg_colours = {
        'Champions':           '#1B3A5C',
        'Loyal Customers':     '#2E75B6',
        'Potential Loyalists': '#5DADE2',
        'New Customers':       '#2ECC71',
        'At Risk':             '#E67E22',
        'Lost':                '#81171b',
    }

    # ── Segment selector ──────────────────────────────────
    selected_seg = st.selectbox(
        "Select a segment to explore",
        options=seg_profile['segment'].tolist(),
        index=0
    )

    seg_data = seg_profile[seg_profile['segment'] == selected_seg].iloc[0]
    seg_customers = rfm[rfm['segment'] == selected_seg]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Customers", f"{int(seg_data['customers']):,}")
    with col2:
        st.metric("% of Total", f"{seg_data['pct_customers']:.1f}%")
    with col3:
        st.metric("% of Revenue", f"{seg_data['pct_revenue']:.1f}%")
    with col4:
        st.metric("Avg Spend", f"R${seg_data['avg_spend']:.0f}")

    st.markdown("---")

    # ── Customer vs revenue share ─────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<p class="section-header">Customer vs Revenue Share</p>',
                    unsafe_allow_html=True)

        seg_sorted = seg_profile.sort_values('total_rev', ascending=False)

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name='% Customers',
            x=seg_sorted['segment'],
            y=seg_sorted['pct_customers'],
            marker_color=C['navy'],
            opacity=0.85,
            text=seg_sorted['pct_customers'].round(1).astype(str) + '%',
            textposition='outside'
        ))
        fig_bar.add_trace(go.Bar(
            name='% Revenue',
            x=seg_sorted['segment'],
            y=seg_sorted['pct_revenue'],
            marker_color=C['accent'],
            opacity=0.85,
            text=seg_sorted['pct_revenue'].round(1).astype(str) + '%',
            textposition='outside'
        ))
        fig_bar.update_layout(
            barmode='group',
            height=350,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(orientation='h', y=1.1)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.markdown('<p class="section-header">Segment Profiles</p>',
                    unsafe_allow_html=True)

        display_df = seg_profile[[
            'segment', 'customers', 'pct_customers',
            'pct_revenue', 'avg_spend', 'avg_recency'
        ]].sort_values('pct_revenue', ascending=False).copy()

        display_df.columns = [
            'Segment', 'Customers', '% Customers',
            '% Revenue', 'Avg Spend', 'Avg Recency (days)'
        ]
        display_df['% Customers'] = display_df['% Customers'].round(1)
        display_df['% Revenue']   = display_df['% Revenue'].round(1)
        display_df['Avg Spend']   = display_df['Avg Spend'].round(0)
        display_df['Avg Recency (days)'] = display_df['Avg Recency (days)'].round(0)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    # ── RFM scatter ───────────────────────────────────────
    st.markdown('<p class="section-header">Customer Map — '
                'Recency vs Total Spend (coloured by segment)</p>',
                unsafe_allow_html=True)

    sample = rfm.sample(min(3000, len(rfm)), random_state=42)

    fig_scatter = px.scatter(
        sample,
        x='recency_days',
        y='total_spend',
        color='segment',
        color_discrete_map=seg_colours,
        opacity=0.5,
        log_y=True,
        labels={
            'recency_days': 'Days Since Last Order',
            'total_spend':  'Total Spend (R$, log scale)',
            'segment':      'Segment'
        },
        hover_data=['total_orders'],
        height=400
    )
    fig_scatter.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# ============================================================
# PAGE 3 — STATISTICAL FINDINGS
# ============================================================

elif page == " Statistical Findings":

    st.markdown('<p class="main-title"> Statistical Findings</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Data Science  |  '
                'What patterns are statistically real?</p>',
                unsafe_allow_html=True)

    # ── H1 Finding ────────────────────────────────────────
    st.markdown('<p class="section-header">Hypothesis 1  '
                'Delivery Delay and Customer Satisfaction</p>',
                unsafe_allow_html=True)

    delivered_stats = (
        master[master['order_status'] == 'delivered']
        .drop_duplicates(subset='order_id')
        .dropna(subset=['review_score'])
        .copy()
    )
    delivered_stats['delay'] = (
        delivered_stats['order_delivered_customer_date'] -
        delivered_stats['order_estimated_delivery_date']
    ).dt.days
    delivered_stats['was_late'] = delivered_stats['delay'] > 0

    on_time_mean = delivered_stats[~delivered_stats['was_late']]['review_score'].mean()
    late_mean    = delivered_stats[ delivered_stats['was_late']]['review_score'].mean()
    gap          = on_time_mean - late_mean

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="finding-card">
            <b>Result: SUPPORTED (p &lt; 0.001)</b><br><br>
            On-time deliveries: <b>{on_time_mean:.2f} stars</b><br>
            Late deliveries: <b>{late_mean:.2f} stars</b><br>
            Gap: <b>{gap:.2f} points</b> on a 5-point scale<br><br>
            Test: Mann-Whitney U | Effect: Medium<br>
            Correlation (delay vs score): <b>-0.27</b>
        </div>
        <div class="finding-card">
            <b>Business implication:</b><br>
            Every percentage point reduction in the 6.8% late
            delivery rate has a statistically confirmed impact
            on satisfaction scores. Amazonas (AM) state averages
            +9 days late and is the priority intervention target.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        score_by_late = (
            delivered_stats.groupby('was_late')['review_score']
            .value_counts(normalize=True)
            .mul(100)
            .reset_index()
        )
        score_by_late['group'] = score_by_late['was_late'].map(
            {True: 'Late', False: 'On Time / Early'}
        )

        fig_h1 = px.bar(
            score_by_late,
            x='review_score',
            y='proportion',
            color='group',
            barmode='group',
            color_discrete_map={
                'On Time / Early': C['accent'],
                'Late': C['highlight']
            },
            labels={
                'review_score': 'Review Score',
                'proportion':   '% of Group',
                'group':        ''
            },
            height=300
        )
        fig_h1.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(orientation='h', y=1.1)
        )
        st.plotly_chart(fig_h1, use_container_width=True)

    st.markdown("---")

    # ── H2 Finding ────────────────────────────────────────
    st.markdown('<p class="section-header">Hypothesis 2  '
                'High vs Low RFM Customer Behaviour</p>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="finding-card">
            <b>Result: PARTIALLY SUPPORTED</b><br><br>
            All 4 variables significant (p &lt; 0.001) but
            only <b>total spend</b> shows a practically
            meaningful effect size.<br><br>
            Total Spend: effect size <b>0.857 (Large)</b><br>
            Total Orders: effect size <b>0.129 (Small)</b><br>
            Category Diversity: effect size <b>0.090 (Negligible)</b><br>
            Payment Type: Cramér's V <b>0.090 (Negligible)</b><br><br>
            High RFM spend: <b>R$300</b> vs Low RFM: <b>R$81</b>
        </div>
        <div class="finding-card">
            <b>Key insight:</b><br>
            Statistical significance ≠ practical significance.
            With 96,000 customers, even negligible differences
            become detectable. Effect size tells the real story.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        effect_data = pd.DataFrame({
            'Variable':    ['Total Spend', 'Total Orders',
                            'Category Diversity', 'Payment Type'],
            'Effect Size': [0.857, 0.129, 0.090, 0.090],
            'Meaningful':  ['Yes', 'No', 'No', 'No']
        })
        fig_h2 = px.bar(
            effect_data,
            x='Effect Size',
            y='Variable',
            orientation='h',
            color='Meaningful',
            color_discrete_map={'Yes': C['navy'], 'No': C['neutral']},
            height=280
        )
        fig_h2.add_vline(x=0.1, line_dash='dot',
                         line_color=C['accent'],
                         annotation_text='Moderate threshold (0.10)')
        fig_h2.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(title='Meaningful?', orientation='h', y=1.15)
        )
        st.plotly_chart(fig_h2, use_container_width=True)

    st.markdown("---")

    # ── Churn foundation ──────────────────────────────────
    st.markdown('<p class="section-header">Churn Foundation  '
                'RFM Segments and Churn Risk</p>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        churn_seg_data = {
            'Segment':    ['Champions', 'New Customers', 'Loyal Customers',
                           'Potential Loyalists', 'At Risk', 'Lost'],
            'Churn Rate': [28.3, 33.0, 57.5, 87.6, 100.0, 100.0]
        }
        churn_seg_df = pd.DataFrame(churn_seg_data).sort_values(
            'Churn Rate'
        )

        fig_churn = px.bar(
            churn_seg_df,
            x='Churn Rate',
            y='Segment',
            orientation='h',
            color='Churn Rate',
            color_continuous_scale=[C['green'], C['highlight']],
            height=300,
            labels={'Churn Rate': 'Churn Rate (%)'}
        )
        fig_churn.add_vline(x=71.1, line_dash='dash',
                            line_color=C['navy'],
                            annotation_text='Overall 71.1%')
        fig_churn.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor='white',
            paper_bgcolor='white',
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_churn, use_container_width=True)

    with col2:
        st.markdown("""
        <div class="finding-card">
            <b>Churn spread: 71.7 percentage points</b><br>
            Cramér's V = 0.5731  Large effect<br><br>
            Champions churn at <b>28.3%</b><br>
            At Risk and Lost churn at <b>100%</b>
        </div>
        <div class="warning-card">
            <b>Target leakage documented:</b><br>
            Churn label defined by 180-day recency threshold.
            Model with recency: AUC <b>1.000</b> (leaked).<br>
            Model without recency: AUC <b>0.625</b> (honest).<br>
            Gap of <b>0.375</b> quantifies the leakage.
        </div>
        <div class="finding-card">
            <b>Churn is a first-repeat-purchase problem:</b><br>
            71.1% of customers never returned after their
            first order. Retention strategy should focus on
            converting first-time buyers , not winning back
            customers who have already been inactive for months.
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PAGE 4 — CHURN PREDICTOR
# ============================================================

elif page == " Churn Predictor":

    st.markdown('<p class="main-title"> Churn Predictor</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-title"> Machine Learning  |  '
                'XGBoost behavioural churn model (no recency)</p>',
                unsafe_allow_html=True)

    # ── Model limitation banner ────────────────────────────
    st.markdown("""
    <div class="warning-card">
        <b>⚠ Model context:</b> This model predicts churn from
        <b>behavioural features only</b> (delivery experience,
        satisfaction, spend, engagement). recency_days was deliberately
        excluded to avoid target leakage. The model achieves
        <b>ROC-AUC 0.625</b> - honest signal, not definitional overlap.
        In production this would be redesigned to predict second-purchase
        probability using first-purchase features only.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col_inputs, col_output = st.columns([1, 1])

    # ── Input sliders ─────────────────────────────────────
    with col_inputs:
        st.markdown('<p class="section-header">Customer Profile</p>',
                    unsafe_allow_html=True)

        avg_delivery_delay = st.slider(
            "Average delivery delay (days)",
            min_value=-30,
            max_value=30,
            value=0,
            help="Negative = arrived early | Zero = on time | Positive = late"
        )

        avg_review_score = st.slider(
            "Average review score given",
            min_value=1.0,
            max_value=5.0,
            value=4.0,
            step=0.1,
            help="Average star rating this customer has given across orders"
        )

        total_spend = st.slider(
            "Total spend (R$)",
            min_value=0,
            max_value=2000,
            value=150,
            step=10,
            help="Total amount spent across all orders"
        )

        avg_installments = st.slider(
            "Average payment instalments",
            min_value=1.0,
            max_value=12.0,
            value=2.0,
            step=0.5,
            help="Average number of payment instalments used"
        )

        avg_processing_time = st.slider(
            "Average order processing time (days)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.5,
            help="Average days from order placement to approval"
        )

        total_orders = st.slider(
            "Total orders placed",
            min_value=1,
            max_value=10,
            value=1,
            help="How many orders this customer has placed"
        )

        category_diversity = st.slider(
            "Category diversity",
            min_value=1,
            max_value=5,
            value=1,
            help="Number of distinct product categories purchased from"
        )

    # ── Prediction output ─────────────────────────────────
    with col_output:
        st.markdown('<p class="section-header">Churn Prediction</p>',
                    unsafe_allow_html=True)

        # Build feature array in same order as training
        # Order matches FEATURE_COLS from Notebook 03
        feature_order = [
            'total_spend',
            'total_orders',
            'avg_delivery_delay',
            'avg_processing_time',
            'category_diversity',
            'avg_review_score',
            'avg_installments',
        ]

        input_features = np.array([[
            total_spend,
            total_orders,
            avg_delivery_delay,
            avg_processing_time,
            category_diversity,
            avg_review_score,
            avg_installments,
        ]])

        # Scale using fitted scaler
        input_scaled = scaler.transform(input_features)

        # Predict
        churn_prob     = model.predict_proba(input_scaled)[0][1]
        churn_pred     = int(churn_prob >= 0.5)
        retain_prob    = 1 - churn_prob

        # ── Gauge chart ────────────────────────────────────
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=churn_prob * 100,
            number={'suffix': '%', 'font': {'size': 40, 'color': C['navy']}},
            delta={
                'reference': 71.1,
                'valueformat': '.1f',
                'suffix': '% vs baseline',
                'increasing': {'color': C['highlight']},
                'decreasing': {'color': C['green']}
            },
            gauge={
                'axis':  {'range': [0, 100], 'ticksuffix': '%'},
                'bar':   {'color': C['highlight'] if churn_prob > 0.5
                          else C['accent']},
                'steps': [
                    {'range': [0, 40],   'color': '#EAFAF1'},
                    {'range': [40, 70],  'color': '#FEF9E7'},
                    {'range': [70, 100], 'color': '#FDEDEC'},
                ],
                'threshold': {
                    'line':  {'color': 'black', 'width': 2},
                    'value': 71.1
                }
            },
            title={'text': "Churn Probability", 'font': {'size': 16}}
        ))
        fig_gauge.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=40, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Risk classification ────────────────────────────
        if churn_prob > 0.75:
            risk_class = "HIGH RISK"
            card_class = "warning-card"
            action = ("Immediate retention intervention recommended. "
                      "Personalised win-back offer or priority support contact.")
        elif churn_prob > 0.50:
            risk_class = "MEDIUM RISK"
            card_class = "finding-card"
            action = ("Monitor closely. Consider a proactive satisfaction "
                      "check or targeted promotion.")
        else:
            risk_class = "LOW RISK"
            card_class = "success-card"
            action = ("Customer appears engaged. "
                      "Standard retention programme sufficient.")

        st.markdown(f"""
        <div class="{card_class}">
            <b>Risk Classification: {risk_class}</b><br><br>
            Churn probability: <b>{churn_prob:.1%}</b><br>
            Retention probability: <b>{retain_prob:.1%}</b><br>
            Platform baseline: <b>71.1%</b><br><br>
            <b>Recommended action:</b><br>
            {action}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Feature contribution table ─────────────────────
        st.markdown("**What is driving this prediction?**")
        st.caption("Based on SHAP feature importance ranking from Notebook 03")

        shap_rank = pd.DataFrame({
            'Feature': [
                'Delivery Delay',
                'Review Score',
                'Total Spend',
                'Instalments',
                'Processing Time',
                'Total Orders',
                'Category Diversity'
            ],
            'Your Value': [
                f"{avg_delivery_delay:+.1f} days",
                f"{avg_review_score:.1f} / 5.0",
                f"R${total_spend:,}",
                f"{avg_installments:.1f}x",
                f"{avg_processing_time:.1f} days",
                f"{total_orders}",
                f"{category_diversity}"
            ],
            'SHAP Importance': [
                '█████████ High',
                '████ Medium',
                '████ Medium',
                '███ Medium',
                '█ Low',
                '█ Low',
                '▌ Negligible'
            ]
        })
        st.dataframe(shap_rank, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Model card ────────────────────────────────────────
    with st.expander("Model Card  Full Technical Details"):
        st.markdown("""
        ### Model Card: XGBoost Churn Predictor

        **Version:** Behavioural model (no recency)
        **Training data:** 76,908 customers (80% of 96,136)
        **Test data:** 19,228 customers (20%)

        | Metric | Value |
        |--------|-------|
        | Algorithm | XGBoost (gradient boosting) |
        | CV ROC-AUC | 0.6810 (±0.0050) |
        | Test ROC-AUC | 0.6250 |
        | Class balance | SMOTE applied to training set |
        | Scaler | StandardScaler (fit on training data only) |

        **Features (in SHAP importance order):**
        1. avg_delivery_delay (0.3164) — strongest predictor
        2. avg_review_score (0.1261)
        3. total_spend (0.1225)
        4. avg_installments (0.1042)
        5. avg_processing_time (0.0313)
        6. total_orders (0.0250)
        7. category_diversity (0.0136)

        **Known limitations:**
        - Excludes recency_days (structural target leakage)
        - Trained on 2016–2018 data — behaviour may have shifted
        - 0.625 AUC = better than random, modest in absolute terms
        - Production version should predict second-purchase
          probability using first-order features only

        **Compared to leaked model:**
        - With recency_days: AUC 1.000 (definitional overlap)
        - Without recency_days: AUC 0.625 (genuine signal)
        - Leakage magnitude: 0.375 AUC points

        **Why delivery delay dominates SHAP:**
        Analysis C predicted recency would dominate but linear
        correlation missed delivery delay's non-linear effect.
        XGBoost found that severely late deliveries predict
        churn at disproportionately high rates — a threshold
        effect invisible to Pearson/point-biserial correlation.
        """)
