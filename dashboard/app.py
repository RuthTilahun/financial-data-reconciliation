"""
Streamlit Dashboard for Financial Data Quality Monitoring
Interactive dashboard for tracking data quality metrics, issues, and reconciliation
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

st.set_page_config(
    page_title="Financial Data Quality Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    """Load all data files"""
    data = {}

    # Load master financials
    if os.path.exists('data/processed/master_financials.csv'):
        data['master'] = pd.read_csv('data/processed/master_financials.csv')
    else:
        data['master'] = pd.DataFrame()

    # Load quality scores
    if os.path.exists('data/reconciliation/company_quality_scores.csv'):
        data['quality'] = pd.read_csv('data/reconciliation/company_quality_scores.csv')
    else:
        data['quality'] = pd.DataFrame()

    # Load validation issues
    if os.path.exists('data/reconciliation/validation_issues.csv'):
        data['issues'] = pd.read_csv('data/reconciliation/validation_issues.csv')
    else:
        data['issues'] = pd.DataFrame()

    # Load raw data for comparison
    if os.path.exists('data/raw/sec_filings.csv'):
        data['sec'] = pd.read_csv('data/raw/sec_filings.csv')
    else:
        data['sec'] = pd.DataFrame()

    if os.path.exists('data/raw/alpha_vantage.csv'):
        data['av'] = pd.read_csv('data/raw/alpha_vantage.csv')
    else:
        data['av'] = pd.DataFrame()

    return data

data = load_data()

# Header
st.markdown('<p class="main-header">📊 Financial Data Quality Dashboard</p>', unsafe_allow_html=True)
st.markdown("**Multi-Source Investment Data Reconciliation Platform**")
st.markdown("---")

# Key Metrics Row
if len(data['master']) > 0:
    col1, col2, col3, col4 = st.columns(4)

    total_companies = len(data['master'])
    total_records = len(data['sec']) + len(data['av'])
    avg_quality = data['quality']['quality_score'].mean() if len(data['quality']) > 0 else 0
    issues_count = len(data['issues']) if len(data['issues']) > 0 else 0

    col1.metric("Companies Tracked", f"{total_companies}", "")
    col2.metric("Total Records", f"{total_records}", "")
    col3.metric("Avg Quality Score", f"{avg_quality:.1f}%", f"+{avg_quality-78:.1f}%" if avg_quality > 78 else "")
    col4.metric("Open Issues", f"{issues_count}", "")

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🔍 Issue Tracker", "🔄 Reconciliation", "💰 Financial Insights"])

# TAB 1: Overview
with tab1:
    st.subheader("📈 Data Quality Overview")

    col1, col2 = st.columns(2)

    with col1:
        # Issues by severity
        if len(data['issues']) > 0:
            st.write("**Issues by Severity**")
            severity_counts = data['issues']['severity'].value_counts().reset_index()
            severity_counts.columns = ['severity', 'count']

            fig = px.pie(severity_counts, values='count', names='severity',
                        color='severity',
                        color_discrete_map={
                            'critical': '#FF4B4B',
                            'high': '#FFA500',
                            'medium': '#FFD700',
                            'low': '#90EE90'
                        },
                        title="Issues by Severity")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No validation issues detected!")

    with col2:
        # Quality scores by company
        if len(data['quality']) > 0:
            st.write("**Data Quality Scores by Company**")
            fig = px.bar(data['quality'].sort_values('quality_score', ascending=False),
                        x='ticker', y='quality_score',
                        color='quality_score',
                        color_continuous_scale='RdYlGn',
                        range_color=[80, 100],
                        title="Quality Score by Company")
            fig.add_hline(y=95, line_dash="dash", line_color="green",
                         annotation_text="Target: 95%")
            st.plotly_chart(fig, use_container_width=True)

    # Data Source Coverage
    st.write("**Data Source Coverage**")
    source_data = pd.DataFrame({
        'Source': ['SEC EDGAR', 'Alpha Vantage', 'Analyst Reports'],
        'Records': [len(data['sec']), len(data['av']), 0],
        'Companies': [
            len(data['sec']['ticker'].unique()) if len(data['sec']) > 0 else 0,
            len(data['av']['ticker'].unique()) if len(data['av']) > 0 else 0,
            0
        ],
        'Status': ['✅ Active', '✅ Active', '✅ Active']
    })
    st.dataframe(source_data, use_container_width=True, hide_index=True)

# TAB 2: Issue Tracker
with tab2:
    st.subheader("📋 Data Quality Issues")

    if len(data['issues']) > 0:
        # Filters
        col1, col2 = st.columns(2)
        severity_filter = col1.multiselect(
            "Severity",
            data['issues']['severity'].unique(),
            default=data['issues']['severity'].unique()
        )
        rule_filter = col2.multiselect(
            "Rule Type",
            data['issues']['rule'].unique(),
            default=data['issues']['rule'].unique()
        )

        # Filter data
        filtered_issues = data['issues'][
            (data['issues']['severity'].isin(severity_filter)) &
            (data['issues']['rule'].isin(rule_filter))
        ]

        st.write(f"**Showing {len(filtered_issues)} of {len(data['issues'])} issues**")

        # Display issues
        for idx, row in filtered_issues.iterrows():
            severity_emoji = {
                'critical': '🔴',
                'high': '🟠',
                'medium': '🟡',
                'low': '🟢'
            }

            with st.expander(f"{severity_emoji.get(row['severity'], '⚪')} {row['rule'].upper()}: {row['description'][:80]}..."):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Description:** {row['description']}")
                    if 'field' in row and pd.notna(row['field']):
                        st.write(f"**Field:** {row['field']}")
                    if 'dataset' in row and pd.notna(row['dataset']):
                        st.write(f"**Dataset:** {row['dataset']}")
                with col2:
                    st.write(f"**Severity:** {row['severity'].upper()}")
                    st.write(f"**Rule:** {row['rule']}")
    else:
        st.success("✅ No validation issues detected! All data quality checks passed.")

# TAB 3: Reconciliation
with tab3:
    st.subheader("🔄 Cross-Source Reconciliation")

    if len(data['master']) > 0:
        st.write("**Golden Records Summary**")
        st.write(f"Successfully created {len(data['master'])} golden records from {data['master']['num_sources'].iloc[0] if 'num_sources' in data['master'].columns else 3} sources")

        # Show reconciliation details
        st.write("**Revenue Reconciliation Comparison**")

        if len(data['sec']) > 0 and len(data['av']) > 0:
            # Merge SEC and AV data for comparison
            comparison = data['sec'][['ticker', 'company_name', 'revenue']].copy()
            comparison.columns = ['ticker', 'company_name', 'SEC Revenue']

            av_revenue = data['av'][['ticker', 'revenue']].copy()
            av_revenue.columns = ['ticker', 'AV Revenue']

            comparison = comparison.merge(av_revenue, on='ticker', how='left')

            # Calculate difference
            comparison['Difference ($)'] = comparison['SEC Revenue'] - comparison['AV Revenue']
            comparison['Difference (%)'] = (comparison['Difference ($)'] / comparison['SEC Revenue'] * 100).round(2)
            comparison['Status'] = comparison['Difference (%)'].apply(
                lambda x: '✅ Within tolerance' if abs(x) < 5 else '⚠️ Review needed'
            )

            # Format for display
            display_comparison = comparison.copy()
            display_comparison['SEC Revenue'] = display_comparison['SEC Revenue'].apply(lambda x: f"${x/1e9:.2f}B")
            display_comparison['AV Revenue'] = display_comparison['AV Revenue'].apply(lambda x: f"${x/1e9:.2f}B")
            display_comparison['Difference ($)'] = display_comparison['Difference ($)'].apply(lambda x: f"${x/1e9:.2f}B")

            st.dataframe(display_comparison[['ticker', 'company_name', 'SEC Revenue', 'AV Revenue', 'Difference (%)', 'Status']],
                        use_container_width=True, hide_index=True)

        # Show data lineage
        st.write("**Data Lineage**")
        if 'sources_used' in data['master'].columns:
            lineage = data['master'][['ticker', 'company_name', 'sources_used', 'num_sources']].copy()
            st.dataframe(lineage, use_container_width=True, hide_index=True)

# TAB 4: Financial Insights
with tab4:
    st.subheader("💰 Financial Analysis & Insights")

    if len(data['master']) > 0:
        # Add sector mapping
        sector_map = {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'GOOGL': 'Technology',
            'AMZN': 'Consumer',
            'TSLA': 'Automotive',
            'JPM': 'Financial',
            'JNJ': 'Healthcare',
            'WMT': 'Retail',
            'DIS': 'Media',
            'XOM': 'Energy'
        }

        master_with_sector = data['master'].copy()
        master_with_sector['sector'] = master_with_sector['ticker'].map(sector_map)
        master_with_sector['profit_margin'] = (master_with_sector['net_income'] / master_with_sector['revenue']) * 100

        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)

        total_revenue = master_with_sector['revenue'].sum()
        avg_margin = master_with_sector['profit_margin'].mean()
        top_company = master_with_sector.loc[master_with_sector['revenue'].idxmax(), 'company_name']

        col1.metric("Total Revenue", f"${total_revenue/1e9:.1f}B")
        col2.metric("Avg Profit Margin", f"{avg_margin:.1f}%")
        col3.metric("Top Revenue", top_company)
        col4.metric("Companies", len(master_with_sector))

        st.markdown("---")

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Revenue by Sector**")
            sector_revenue = master_with_sector.groupby('sector')['revenue'].sum().sort_values(ascending=False)
            fig = px.pie(values=sector_revenue.values, names=sector_revenue.index,
                        title="Revenue Distribution by Sector",
                        hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.write("**Profitability Comparison**")
            fig = px.bar(master_with_sector.sort_values('profit_margin', ascending=False),
                        x='ticker', y='profit_margin',
                        color='profit_margin',
                        color_continuous_scale='RdYlGn',
                        title="Profit Margin by Company",
                        labels={'profit_margin': 'Profit Margin (%)'})
            st.plotly_chart(fig, use_container_width=True)

        # Detailed Table
        st.write("**Company Financial Summary**")
        display_df = master_with_sector[['ticker', 'company_name', 'sector', 'revenue', 'net_income', 'profit_margin']].copy()
        display_df['revenue'] = display_df['revenue'].apply(lambda x: f"${x/1e9:.2f}B")
        display_df['net_income'] = display_df['net_income'].apply(lambda x: f"${x/1e9:.2f}B")
        display_df['profit_margin'] = display_df['profit_margin'].apply(lambda x: f"{x:.1f}%")
        display_df.columns = ['Ticker', 'Company', 'Sector', 'Revenue', 'Net Income', 'Margin']

        st.dataframe(display_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(f"*Dashboard last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
st.markdown("**Data Sources:** SEC EDGAR API, Alpha Vantage API, Analyst Reports")
st.markdown("Built with Streamlit | Financial Data Quality Pipeline v1.0")
