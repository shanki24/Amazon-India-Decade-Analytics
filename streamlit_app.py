# scripts/streamlit_app.py
import os
from urllib.parse import quote_plus
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import create_engine, text
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from functools import lru_cache

load_dotenv()

# Connection helpers
def make_engine():
    DB_USER = os.getenv('DB_USER', 'root')
    DB_PASS = quote_plus(os.getenv('DB_PASSWORD', ''))
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '3306')
    DB_NAME = os.getenv('DB_NAME', 'amazon_analytics')
    url = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
    return create_engine(url), DB_NAME

def get_table_columns(engine, db_name, table_name='transactions'):
    sql = text("""
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = :db AND TABLE_NAME = :table;
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"db": db_name, "table": table_name}).fetchall()
    return [r[0] for r in rows]

def pick_first(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None

# Data loader / cleaner
@st.cache_data(ttl=300)
def load_data(limit=1_000_000):
    engine, db_name = make_engine()
    try:
        cols = get_table_columns(engine, db_name, table_name='transactions')
    except Exception as e:
        st.error("Could not read table schema from INFORMATION_SCHEMA. Check DB credentials and that table `transactions` exists.")
        raise

    # candidate columns (extend as needed)
    amount_candidates = ['final_amount', 'final_amount_inr', 'final_price', 'order_value', 'amount', 'total_amount', 'price']
    product_candidates = ['product_id','product_name']
    category_candidates = ['category', 'product_category', 'category_name', 'cat']
    city_candidates = ['customer_city', 'city', 'ship_city']
    state_candidates = ['customer_state', 'state', 'ship_state']
    order_date_candidates = ['order_date', 'orderdate', 'order_date_time', 'order_timestamp', 'date']
    order_year_candidates = ['order_year', 'year', 'order_yr']
    order_month_candidates = ['order_month', 'month', 'order_mo']
    customer_candidates = ['customer_id']
    delivery_days_candidates = ['delivery_days','delivery_time_days']
    prime_candidates = ['is_prime_member','is_prime','prime']
    rating_candidates = ['product_rating','customer_rating','rating']
    return_candidates = ['return_status','is_returned','returned']

    amount_col = pick_first(cols, amount_candidates)
    product_col = pick_first(cols, product_candidates)
    category_col = pick_first(cols, category_candidates)
    city_col = pick_first(cols, city_candidates)
    state_col = pick_first(cols, state_candidates)
    order_date_col = pick_first(cols, order_date_candidates)
    order_year_col = pick_first(cols, order_year_candidates)
    order_month_col = pick_first(cols, order_month_candidates)
    customer_col = pick_first(cols, customer_candidates)
    delivery_days_col = pick_first(cols, delivery_days_candidates)
    prime_col = pick_first(cols, prime_candidates)
    rating_col = pick_first(cols, rating_candidates)
    return_col = pick_first(cols, return_candidates)

    if order_date_col is None:
        st.error("No order date column found in `transactions`. At minimum the table must have a date column.")
        raise RuntimeError("No date column")

    select_cols = [f"`{order_date_col}` AS order_date"]
    optional_map = {
        'order_year': order_year_col,
        'order_month': order_month_col,
        'final_amount': amount_col,
        'product_id': product_col,
        'category': category_col,
        'customer_city': city_col,
        'customer_state': state_col,
        'customer_id': customer_col,
        'delivery_days': delivery_days_col,
        'is_prime_member': prime_col,
        'rating': rating_col,
        'return_status': return_col
    }
    for alias, col in optional_map.items():
        if col:
            select_cols.append(f"`{col}` AS {alias}")

    select_sql = "SELECT " + ", ".join(select_cols) + f" FROM `transactions`"
    if limit:
        select_sql += f" LIMIT {limit}"

    try:
        df = pd.read_sql_query(select_sql, make_engine()[0], parse_dates=['order_date'])
    except Exception as e:
        st.error("Error running SELECT on DB. Check connection + query.")
        raise

    # Clean numeric textual amounts -> numeric
    if 'final_amount' in df.columns:
        df['final_amount'] = pd.to_numeric(df['final_amount'].astype(str).str.replace('[^0-9.\-]','', regex=True), errors='coerce')

    # ensure order_year/month exist
    if 'order_year' not in df.columns:
        df['order_year'] = df['order_date'].dt.year
    if 'order_month' not in df.columns:
        df['order_month'] = df['order_date'].dt.month

    # canonical columns present
    for c in ['category','customer_city','customer_state','customer_id','delivery_days','is_prime_member','rating','return_status','product_id','product_name']:
        if c not in df.columns:
            df[c] = pd.NA

    return df

# Utility KPI functions
def format_currency(x):
    if pd.isna(x):
        return "N/A"
    return f"₹{x:,.0f}"

def yoy_growth(series):
    # expects index = year sorted
    s = series.sort_index()
    return s.pct_change().fillna(0)

# RFM helper
@st.cache_data(ttl=600)
def compute_rfm(df):
    if 'customer_id' not in df.columns or df['customer_id'].isna().all():
        return None
    snapshot = df['order_date'].max() + pd.Timedelta(days=1)
    grouped = df.dropna(subset=['customer_id']).groupby('customer_id').agg(
        recency=('order_date', lambda x: (snapshot - x.max()).days),
        frequency=('order_date','count'),
        monetary=('final_amount','sum')
    ).reset_index()
    grouped['monetary'] = grouped['monetary'].fillna(0)
    return grouped

# Cohort helper
@st.cache_data(ttl=600)
def compute_cohort_retention(df):
    if 'customer_id' not in df.columns or df['customer_id'].isna().all():
        return None
    users = df[['customer_id','order_date']].dropna()
    users['order_month'] = users['order_date'].dt.to_period('M').astype(str)
    users['cohort_month'] = users.groupby('customer_id')['order_date'].transform('min').dt.to_period('M').astype(str)
    cohort_counts = users.groupby(['cohort_month','order_month'])['customer_id'].nunique().reset_index()
    cohort_pivot = cohort_counts.pivot(index='cohort_month', columns='order_month', values='customer_id').fillna(0)
    cohort_sizes = cohort_pivot.iloc[:,0]
    retention = cohort_pivot.divide(cohort_sizes, axis=0)
    return retention

# UI layout
st.set_page_config(page_title="Amazon India — Decade Analytics", layout="wide", initial_sidebar_state="expanded")
st.title("Amazon India — Decade Analytics")

# Load data
with st.spinner("Loading data from DB..."):
    try:
        df = load_data(limit=int(os.getenv("STREAMLIT_LIMIT", "500000")))
    except Exception as e:
        st.error("Failed to load data. See console for details.")
        st.stop()

# Sidebar navigation
section = st.sidebar.selectbox("Dashboard area", [
    "Executive Summary",
    "Real-time Monitor",
    "Strategic Overview",
    "Financials",
    "Growth Analytics",
    "Revenue Analytics",
    "Category Performance",
    "Geographic Revenue",
    "Festival Analytics",
    "Price Optimization",
    "Customer Segmentation",
    "Customer Journey",
    "Prime Analytics",
    "Retention & Cohorts",
    "Demographics & Behavior",
    "Product Performance",
    "Brand Analytics",
    "Inventory Optimization",
    "Ratings & Reviews",
    "New Product Launch",
    "Delivery Performance",
    "Payment Analytics",
    "Returns & Cancellations",
    "Customer Service",
    "Supply Chain",
    "Predictive Analytics",
    "Market Intelligence",
    "Cross-sell & Upsell",
    "Seasonal Planning",
    "BI Command Center"
])

# Global filters
years = sorted(df['order_year'].dropna().unique())
sel_year = st.sidebar.selectbox("Year (all)", ["All"] + years)
sel_month = st.sidebar.selectbox("Month (all)", ["All"] + list(range(1,13)))

filtered = df.copy()
if sel_year != "All":
    filtered = filtered[filtered['order_year'] == int(sel_year)]
if sel_month != "All":
    filtered = filtered[filtered['order_month'] == int(sel_month)]

# Executive Summary
if section == "Executive Summary":
    st.header("Executive Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    # Total revenue
    if 'final_amount' in filtered.columns and filtered['final_amount'].notna().any():
        total_revenue = filtered['final_amount'].sum()
        col1.metric("Total Revenue", format_currency(total_revenue))
        # YoY basic: comparing selected year to previous year
        if sel_year != "All":
            yr = int(sel_year)
            prev = df[df['order_year']==yr-1]['final_amount'].sum() if 'final_amount' in df.columns else np.nan
            growth = (total_revenue - prev) / prev if prev and prev != 0 else np.nan
            col1.metric("YoY vs prev year", f"{growth*100:.1f}%" if not pd.isna(growth) else "N/A")
    else:
        col1.metric("Total Revenue","N/A")

    # Orders
    orders = len(filtered)
    col2.metric("Orders", f"{orders:,}")

    # Active Customers
    active_customers = filtered['customer_id'].nunique() if 'customer_id' in filtered.columns else None
    col3.metric("Active Customers", f"{active_customers:,}" if active_customers else "N/A")

    # Avg Order Value
    if 'final_amount' in filtered.columns and filtered['final_amount'].notna().any():
        aov = filtered['final_amount'].mean()
        col4.metric("Avg Order Value", format_currency(aov))
    else:
        col4.metric("Avg Order Value", "N/A")

    # Top Categories
    col5.metric("Top Category", filtered.groupby('category')['final_amount'].sum().idxmax() if 'category' in filtered.columns and filtered['category'].notna().any() else "N/A")

    st.markdown("#### Revenue & Trend")
    if 'final_amount' in df.columns and df['final_amount'].notna().any():
        annual = df.groupby('order_year')['final_amount'].sum().sort_index()
        fig = px.line(annual.reset_index(), x='order_year', y='final_amount', title='Annual Revenue Trend', markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No revenue column available to show annual trend.")

    st.markdown("#### Top performing categories (by revenue)")
    if 'final_amount' in filtered.columns and filtered['final_amount'].notna().any() and filtered['category'].notna().any():
        cat = filtered.groupby('category')['final_amount'].sum().sort_values(ascending=False).reset_index().head(10)
        fig = px.bar(cat, x='final_amount', y='category', orientation='h', title='Top Categories by Revenue')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Category or amount missing: cannot display top categories by revenue.")

# Real-time Monitor
elif section == "Real-time Monitor":
    st.header("Real-time Business Performance Monitor (approx)")
    st.info("This monitor uses most recent month available in the data as 'current month'. For true real-time, connect to streaming / events source.")
    latest_date = df['order_date'].max()
    current_month = latest_date.month
    current_year = latest_date.year
    st.subheader(f"Current month (data available): {current_year}-{str(current_month).zfill(2)}")

    cur = df[(df['order_date'].dt.year==current_year) & (df['order_date'].dt.month==current_month)]
    revenue_cur = cur['final_amount'].sum() if 'final_amount' in cur.columns else np.nan
    orders_cur = len(cur)
    customers_cur = cur['customer_id'].nunique() if 'customer_id' in cur.columns else np.nan

    c1,c2,c3 = st.columns(3)
    c1.metric("Month Revenue (actual)", format_currency(revenue_cur))
    c2.metric("Orders (month)", f"{orders_cur:,}")
    c3.metric("New Customers (month)", f"{customers_cur:,}" if customers_cur else "N/A")

    # simple run-rate = daily avg * days in month
    if not cur.empty and 'final_amount' in cur.columns:
        days_obs = cur['order_date'].dt.day.nunique()
        run_rate = (cur['final_amount'].sum()/days_obs) * 30 if days_obs else np.nan
        st.metric("Revenue Run-rate (30d)", format_currency(run_rate))
    else:
        st.info("No revenue data for current month.")

    # Alerts placeholder: threshold-based
    target_revenue = st.sidebar.number_input("Monthly revenue target (optional)", value=0)
    if target_revenue:
        if revenue_cur < target_revenue:
            st.error(f"Alert: current-month revenue {format_currency(revenue_cur)} below target {format_currency(target_revenue)}")
        else:
            st.success("On track vs monthly revenue target.")

# Strategic Overview
elif section == "Strategic Overview":
    st.header("Strategic Overview")
    st.markdown("Market share, geographic expansion, and high-level business health KPIs.")
    if 'final_amount' in df.columns and df['final_amount'].notna().any():
        # market share by top 5 categories as proxy
        ms = df.groupby('category')['final_amount'].sum().nlargest(6).reset_index()
        fig = px.pie(ms, values='final_amount', names='category', title='Category Market Share (top 6)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No revenue data to compute market share.")

    # Geo expansion: top states
    if 'customer_state' in df.columns and df['customer_state'].notna().any():
        st.subheader("Top states by revenue")
        if 'final_amount' in df.columns:
            st.bar_chart(df.groupby('customer_state')['final_amount'].sum().nlargest(10))
        else:
            st.bar_chart(df['customer_state'].value_counts().nlargest(10))
    else:
        st.info("No customer_state column present.")

# Financials
elif section == "Financials":
    st.header("Financial Performance")
    st.markdown("Revenue breakdowns and basic discount effectiveness analysis.")
    if 'final_amount' in df.columns:
        st.subheader("Revenue by month")
        monthly = df.groupby(['order_year','order_month'])['final_amount'].sum().reset_index()
        monthly['ym'] = monthly['order_year'].astype(str) + "-" + monthly['order_month'].astype(str).str.zfill(2)
        fig = px.line(monthly, x='ym', y='final_amount', title='Monthly Revenue', markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No final_amount column for financials.")

    # Discount effectiveness: if discount % exists
    if 'discount_percent' in df.columns:
        disp = df.dropna(subset=['discount_percent','final_amount'])
        # clean discount text numeric
        disp['discount_percent'] = pd.to_numeric(disp['discount_percent'].astype(str).str.replace('[^0-9.\-]','', regex=True), errors='coerce')
        st.subheader("Discount % vs Avg Order Value (sample)")
        sample = disp.sample(min(len(disp), 5000))
        fig = px.scatter(sample, x='discount_percent', y='final_amount', trendline='ols', title='Discount % vs Final Amount')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No discount_percent column to analyze discounts.")

# Growth Analytics
elif section == "Growth Analytics":
    st.header("Growth Analytics")
    st.markdown("Customer growth, penetration, product portfolio expansion.")
    if 'customer_id' in df.columns:
        cust_by_year = df.groupby('order_year')['customer_id'].nunique().reset_index()
        fig = px.line(cust_by_year, x='order_year', y='customer_id', title='Active Customers by Year', markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No customer_id column available.")

# Revenue Analytics
elif section == "Revenue Analytics":
    st.header("Revenue Trend Analysis")
    if 'final_amount' in df.columns:
        monthly = df.groupby(['order_year','order_month'])['final_amount'].sum().reset_index()
        monthly['date'] = pd.to_datetime(monthly['order_year'].astype(str) + '-' + monthly['order_month'].astype(str) + '-01')
        fig = px.line(monthly.sort_values('date'), x='date', y='final_amount', title='Revenue Time Series', markers=True)
        st.plotly_chart(fig, use_container_width=True)
        # simple forecast placeholder
        st.info("Forecasting placeholder: plug in ARIMA/Prophet/ML model here to show forward projection.")
    else:
        st.info("No revenue column available.")

# Category Performance
elif section == "Category Performance":
    st.header("Category Performance")
    if 'category' in df.columns:
        cat = df.groupby('category').agg(revenue=('final_amount','sum'), orders=('order_date','count')).reset_index().sort_values('revenue', ascending=False).head(25)
        st.dataframe(cat)
        fig = px.bar(cat.head(10), x='revenue', y='category', orientation='h', title='Top categories (revenue)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Category column not available.")

# Geographic Revenue
elif section == "Geographic Revenue":
    st.header("Geographic Revenue Analysis")
    if 'customer_state' in df.columns:
        if 'final_amount' in df.columns:
            state_rev = df.groupby('customer_state')['final_amount'].sum().reset_index().sort_values('final_amount', ascending=False).head(20)
            fig = px.bar(state_rev, x='final_amount', y='customer_state', orientation='h', title='Top states by revenue')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(df['customer_state'].value_counts().nlargest(20))
    else:
        st.info("No state column present.")

# Festival Analytics
elif section == "Festival Analytics":
    st.header("Festival / Campaign Sales")
    if 'is_festival_sale' in df.columns or 'festival_name' in df.columns:
        if 'order_date' in df.columns and 'final_amount' in df.columns:
            st.info("Note: festival windows are best analyzed using exact date calendar; this is a best-effort view.")
            fest = df[df['is_festival_sale'].notna()] if 'is_festival_sale' in df.columns else df[df['festival_name'].notna()]
            if not fest.empty:
                monthly = fest.groupby(['order_year','order_month'])['final_amount'].sum().reset_index()
                monthly['ym'] = monthly['order_year'].astype(str) + '-' + monthly['order_month'].astype(str).str.zfill(2)
                fig = px.bar(monthly, x='ym', y='final_amount', title='Festival window revenue by month')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No festival-flagged rows found.")
        else:
            st.info("Need order_date and final_amount for festival analysis.")
    else:
        st.info("No festival columns in dataset.")

# Price Optimization (simple elasticity analysis stub)
elif section == "Price Optimization":
    st.header("Price Optimization")
    st.markdown("Price vs orders / revenue. For elasticity compute grouped regressions or causal models.")
    if 'product_id' in df.columns and 'final_amount' in df.columns:
        price_col = None
        # attempt to pick a price column if present in DB (could be product table join)
        for c in ['original_price_inr','original_price_inr_clean','price','base_price_2015']:
            if c in df.columns:
                price_col = c; break
        if price_col:
            df['price_num'] = pd.to_numeric(df[price_col].astype(str).str.replace('[^0-9.\-]','', regex=True), errors='coerce')
            demand = df.groupby('product_id').agg(price=('price_num','mean'), orders=('order_date','count'), revenue=('final_amount','sum')).dropna().reset_index()
            fig = px.scatter(demand.sample(min(len(demand),2000)), x='price', y='orders', size='revenue', hover_name='product_id', title='Price vs Orders (bubble size=revenue)')
            st.plotly_chart(fig, use_container_width=True)
            st.info("For elasticity: run log-log regressions per product/category and aggregate elasticity estimates.")
        else:
            st.info("No price column in transactions; consider joining product catalog or adding price column to transactions.")
    else:
        st.info("Need product_id and final_amount to run price-demand analysis.")

# Customer Segmentation (RFM)
elif section == "Customer Segmentation":
    st.header("RFM Segmentation")
    rfm = compute_rfm(df)
    if rfm is None:
        st.info("No customer_id/order_date present to compute RFM.")
    else:
        st.dataframe(rfm.describe().T)
        # quick scatter of frequency vs monetary colored by recency quartile
        rfm['recency_q'] = pd.qcut(rfm['recency'].rank(method='first'), 4, labels=False)
        sample = rfm.sample(min(len(rfm), 5000))
        fig = px.scatter(sample, x='frequency', y='monetary', color='recency_q', title='RFM scatter (sample)')
        st.plotly_chart(fig, use_container_width=True)

# Customer Journey
elif section == "Customer Journey":
    st.header("Customer Journey & Transitions")
    if 'customer_id' in df.columns and 'category' in df.columns:
        m = df.sort_values(['customer_id','order_date']).dropna(subset=['customer_id','category'])
        # compute transitions counts (from -> to)
        trans = {}
        last = {}
        for rid, row in m[['customer_id','category']].iterrows():
            cid = row['customer_id']; cat = row['category']
            if cid in last:
                prev = last[cid]
                trans[(prev, cat)] = trans.get((prev, cat), 0) + 1
            last[cid] = cat
        trans_df = pd.DataFrame([{'from':k[0], 'to':k[1], 'count':v} for k,v in trans.items()]).sort_values('count', ascending=False).head(30)
        st.dataframe(trans_df)
    else:
        st.info("Need customer_id and category to build journey transitions.")

# Prime Analytics
elif section == "Prime Analytics":
    st.header("Prime vs Non-Prime Analysis")
    if 'is_prime_member' in df.columns and 'final_amount' in df.columns:
        df['is_prime_member'] = df['is_prime_member'].astype(str).str.lower().isin(['1','true','yes','y','t'])
        aov = df.groupby('is_prime_member')['final_amount'].agg(total='sum', count='count', mean='mean').reset_index()
        st.table(aov)
        fig = px.bar(aov, x='is_prime_member', y='mean', title='Avg Order Value: Prime vs Non-Prime')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Prime flag or final_amount missing.")

# Retention & Cohorts
elif section == "Retention & Cohorts":
    st.header("Cohort Retention")
    retention = compute_cohort_retention(df)
    if retention is None or retention.empty:
        st.info("Not enough data to compute cohorts.")
    else:
        fig = px.imshow(retention.fillna(0).values, x=retention.columns, y=retention.index, labels=dict(x="Order month", y="Cohort month", color="Retention"), aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

# Demographics & Behavior
elif section == "Demographics & Behavior":
    st.header("Demographics & Behavior")
    if 'customer_age_group' in df.columns and 'final_amount' in df.columns:
        spend = df.groupby('customer_age_group')['final_amount'].agg(sum='sum', mean='mean', count='count').reset_index().dropna()
        st.dataframe(spend)
        fig = px.bar(spend, x='customer_age_group', y='mean', title='Avg spending by age group')
        st.plotly_chart(fig)
    else:
        st.info("No age_group or final_amount available.")

# Product Performance
elif section == "Product Performance":
    st.header("Product Performance")
    if 'product_id' in df.columns:
        prod = df.groupby('product_id').agg(units=('order_date','count'), revenue=('final_amount','sum')).sort_values('revenue', ascending=False).reset_index().head(50)
        st.dataframe(prod)
        fig = px.bar(prod.head(10), x='revenue', y='product_id', orientation='h', title='Top products by revenue')
        st.plotly_chart(fig)
    else:
        st.info("No product_id column available.")

# Brand Analytics
elif section == "Brand Analytics":
    st.header("Brand Analytics")
    # requires product join; placeholder
    st.info("Brand analysis requires brand column either in transactions or a join to products table. Add brand to transactions or join in SQL.")

# Inventory Optimization
elif section == "Inventory Optimization":
    st.header("Inventory Optimization")
    st.info("Inventory dashboard requires inventory feed (stock levels, reorder points). This section is a placeholder for demand forecasting and turnover metrics.")

# Ratings & Reviews
elif section == "Ratings & Reviews":
    st.header("Ratings & Reviews")
    if 'rating' in df.columns:
        r = pd.to_numeric(df['rating'], errors='coerce').dropna()
        st.write(r.describe())
        fig = px.histogram(r, nbins=20, title='Rating distribution')
        st.plotly_chart(fig)
    else:
        st.info("No rating column present in transactions. Consider table join with reviews dataset.")

# New Product Launch
elif section == "New Product Launch":
    st.header("New Product Launch Monitoring")
    st.info("Track new product revenue adoption vs baseline. Requires product launch date in products catalog and joins.")

# ---------- Delivery Performance
elif section == "Delivery Performance":
    st.header("Delivery Performance")
    if 'delivery_days' in df.columns:
        dd = pd.to_numeric(df['delivery_days'], errors='coerce').dropna()
        fig = px.histogram(dd, nbins=40, title='Delivery Days Distribution')
        st.plotly_chart(fig)
        st.metric("On-time <=7 days", f"{(dd <= 7).mean():.2%}")
    else:
        st.info("No delivery_days column.")

# ---------- Payment Analytics ----------
elif section == "Payment Analytics":
    st.header("Payment Methods")
    if 'payment_method' in df.columns:
        pm = df['payment_method'].value_counts().reset_index().rename(columns={'index':'method','payment_method':'count'})
        st.dataframe(pm.head(20))
        st.bar_chart(pm.set_index('method')['count'].head(10))
    else:
        st.info("No payment_method column found.")

# ---------- Returns & Cancellations ----------
elif section == "Returns & Cancellations":
    st.header("Returns & Cancellations")
    if 'return_status' in df.columns:
        r = df['return_status'].value_counts().reset_index().rename(columns={'index':'status','return_status':'count'})
        st.dataframe(r)
    else:
        st.info("No return status column.")

# ---------- Customer Service ----------
elif section == "Customer Service":
    st.header("Customer Service")
    st.info("Requires ticket/CS dataset: response time, NPS, CSAT. Placeholder.")

# ---------- Supply Chain ----------
elif section == "Supply Chain":
    st.header("Supply Chain")
    st.info("Requires supplier/delivery partner datasets. Placeholder.")

# ---------- Predictive Analytics ----------
elif section == "Predictive Analytics":
    st.header("Predictive Analytics")
    st.info("Plug in forecasting / churn models here (Prophet, ARIMA, XGBoost etc.). This scaffold leaves hooks for model integration.")

# ---------- Market Intelligence ----------
elif section == "Market Intelligence":
    st.header("Market Intelligence")
    st.info("Requires external competitor price feeds and market datasets. Placeholder.")

# ---------- Cross-sell & Upsell ----------
elif section == "Cross-sell & Upsell":
    st.header("Cross-sell & Upsell")
    st.info("Compute product association rules (Apriori) or market-basket analysis on transactions. Placeholder for association mining.")

# ---------- Seasonal Planning ----------
elif section == "Seasonal Planning":
    st.header("Seasonal Planning")
    st.info("Seasonal demand planning requires forecast & inventory — placeholder for calendar & resource allocation charts.")

# ---------- BI Command Center ----------
elif section == "BI Command Center":
    st.header("BI Command Center")
    st.markdown("""
    - Integrate alerting (email/Slack) when KPIs cross thresholds.
    - Add scheduled refresh and caching strategies.
    - Provide export endpoints (CSV / API) for dashboards.
    - Role-based access controls for C-level vs analysts.
    """)
    st.info("This is the central operations & alerting hub placeholder.")

# ---------- Footer: sample data ----------
with st.expander("Show sample rows (first 100)"):
    st.dataframe(df.head(100))
