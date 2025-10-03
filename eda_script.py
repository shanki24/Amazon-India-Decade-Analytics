# scripts/eda_script.py
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import unicodedata
import re

# small compatibility shim: some older code expected np.bool attribute; avoid breaking imports
if not hasattr(np, "bool"):
    np.bool = bool

import matplotlib.pyplot as plt
import seaborn as sns

# optional treemap dependency
try:
    import squarify
    HAS_SQUARIFY = True
except Exception:
    HAS_SQUARIFY = False
    print("[WARN] 'squarify' not installed — treemap will fallback to bar chart. Install with: pip install squarify")

import plotly.express as px
import plotly.io as pio
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sns.set(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (12,7)


TXN_CSV = Path('/Users/shashankshandilya/Desktop/amazon_decade_project/cleaned/transactions_cleaned.csv')
PROD_CSV = Path('/Users/shashankshandilya/Desktop/amazon_decade_project/cleaned/products_cleaned.csv')
OUT_IMG = Path('/Users/shashankshandilya/Desktop/amazon_decade_project/images')
OUT_SUM = Path('/Users/shashankshandilya/Desktop/amazon_decade_project/outputs')
OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_SUM.mkdir(parents=True, exist_ok=True)

def savefig(fig, fname, tight=True):
    path = OUT_IMG / fname
    if isinstance(fig, plt.Figure):
        if tight: fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    else:
        if fname.endswith('.html'):
            pio.write_html(fig, file=str(path), auto_open=False)
        else:
            try:
                pio.write_image(fig, str(path))
            except Exception as e:
                alt = str(path.with_suffix('.html'))
                print(f"[WARN] Plotly image write failed ({e}). Falling back to HTML: {alt}")
                pio.write_html(fig, file=alt, auto_open=False)
    print("[SAVED]", path)


def _clean_colname_raw(s):
    """Turn a column name into safe ascii lowercase with underscores."""
    if s is None:
        return ''
    s = str(s)
    # remove BOM & NBSP
    s = s.replace('\ufeff', '').replace('\xa0', ' ')
    s = unicodedata.normalize('NFKD', s)
    s = s.encode('ascii', 'ignore').decode('ascii')
    s = s.strip().lower()
    s = re.sub(r'[^0-9a-z]+', '_', s)
    s = re.sub(r'__+', '_', s)
    s = s.strip('_')
    return s

def _normalize_name_for_match(s: str) -> str:
    if s is None:
        return ''
    return _clean_colname_raw(s).replace('_', '')

def find_col(df, candidates):
    """Return first candidate column name that exists in df (robust to weird chars)."""
    if df is None or len(df.columns) == 0:
        return None

    cleaned_to_orig = {}
    for c in df.columns:
        k = _clean_colname_raw(c)
        if k not in cleaned_to_orig:
            cleaned_to_orig[k] = c

    # exact cleaned match
    for cand in candidates:
        if not cand:
            continue
        ck = _clean_colname_raw(cand)
        if ck in cleaned_to_orig:
            return cleaned_to_orig[ck]

    # token-insensitive (no underscores)
    map_no_unders = {k.replace('_',''): v for k,v in cleaned_to_orig.items()}
    for cand in candidates:
        if not cand:
            continue
        ct = _normalize_name_for_match(cand)
        if ct in map_no_unders:
            return map_no_unders[ct]
        for k_no, orig in map_no_unders.items():
            if ct in k_no or k_no in ct:
                return orig

    # last-resort: ensure all parts in column name
    for cand in candidates:
        if not cand:
            continue
        parts = [p for p in re.split(r'[\s_\-]+', str(cand).lower()) if p]
        for c in df.columns:
            if all(p in c.lower() for p in parts):
                return c

    return None

# Loading the Data
print("Loading transactions:", TXN_CSV)
df = pd.read_csv(TXN_CSV, low_memory=False)
print("Loading products:", PROD_CSV)
prod = pd.read_csv(PROD_CSV, low_memory=False)

# canonicalise column names aggressively (ASCII-only, underscore separated)
df.columns = [_clean_colname_raw(c) for c in df.columns]
prod.columns = [_clean_colname_raw(c) for c in prod.columns]

# debug: show final columns so you can confirm the names the script sees
print("Transactions columns:", df.columns.tolist())
print("Products columns:", prod.columns.tolist())

# ensure date & numeric
if 'order_date' in df.columns:
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
else:
    date_cols = [c for c in df.columns if 'date' in c]
    if date_cols:
        df['order_date'] = pd.to_datetime(df[date_cols[0]], errors='coerce')

if 'final_amount' not in df.columns and 'final_amount_inr' in df.columns:
    df['final_amount'] = pd.to_numeric(df['final_amount_inr'].astype(str).str.replace('[^0-9.\-]','', regex=True), errors='coerce')
else:
    df['final_amount'] = pd.to_numeric(df.get('final_amount'), errors='coerce')

# canonicalize product_id if present
if 'product_id' in df.columns:
    df['product_id'] = df['product_id'].astype(str).str.strip().str.upper()
if 'product_id' in prod.columns:
    prod['product_id'] = prod['product_id'].astype(str).str.strip().str.upper()

# derived cols
df['order_year'] = df['order_date'].dt.year
df['order_month'] = df['order_date'].dt.month
df['order_ym'] = df['order_date'].dt.to_period('M').astype(str)

# Helpers
def safe_merge_category(left_df, prod_df, prod_cat_col_candidates):
    """Return merged df and detected category column name canonicalized to 'category' (or None)."""
    cat_col = find_col(prod_df, prod_cat_col_candidates)
    if not cat_col:
        print(f"[safe_merge_category] could not detect category column. Tried candidates: {prod_cat_col_candidates}. Product columns: {prod_df.columns.tolist()}")
        return left_df.copy(), None

    if 'product_id' not in left_df.columns or 'product_id' not in prod_df.columns:
        print("[safe_merge_category] missing product_id column in one of the frames.")
        return left_df.copy(), None

    # right side sample debug
    try:
        sample_vals = prod_df[[ 'product_id', cat_col ]].head(5)
    except Exception:
        sample_vals = None
    print(f"[safe_merge_category] detected product column = {cat_col}. sample head:\n{sample_vals}")

    right = prod_df[['product_id', cat_col]].drop_duplicates()
    merged = left_df.merge(right, on='product_id', how='left')

    # if no values present, treat as not found
    if cat_col not in merged.columns or merged[cat_col].isna().all():
        print(f"[safe_merge_category] after merge, column {cat_col} missing or all-NA in merged. Merged columns: {merged.columns.tolist()}")
        return left_df.copy(), None

    merged = merged.rename(columns={cat_col: 'category'})
    return merged, 'category'

# ---------- Q1 ----------
def q1():
    yearly = df.groupby('order_year', dropna=True)['final_amount'].sum().sort_index().fillna(0)
    pct = yearly.pct_change().fillna(0) * 100
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(yearly.index, yearly.values, marker='o', label='Revenue (INR)')
    valid = ~np.isnan(yearly.values)
    yrs = np.array(yearly.index[valid], dtype=float)
    vals = np.array(yearly.values[valid], dtype=float)
    if len(yrs) >= 2:
        coef = np.polyfit(yrs, vals, 1)
        trend = np.poly1d(coef)
        ax.plot(yrs, trend(yrs), linestyle='--', color='orange', label='Linear trend')
    topg = pct.sort_values(ascending=False).head(3)
    for y, g in topg.items():
        if y in yearly.index:
            ax.annotate(f"+{g:.1f}%", xy=(y, yearly.loc[y]), xytext=(y, yearly.loc[y]*1.05),
                        arrowprops=dict(arrowstyle='->', color='green'))
    ax.set_title("Yearly Revenue (2015-2025) with YoY % Growth")
    ax.set_ylabel("Revenue (INR)")
    ax.set_xlabel("Year")
    ax.grid(True, alpha=0.25)
    ax2 = ax.twinx()
    ax2.bar(yearly.index, pct.values, alpha=0.12, color='green', label='YoY %')
    ax2.set_ylabel("YoY Growth (%)")
    fig.legend(loc='upper left')
    savefig(fig, 'Q01_yearly_revenue_growth.png')
    out = pd.DataFrame({'year':yearly.index, 'revenue':yearly.values, 'pct_growth':pct.values})
    out.to_csv(OUT_SUM / 'Q01_yearly_revenue.csv', index=False)
    print("Q1 done")

# ---------- Q2 ----------
def q2():
    pivot = df.pivot_table(values='final_amount', index='order_year', columns='order_month', aggfunc='sum', fill_value=0)
    fig, ax = plt.subplots(figsize=(12,6))
    sns.heatmap(pivot, cmap='YlGnBu', ax=ax, fmt=".0f", cbar_kws={'format':'%0.0f'})
    ax.set_title('Monthly Revenue Heatmap')
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')
    savefig(fig, 'Q02_monthly_heatmap.png')

    monthly_tot = df.groupby(df['order_date'].dt.month)['final_amount'].sum().sort_values(ascending=False)
    monthly_tot.index = monthly_tot.index.map(int)
    monthly_tot.to_csv(OUT_SUM / 'Q02_monthly_totals.csv')

    merged, cat_col = safe_merge_category(df, prod, ['category','product_category','category_name','cat'])
    if cat_col and cat_col in merged.columns:
        if merged[cat_col].notna().any():
            top_cats = merged.groupby(cat_col)['final_amount'].sum().nlargest(6).index.tolist()
            for cat in top_cats:
                m = merged[merged[cat_col]==cat].pivot_table(values='final_amount', index='order_year', columns=merged['order_date'].dt.month, aggfunc='sum', fill_value=0)
                fig2, ax2 = plt.subplots(figsize=(10,4))
                sns.heatmap(m, ax=ax2, cmap='YlOrBr', cbar=False)
                ax2.set_title(f'Seasonality heatmap - {cat}')
                savefig(fig2, f'Q02_seasonality_{str(cat)[:30].replace(" ","_")}.png')
        else:
            print("Q2: Category column exists but has no values — skipping per-category seasonality.")
    else:
        print("Q2: No category column found in products — skipping per-category seasonality.")
    print("Q2 done")

# ---------- Q3 (RFM segmentation) ----------
def q3(k=4):
    if 'customer_id' not in df.columns or 'order_date' not in df.columns:
        print("Q3 skipped (no customer_id/order_date)")
        return
    snapshot = df['order_date'].max() + pd.Timedelta(days=1)
    grouped = df.groupby('customer_id').agg(
        recency = ('order_date', lambda x: (snapshot - x.max()).days),
        frequency = ('order_date', 'count'),
        monetary = ('final_amount', 'sum')
    ).reset_index()
    if grouped.empty:
        print("Q3: no customer data after grouping, skipping.")
        return
    grouped['monetary'] = grouped['monetary'].fillna(0)
    X = grouped[['recency','frequency','monetary']].copy()
    X['monetary'] = X['monetary'].clip(lower=0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    grouped['segment'] = km.fit_predict(Xs)

    sample_df = grouped.sample(n=min(len(grouped), 20000), random_state=42).copy()
    sample_df['segment'] = sample_df['segment'].astype(str)

    fig = px.scatter(sample_df, x='frequency', y='monetary', color='segment',
                     title='RFM Scatter (Frequency vs Monetary) — segments',
                     labels={'monetary':'Monetary (INR)','frequency':'Frequency'})
    savefig(fig, 'Q03_rfm_scatter.html')

    segsum = grouped.groupby('segment').agg(count=('customer_id','count'), avg_recency=('recency','mean'), avg_freq=('frequency','mean'), avg_monetary=('monetary','mean')).reset_index()
    segsum.to_csv(OUT_SUM / 'Q03_rfm_segments_summary.csv', index=False)
    print("Q3 done")

# ---------- Q4 ----------
def q4():
    pay_cols = [c for c in df.columns if ('payment' in c or 'pay' in c) and 'method' in c]
    pay_col = pay_cols[0] if pay_cols else (find_col(df, ['payment_method','payment','pay_method','mode_of_payment','pay_type']))
    if not pay_col:
        print("Q4 skipped (no payment column)")
        return

    ts = df.groupby([df['order_date'].dt.to_period('M').astype(str), df[pay_col]]).size().unstack(fill_value=0)
    ts.index = pd.to_datetime(ts.index + '-01')

    fig, ax = plt.subplots(figsize=(12,6))
    ts.plot.area(ax=ax, linewidth=0)
    ax.set_title('Payment Methods Evolution (counts over time)')
    savefig(fig, 'Q04_payment_methods_area.png')

    ts_pct = ts.div(ts.sum(axis=1), axis=0).fillna(0)
    df_pct = ts_pct.reset_index()
    idx_col = df_pct.columns[0]
    df_melt = df_pct.melt(id_vars=idx_col, var_name='method', value_name='share')
    df_melt = df_melt.rename(columns={idx_col: 'period'})

    fig2 = px.area(df_melt, x='period', y='share', color='method', title='Payment Methods Market Share (%)')
    savefig(fig2, 'Q04_payment_methods_share.html')

    ts.to_csv(OUT_SUM / 'Q04_payment_methods_timeseries.csv')
    print("Q4 done")

# ---------- Q5 ----------
def q5():
    merged, cat_col = safe_merge_category(df, prod,
                                          ['category','product_category','category_name','cat','subcategory'])
    if not cat_col or cat_col not in merged.columns:
        print("Q5 skipped (no valid category column found in products).")
        return

    cat_rev = merged.groupby(cat_col)['final_amount'].sum().sort_values(ascending=False)
    if cat_rev.empty:
        print("Q5: Category column present but has no data — skipping.")
        return

    cat_df = cat_rev.reset_index().rename(columns={cat_col:'category','final_amount':'revenue'})
    cat_df.to_csv(OUT_SUM / 'Q05_category_revenue.csv', index=False)

    top = cat_df.head(20)

    if HAS_SQUARIFY:
        fig, ax = plt.subplots(figsize=(14,7))
        squarify.plot(sizes=top['revenue'],
                      label=[f"{a}\n₹{b:,.0f}" for a,b in zip(top['category'], top['revenue'])],
                      alpha=0.8, ax=ax)
        ax.axis('off'); ax.set_title('Treemap: Top Categories by Revenue')
        savefig(fig, 'Q05_treemap_categories.png')
    else:
        fig, ax = plt.subplots(figsize=(14,7))
        sns.barplot(x='revenue', y='category', data=top, palette='viridis', ax=ax)
        ax.set_title('Top Categories by Revenue (fallback bar)')
        savefig(fig, 'Q05_treemap_fallback_bar.png')

    fig2, ax2 = plt.subplots(figsize=(10,8))
    sns.barplot(x='revenue', y='category', data=top, palette='muted', ax=ax2)
    ax2.set_title('Top Categories by Revenue (bar)')
    savefig(fig2, 'Q05_bar_categories.png')

    pie = cat_df.head(10)
    fig3, ax3 = plt.subplots(figsize=(8,8))
    ax3.pie(pie['revenue'], labels=pie['category'], autopct='%1.1f%%', startangle=120)
    ax3.set_title('Category Market Share (Top 10)')
    savefig(fig3, 'Q05_pie_categories.png')

    print("Q5 done")

# ---------- Q6 ----------
def q6():
    prime_col = find_col(df, ['is_prime_member','is_prime','prime','prime_member'])
    if not prime_col:
        print("Q6 skipped (no prime column)")
        return

    s = df.copy()
    s[prime_col] = s[prime_col].astype(str).str.lower().isin(['true','1','yes','y','t'])

    aov = s.groupby(prime_col)['final_amount'].agg(['sum','count','mean']).reset_index().rename(columns={prime_col:'is_prime'})
    aov.to_csv(OUT_SUM / 'Q06_prime_aov.csv', index=False)
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x='is_prime', y='mean', data=aov, ax=ax)
    ax.set_title('Avg Order Value: Prime vs Non-Prime')
    savefig(fig, 'Q06_prime_aov.png')

    if 'customer_id' in s.columns:
        freq = s.groupby(['customer_id', prime_col]).size().reset_index(name='orders')
        freq_summary = freq.groupby(prime_col)['orders'].mean().reset_index().rename(
            columns={prime_col:'is_prime','orders':'avg_orders_per_customer'})
        freq_summary.to_csv(OUT_SUM / 'Q06_prime_freq_summary.csv', index=False)

    merged, cat_col = safe_merge_category(s, prod, ['category','product_category','category_name','cat','subcategory'])
    if cat_col and cat_col in merged.columns:
        if merged[cat_col].notna().any():
            pref = merged.groupby([prime_col, cat_col])['final_amount'].sum().reset_index().rename(
                columns={prime_col:'is_prime', cat_col:'category'})
            pref.to_csv(OUT_SUM / 'Q06_prime_category_prefs.csv', index=False)
        else:
            print("Q6: category column exists but empty — skipping category prefs.")
    else:
        print("Q6: no category info available — skipping category prefs.")

    print("Q6 done")

# ---------- Q7 ----------
def q7(geojson_path=None):
    city_col = find_col(df, ['customer_city','city','ship_city','billing_city'])
    state_col = find_col(df, ['customer_state','state','region'])
    if state_col:
        state_rev = df.groupby(state_col)['final_amount'].sum().sort_values(ascending=False).reset_index()
        state_rev.to_csv(OUT_SUM / 'Q07_state_revenue.csv', index=False)
        fig, ax = plt.subplots(figsize=(12,8))
        sns.barplot(x='final_amount', y=state_col, data=state_rev.head(20), ax=ax)
        ax.set_title('Top 20 States by Revenue')
        savefig(fig, 'Q07_top_states.png')
    if city_col:
        city_rev = df.groupby(city_col)['final_amount'].sum().sort_values(ascending=False).reset_index()
        city_rev.to_csv(OUT_SUM / 'Q07_city_revenue.csv', index=False)
        fig2, ax2 = plt.subplots(figsize=(12,10))
        sns.barplot(x='final_amount', y=city_col, data=city_rev.head(30), ax=ax2)
        ax2.set_title('Top 30 Cities by Revenue')
        savefig(fig2, 'Q07_top_cities.png')
    if geojson_path and state_col:
        try:
            import json
            geo = json.load(open(geojson_path))
            fig_map = px.choropleth(state_rev, geojson=geo, locations=state_col, color='final_amount',
                                    featureidkey='properties.NAME', projection='mercator',
                                    title='State revenue choropleth (requires mapping)')
            fig_map.update_geos(fitbounds="locations", visible=False)
            savefig(fig_map, 'Q07_state_choropleth.html')
        except Exception as e:
            print("Could not build choropleth:", e)
    print("Q7 done")

# ---------- Q8 ----------
def q8(festival_calendar=None):
    if festival_calendar is None:
        festival_calendar = {'Diwali': None}
    results = []
    years = sorted(df['order_year'].dropna().unique())
    for y in years:
        try:
            anchor = pd.to_datetime(f"{int(y)}-10-25")
            before = df[(df['order_date'] >= anchor - pd.Timedelta(days=14)) & (df['order_date'] < anchor)]['final_amount'].sum()
            during = df[(df['order_date'] >= anchor) & (df['order_date'] <= anchor + pd.Timedelta(days=7))]['final_amount'].sum()
            after = df[(df['order_date'] > anchor + pd.Timedelta(days=7)) & (df['order_date'] <= anchor + pd.Timedelta(days=30))]['final_amount'].sum()
            results.append({'year':int(y), 'before':before, 'during':during, 'after':after})
        except Exception:
            pass
    resdf = pd.DataFrame(results)
    if not resdf.empty:
        resdf.to_csv(OUT_SUM / 'Q08_diwali_window.csv', index=False)
        fig = px.bar(resdf.melt(id_vars='year', var_name='window', value_name='revenue'), x='year', y='revenue', color='window', barmode='group', title='Diwali approx. window revenue (before/during/after)')
        savefig(fig, 'Q08_diwali_window.html')
    print("Q8 done (approximate).")

# ---------- Q9 ----------
def q9():
    age_col = find_col(df, ['age','customer_age'])
    if not age_col:
        print("Q9 skipped (no age column)")
        return
    s = df.copy()
    s[age_col] = pd.to_numeric(s[age_col], errors='coerce')
    bins = [0,18,25,35,45,55,65,200]
    labels = ['<18','18-24','25-34','35-44','45-54','55-64','65+']
    s['age_bucket'] = pd.cut(s[age_col], bins=bins, labels=labels)
    spend = s.groupby('age_bucket')['final_amount'].agg(['sum','count','mean']).reset_index()
    spend.to_csv(OUT_SUM / 'Q09_age_spending.csv', index=False)
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x='age_bucket', y='mean', data=spend, order=labels, ax=ax)
    ax.set_title('Avg spending by age bucket')
    savefig(fig, 'Q09_spending_by_age.png')
    print("Q9 done")

# ---------- Q10 ----------
def q10():
    if 'product_id' not in df.columns:
        print("Q10 skipped (no product_id)")
        return
    price_col = find_col(prod, ['price','mrp','selling_price','original_price','list_price'])
    merged = df.copy()
    if price_col:
        merged = merged.merge(prod[['product_id', price_col]].drop_duplicates(), on='product_id', how='left')
        merged[price_col] = pd.to_numeric(merged[price_col], errors='coerce')
    else:
        price_col = 'avg_order_price'
        merged = merged.groupby('product_id').agg(avg_order_price=('final_amount','mean')).reset_index().merge(merged, on='product_id', how='right')

    demand = merged.groupby('product_id').agg(price=(price_col,'first'), orders=('transaction_id' if 'transaction_id' in merged.columns else 'order_date','count'), revenue=('final_amount','sum')).reset_index()
    demand['price'] = pd.to_numeric(demand['price'], errors='coerce')
    demand = demand.dropna(subset=['price'])
    corr = demand[['price','orders','revenue']].corr()
    corr.to_csv(OUT_SUM / 'Q10_price_demand_corr.csv')
    fig = px.scatter(demand.sample(min(len(demand),20000)), x='price', y='orders', size='revenue', hover_name='product_id', title='Price vs Orders (bubble size = revenue)')
    savefig(fig, 'Q10_price_vs_demand.html')
    print("Q10 done")

# ---------- Q11 ----------
def q11():
    if 'delivery_days' not in df.columns:
        print("Q11 skipped (no delivery_days)")
        return
    dd = pd.to_numeric(df['delivery_days'], errors='coerce').dropna()
    fig, ax = plt.subplots(figsize=(10,6))
    sns.histplot(dd, bins=40, kde=False, ax=ax)
    ax.set_title('Delivery Days Distribution')
    savefig(fig, 'Q11_delivery_days_dist.png')
    ontime_prop = (dd <= 7).mean()
    with open(OUT_SUM / 'Q11_delivery_summary.txt', 'w') as f:
        f.write(f"On-time (<=7 days) proportion: {ontime_prop:.3f}\n")
    print("Q11 done")

# ---------- Q12 ----------
def q12():
    ret_col = find_col(df, ['is_returned','returned','return_status','return'])
    if not ret_col:
        print("Q12 skipped (no return column)")
        return
    r = df.groupby(ret_col)['final_amount'].agg(['count','sum']).reset_index()
    r.to_csv(OUT_SUM / 'Q12_returns_summary.csv', index=False)
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=ret_col, y='count', data=r, ax=ax)
    ax.set_title('Returns by status')
    savefig(fig, 'Q12_returns_by_status.png')
    rating_col = find_col(df, ['rating','product_rating','customer_rating'])
    if rating_col:
        corrmat = df[[ret_col, rating_col, 'final_amount']].dropna()
        try:
            corrmat[ret_col] = corrmat[ret_col].apply(lambda x: 1 if str(x).lower() in ['1','true','yes','y','t'] else 0)
            corr = corrmat.corr()
            corr.to_csv(OUT_SUM / 'Q12_return_rating_corr.csv')
        except Exception:
            pass
    print("Q12 done")

# ---------- Q13 ----------
def q13():
    brand_col = find_col(prod, ['brand','manufacturer','maker'])
    print("[Q13] detected brand_col:", brand_col)
    if not brand_col or brand_col not in prod.columns or 'product_id' not in df.columns:
        print("Q13 skipped (no brand/product info available). Detected brand_col:", brand_col, "Prod cols:", prod.columns.tolist())
        return

    merged = df.merge(prod[['product_id', brand_col]], on='product_id', how='left')
    print("[Q13] merged columns sample:", merged.columns.tolist()[:20])

    # double-check column exists after merge and has values
    if brand_col not in merged.columns:
        print("Q13 skipped (brand column missing after merge).")
        return

    # show a tiny sample for debugging
    print("[Q13] merged brand sample:\n", merged[[ 'product_id', brand_col ]].drop_duplicates().head(10))

    if merged[brand_col].isna().all():
        print("Q13 skipped (brand column exists but has no values).")
        return

    brand_rev = merged.groupby(brand_col)['final_amount'].sum().sort_values(ascending=False)
    if brand_rev.empty:
        print("Q13: brand data empty — skipping.")
        return

    brand_rev.head(20).to_csv(OUT_SUM / 'Q13_brand_top20.csv', header=['revenue'])

    fig, ax = plt.subplots(figsize=(12,8))
    sns.barplot(y=brand_rev.head(15).index, x=brand_rev.head(15).values, palette='deep', ax=ax)
    ax.set_title('Top 15 Brands by Revenue')
    savefig(fig, 'Q13_top_brands.png')

    top_brands = brand_rev.head(5).index.tolist()
    ts = merged[merged[brand_col].isin(top_brands)].groupby(
        [merged['order_date'].dt.to_period('M').astype(str), brand_col]
    )['final_amount'].sum().unstack(fill_value=0)

    if not ts.empty:
        ts.index = pd.to_datetime(ts.index + '-01')
        df_area = ts.reset_index().melt(id_vars='index', var_name='brand', value_name='revenue')
        fig2 = px.area(df_area, x='index', y='revenue', color='brand', title='Top Brands Market Share (area)')
        savefig(fig2, 'Q13_brand_market_share.html')
    else:
        print("Q13: no time series data for brands — skipping area chart.")
    print("Q13 done")

# ---------- Q14 ----------
def q14():
    if 'customer_id' not in df.columns or 'order_date' not in df.columns:
        print("Q14 skipped (no customer_id/order_date)")
        return
    users = df[['customer_id','order_date','final_amount']].dropna(subset=['customer_id'])
    users['order_month'] = users['order_date'].dt.to_period('M').astype(str)
    users['cohort_month'] = users.groupby('customer_id')['order_date'].transform('min').dt.to_period('M').astype(str)
    cohort_counts = users.groupby(['cohort_month','order_month'])['customer_id'].nunique().reset_index()
    cohort_pivot = cohort_counts.pivot(index='cohort_month', columns='order_month', values='customer_id').fillna(0)
    cohort_pivot.to_csv(OUT_SUM / 'Q14_cohort_retention_counts.csv')

    cohort_sizes = cohort_pivot.iloc[:,0]
    retention = cohort_pivot.divide(cohort_sizes, axis=0)

    fig = px.imshow(
        retention.to_numpy(dtype=float),
        labels=dict(x="Order month", y="Cohort month", color="Retention"),
        x=retention.columns,
        y=retention.index,
        title='Cohort retention heatmap'
    )
    savefig(fig, 'Q14_cohort_retention.html')

    clv = users.groupby('cohort_month')['final_amount'].sum() / users.groupby('cohort_month')['customer_id'].nunique()
    clv = clv.reset_index().rename(columns={0:'clv','final_amount':'clv'})
    clv.to_csv(OUT_SUM / 'Q14_cohort_clv.csv', index=False)
    print("Q14 done")

# ---------- Q15 ----------
def q15():
    orig_col = find_col(df, ['original_price','original_price_inr','list_price','mrp','price_before_discount'])
    if not orig_col:
        print("Q15 skipped (no original price column)")
        return
    df['orig_price_num'] = pd.to_numeric(df[orig_col].astype(str).str.replace('[^0-9.\-]','',regex=True), errors='coerce')
    df['disc_pct'] = (df['orig_price_num'] - df['final_amount']) / df['orig_price_num'] * 100
    d = df.dropna(subset=['disc_pct'])
    d.to_csv(OUT_SUM / 'Q15_discount_full.csv', index=False)

    try:
        import statsmodels
        fig = px.scatter(d.sample(min(len(d),5000)), x='disc_pct', y='final_amount', trendline='lowess', title='Discount % vs Final Amount (sample)')
    except Exception:
        fig = px.scatter(d.sample(min(len(d),5000)), x='disc_pct', y='final_amount', title='Discount % vs Final Amount (sample) — no LOWESS (statsmodels missing)')
    savefig(fig, 'Q15_discount_vs_amount.html')
    print("Q15 done")

# ---------- Q16 ----------
def q16():
    rating_col = find_col(df, ['rating','product_rating','customer_rating'])
    if not rating_col:
        print("Q16 skipped (no rating)")
        return
    r = df.dropna(subset=[rating_col])
    r[rating_col] = pd.to_numeric(r[rating_col], errors='coerce')
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(r[rating_col].dropna(), bins=20, ax=ax)
    ax.set_title('Rating distribution')
    savefig(fig, 'Q16_rating_dist.png')
    if 'product_id' in df.columns:
        rr = r.groupby('product_id').agg(avg_rating=(rating_col,'mean'), revenue=('final_amount','sum')).dropna()
        rr.to_csv(OUT_SUM / 'Q16_rating_product_level.csv')
        fig2 = px.scatter(rr.sample(min(len(rr),5000)), x='avg_rating', y='revenue', title='Avg product rating vs revenue')
        savefig(fig2, 'Q16_rating_vs_revenue.html')
    print("Q16 done")

# ---------- Q17 ----------
def q17():
    if 'customer_id' not in df.columns:
        print("Q17 skipped (no customer_id)")
        return
    merged, cat_col = safe_merge_category(df, prod, ['category','product_category','category_name','cat'])
    if not cat_col:
        print("Q17 skipped (no category info)")
        return
    merged = merged.sort_values(['customer_id','order_date'])
    transitions = {}
    last = {}
    for _, row in merged[['customer_id','category']].iterrows():
        cid = row['customer_id']; cat = row['category']
        if pd.isna(cid) or pd.isna(cat): continue
        if cid in last:
            prev = last[cid]
            transitions[(prev, cat)] = transitions.get((prev, cat), 0) + 1
        last[cid] = cat
    trans_df = pd.DataFrame([{'from':k[0], 'to':k[1], 'count':v} for k,v in transitions.items()]).sort_values('count', ascending=False)
    trans_df.to_csv(OUT_SUM / 'Q17_category_transitions.csv', index=False)
    top = trans_df.head(30)
    if not top.empty:
        fig, ax = plt.subplots(figsize=(10,8))
        sns.barplot(x='count', y=top.apply(lambda x: f"{x['from']} -> {x['to']}", axis=1), data=top, ax=ax)
        ax.set_title('Top category transitions (customer journeys)')
        savefig(fig, 'Q17_top_transitions.png')
    else:
        print("Q17: no transitions to plot.")
    print("Q17 done")

# ---------- Q18 ----------
def q18():
    if 'product_id' not in df.columns:
        print("Q18 skipped (no product_id)")
        return
    pstats = df.groupby('product_id').agg(first_sale=('order_date','min'), last_sale=('order_date','max'), sales_count=('order_date','count'), revenue=('final_amount','sum')).reset_index()
    pstats['lifetime_days'] = (pd.to_datetime(pstats['last_sale']) - pd.to_datetime(pstats['first_sale'])).dt.days
    pstats.to_csv(OUT_SUM / 'Q18_product_lifecycle.csv', index=False)
    top = pstats.sort_values('revenue', ascending=False).head(10)['product_id'].tolist()
    fig, ax = plt.subplots(figsize=(12,8))
    for pid in top:
        tmp = df[df['product_id']==pid].groupby(df['order_date'].dt.to_period('M'))['final_amount'].sum()
        if tmp.empty: continue
        tmp.index = pd.to_datetime(tmp.index.astype(str)+'-01')
        ax.plot(tmp.index, tmp.values, label=str(pid))
    ax.legend(); ax.set_title('Monthly revenue evolution for top products')
    savefig(fig, 'Q18_top_product_lifecycle.png')
    print("Q18 done")

# ---------- Q19 (fixed plotting) ----------
def q19():
    price_col = find_col(prod, ['price','mrp','selling_price','original_price','list_price'])
    brand_col = find_col(prod, ['brand','manufacturer'])
    print("[Q19] detected price_col:", price_col, "brand_col:", brand_col)
    if not price_col or not brand_col:
        print("Q19 skipped (no brand/price). Detected price_col:", price_col, "brand_col:", brand_col, "prod cols:", prod.columns.tolist())
        return

    merged = df.merge(prod[['product_id', price_col, brand_col]], on='product_id', how='left')

    # defensive: if brand_col somehow not present after merge, try to find it on merged
    if brand_col not in merged.columns:
        print("[Q19] brand_col missing in merged; merged cols sample:", merged.columns.tolist()[:40])
        brand_col = find_col(merged, ['brand','manufacturer'])
        print("[Q19] re-detected brand_col in merged:", brand_col)
        if not brand_col:
            print("Q19 skipped (brand column missing after merge).")
            return

    # coerce types & drop NA rows for plotting
    merged[price_col] = pd.to_numeric(merged[price_col], errors='coerce')
    merged['final_amount'] = pd.to_numeric(merged['final_amount'], errors='coerce')
    # ensure brand is string for grouping
    merged[brand_col] = merged[brand_col].astype(str).fillna('')

    data = merged.dropna(subset=[price_col, brand_col])
    data = data[data[brand_col] != '']  # drop blank brands

    if data.empty:
        print("Q19: no data available after cleaning for plotting.")
        return

    n_brands = data[brand_col].nunique()
    print(f"[Q19] plotting data: {len(data)} rows, {n_brands} unique brands (top 8 will be used).")
    if n_brands < 2:
        print("Q19: fewer than 2 brands found — skipping boxplot.")
        return

    # select top brands by revenue for stable plotting
    brand_rev_series = data.groupby(brand_col)['final_amount'].sum().sort_values(ascending=False)
    top_brands = brand_rev_series.head(8).index.tolist()
    data = data[data[brand_col].isin(top_brands)]

    if data.empty:
        print("Q19: after filtering to top brands, no data — skipping.")
        return

    # final safety: ensure each group has at least one numeric price
    grp_counts = data.groupby(brand_col)[price_col].count()
    grp_valid = grp_counts[grp_counts > 0]
    if grp_valid.empty or len(grp_valid) < 2:
        print("Q19: not enough groups with numeric prices to plot. counts:", grp_counts.to_dict())
        return

    fig, ax = plt.subplots(figsize=(12,8))
    sns.boxplot(x=brand_col, y=price_col, data=data, ax=ax)
    ax.set_title('Price distribution by brand (top brands)')
    savefig(fig, 'Q19_price_by_brand_boxplot.png')
    print("Q19 done")

# ---------- Q20 ----------
def q20():
    total_revenue = df['final_amount'].sum()
    orders = len(df)
    customers = df['customer_id'].nunique() if 'customer_id' in df.columns else None
    avg_order_value = df['final_amount'].mean()
    yoy = None
    if 'order_year' in df.columns:
        annual = df.groupby('order_year')['final_amount'].sum().sort_index()
        yoy = annual.pct_change().iloc[-1] if len(annual)>1 else np.nan
    fig, axes = plt.subplots(2,2, figsize=(16,10))
    axes[0,0].axis('off')
    text = f"Total revenue: ₹{total_revenue:,.0f}\nOrders: {orders:,}\nCustomers: {customers}\nAOV: ₹{avg_order_value:,.2f}\nYoY (last): {pd.Series([yoy]).iloc[0] if yoy is not None else 'N/A'}"
    axes[0,0].text(0,0.5, text, fontsize=14)
    if 'order_year' in df.columns:
        annual = df.groupby('order_year')['final_amount'].sum().sort_index()
        axes[0,1].plot(annual.index, annual.values, marker='o')
        axes[0,1].set_title('Annual Revenue')
    if 'order_ym' in df.columns:
        aov = df.groupby('order_ym')['final_amount'].mean()
        aov.index = pd.to_datetime(aov.index.astype(str)+'-01')
        axes[1,0].plot(aov.index, aov.values)
        axes[1,0].set_title('AOV over Time')
        axes[1,0].tick_params(axis='x', rotation=45)
    merged, cat_col = safe_merge_category(df, prod, ['category','product_category','category_name','cat'])
    if cat_col:
        cat = merged.groupby('category')['final_amount'].sum().nlargest(10)
        axes[1,1].barh(cat.index, cat.values)
        axes[1,1].set_title('Top 10 Categories by Revenue')
    plt.suptitle('Business Health Dashboard')
    savefig(fig, 'Q20_business_health_dashboard.png')

    kpis = pd.DataFrame({
        'metric':['total_revenue','orders','customers','avg_order_value','yoy_last'],
        'value':[total_revenue, orders, customers, avg_order_value, float(yoy) if yoy is not None and not pd.isna(yoy) else None]
    })
    kpis.to_csv(OUT_SUM / 'Q20_kpis.csv', index=False)
    print("Q20 done")

# Execute
if __name__ == '__main__':
    q1(); q2(); q3(k=4); q4(); q5(); q6(); q7(); q8(); q9(); q10()
    q11(); q12(); q13(); q14(); q15(); q16(); q17(); q18(); q19(); q20()
    print("All Q1-Q20 finished. Images in:", OUT_IMG, "Summaries in:", OUT_SUM)
