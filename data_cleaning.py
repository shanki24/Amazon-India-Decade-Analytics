# scripts/data_cleaning_practice.py
import re
from pathlib import Path
import pandas as pd
import numpy as np
from dateutil import parser
from dotenv import load_dotenv

load_dotenv()

# Paths (user-provided)
TRANSACTIONS_RAW = Path('/Users/shashankshandilya/Desktop/amazon_decade_project/data/amazon_india_complete_2015_2025.csv')
PRODUCTS_RAW     = Path('/Users/shashankshandilya/Desktop/amazon_decade_project/data/amazon_india_products_catalog.csv')

OUT_CLEAN = Path('/Users/shashankshandilya/Desktop/amazon_decade_project/cleaned')
OUT_CLEAN.mkdir(parents=True, exist_ok=True)
OUT = Path('/Users/shashankshandilya/Desktop/amazon_decade_project/outputs')
OUT.mkdir(parents=True, exist_ok=True)

# Utilities
def save_df(df, name):
    p = OUT / name
    df.to_csv(p, index=False)
    print(f"[SAVED] {p}")

def try_parse_date_series(s: pd.Series) -> pd.Series:
    """Robust parsing: try common formats first, then dateutil parser fallback, return datetime64[ns] or NaT."""
    s0 = s.astype(str).fillna('').replace({'nan':''})
    out = pd.to_datetime(pd.Series([pd.NaT]*len(s0)), errors='coerce')
  
    formats = ['%Y-%m-%d','%d/%m/%Y','%d-%m-%y','%d-%m-%Y','%d/%m/%y','%Y/%m/%d','%d %b %Y','%d %B %Y']
    for fmt in formats:
        try:
            parsed = pd.to_datetime(s0, format=fmt, errors='coerce', dayfirst=True)
            out = out.fillna(parsed)
        except Exception:
            pass
    
    mask = out.isna() & (s0 != '')
    if mask.any():
        def p(x):
            try:
                return parser.parse(x, dayfirst=True, fuzzy=True)
            except Exception:
                return pd.NaT
        parsed2 = s0[mask].map(p)
        out.loc[mask] = parsed2.values
    return pd.to_datetime(out, errors='coerce')

def parse_price_string(x):
    """Parse price-like strings: strip currency, handle lakh/crore/k/million, remove commas."""
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if s == '' or re.search(r'price\s*on\s*request|not\s*available|na', s, re.I):
        return np.nan
    # remove rupee sign and common tokens
    s_orig = s
    s = re.sub(r'[₹Rs\.,]', '', s, flags=re.I)  
    
    m = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*(lakh|lac|crore|cr|k|m|million|billion)', str(x), re.I)
    if m:
        num = float(m.group(1))
        unit = m.group(2).lower()
        if unit in ('lakh','lac'): return num * 1e5
        if unit in ('crore','cr'): return num * 1e7
        if unit in ('k',): return num * 1e3
        if unit in ('m','million'): return num * 1e6
        if unit in ('billion',): return num * 1e9
    
    s2 = re.sub(r'[^0-9\.\-]', '', str(x))
   
    if s2.count('.') > 1:
        parts = s2.split('.')
        s2 = parts[0] + '.' + ''.join(parts[1:])
    try:
        return float(s2)
    except:
        return np.nan

def parse_rating_string(x):
    """Standardize rating strings to float 1.0-5.0"""
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if s == '' or s in ('nan','none','n/a'):
        return np.nan
    
    m = re.search(r'([0-9]+(\.[0-9]+)?)\s*/\s*([0-9]+(\.[0-9]+)?)', s)
    if m:
        val = float(m.group(1)); base = float(m.group(3))
        if base > 0 and base != 5:
            return float(val / base * 5.0)
        return val
   
    m2 = re.search(r'([0-9]+(\.[0-9]+)?)', s)
    if m2:
        v = float(m2.group(1))
        if 1.0 <= v <= 5.0:
            return v
     
        if v <= 10:
            return min(5.0, v/2.0)
    return np.nan

# TRANSACTIONS CLEANING (Q1 - Q10)
def clean_transactions():
    if not TRANSACTIONS_RAW.exists():
        print("[TRANSACTIONS] RAW file not found:", TRANSACTIONS_RAW)
        return None
    print("[TRANSACTIONS] Loading RAW:", TRANSACTIONS_RAW)
    df = pd.read_csv(TRANSACTIONS_RAW, low_memory=False, dtype=str)
    df.columns = [c.strip().lower().replace(' ','_') for c in df.columns]
    # ensure final_amount present
    if 'final_amount' not in df.columns and 'final_amount_inr' in df.columns:
        df['final_amount'] = df['final_amount_inr']
    # Q1: dates
    date_col_candidates = [c for c in df.columns if 'date' in c]
    date_col = 'order_date' if 'order_date' in df.columns else (date_col_candidates[0] if date_col_candidates else None)
    if date_col:
        print("[TRANSACTIONS][Q1] Parsing date column:", date_col)
        df[date_col] = try_parse_date_series(df[date_col])
        invalid_dates = df[df[date_col].isna() & df.apply(lambda r: any(str(r[c]).strip()!='' for c in date_col_candidates), axis=1)].head(200)
        if not invalid_dates.empty:
            save_df(invalid_dates[[date_col]+date_col_candidates].drop_duplicates(), 'Q1_invalid_dates_sample.csv')
    # Q2: price cleaning
    price_cols = [c for c in ('original_price_inr','original_price','price','list_price') if c in df.columns]
    if price_cols:
        src = price_cols[0]
        print("[TRANSACTIONS][Q2] Cleaning price column:", src)
        df['original_price_inr_clean'] = df[src].apply(parse_price_string)
        bad = df[df['original_price_inr_clean'].isna() & df[src].notna()].head(200)
        if not bad.empty:
            save_df(bad[[src]].drop_duplicates(), 'Q2_unparsed_price_sample.csv')
    if 'final_amount' in df.columns:
        print("[TRANSACTIONS][Q2] Cleaning final_amount to numeric")
        df['final_amount'] = df['final_amount'].apply(parse_price_string)
    # Q3: ratings
    rating_candidates = [c for c in df.columns if 'rating' in c]
    if rating_candidates:
        for col in rating_candidates:
            print("[TRANSACTIONS][Q3] Cleaning rating column:", col)
            df[col + '_clean'] = df[col].apply(parse_rating_string)
        save_df(df[[c + '_clean' for c in rating_candidates]].head(200), 'Q3_ratings_sample.csv')
        # Optionally replace originals:
        for col in rating_candidates:
            df[col] = df[col + '_clean']; df.drop(columns=[col + '_clean'], inplace=True)
    # Q4: city normalization
    city_col = 'customer_city' if 'customer_city' in df.columns else (next((c for c in df.columns if 'city' in c), None))
    if city_col:
        print("[TRANSACTIONS][Q4] Standardizing city column:", city_col)
        CITY_MAP = {
            'bangalore':'Bengaluru','bengaluru':'Bengaluru','blr':'Bengaluru',
            'mumbai':'Mumbai','bombay':'Mumbai','mum':'Mumbai',
            'delhi':'New Delhi','new delhi':'New Delhi','ncr':'New Delhi',
            'chennai':'Chennai','madras':'Chennai','kolkata':'Kolkata','calcutta':'Kolkata',
            'pune':'Pune','hyderabad':'Hyderabad','ahmedabad':'Ahmedabad',
            'gurgaon':'Gurugram','gurugram':'Gurugram','noida':'Noida'
        }
        def std_city(x):
            if pd.isna(x): return np.nan
            s = str(x).strip().lower()
            s = re.sub(r'[^a-z0-9\s]', ' ', s)
            s = re.sub(r'\s+', ' ', s).strip()
            if s == '' or s in ('nan','none','na'): return np.nan
            if s in CITY_MAP: return CITY_MAP[s]
            for k,v in CITY_MAP.items():
                if k in s: return v
            return s.title()
        df[city_col + '_clean'] = df[city_col].apply(std_city)
        unusual = df[df[city_col + '_clean'].isna() & df[city_col].notna()][city_col].drop_duplicates().head(200)
        if not unusual.empty:
            save_df(unusual.to_frame(name=city_col), 'Q4_unusual_cities_sample.csv')
        df[city_col] = df[city_col + '_clean']; df.drop(columns=[city_col + '_clean'], inplace=True)
    # Q5: booleans
    bool_cols = [c for c in df.columns if any(k in c for k in ['is_prime','prime','is_festival','festival'])]
    if bool_cols:
        print("[TRANSACTIONS][Q5] Normalizing boolean columns:", bool_cols)
        truthy = set(['1','true','yes','y','t','1.0'])
        falsy = set(['0','false','no','n','f','0.0'])
        for c in bool_cols:
            s = df[c].astype(str).fillna('').str.strip().str.lower()
            def to_bool(x):
                if x in truthy: return True
                if x in falsy: return False
                if 'prime' in x and x != '': return True
                if x == '' or x in ('nan','none','na'): return np.nan
                try: return bool(float(x))
                except: return np.nan
            df[c + '_clean'] = s.apply(to_bool)
            missing = df[df[c + '_clean'].isna()][c].drop_duplicates().head(200)
            if not missing.empty:
                save_df(missing.to_frame(name=c), f'Q5_{c}_missing_sample.csv')
            df[c] = df[c + '_clean'].fillna(False).astype(bool); df.drop(columns=[c + '_clean'], inplace=True)
    # Q6: categories - try to bring categories from product catalog later; leave placeholder (done in product cleaning)
    # Q7: delivery_days
    if 'delivery_days' in df.columns:
        print("[TRANSACTIONS][Q7] Cleaning delivery_days")
        def parse_delivery(x):
            if pd.isna(x): return np.nan
            s = str(x).strip().lower()
            if s == '' or s in ('nan','none','na'): return np.nan
            if 'same' in s: return 0
            nums = re.findall(r'(\d+)', s)
            if nums:
                vals = list(map(int, nums))
                v = int(round(np.median(vals)))
                if v < 0 or v > 30: return np.nan
                return v
            return np.nan
        df['delivery_days'] = df['delivery_days'].apply(parse_delivery)
        issues = df[df['delivery_days'].isna() & df['delivery_days'].notna()].head(200)
        if not issues.empty:
            save_df(issues[['delivery_days']].drop_duplicates(), 'Q7_delivery_parse_issues_sample.csv')
    # Q8: duplicates (customer_id, product_id, order_date, final_amount)
    key_cols = [c for c in ('customer_id','product_id','order_date','final_amount') if c in df.columns]
    if key_cols:
        print("[TRANSACTIONS][Q8] Checking duplicates on keys:", key_cols)
        df['_order_date_str'] = pd.to_datetime(df['order_date'], errors='coerce').dt.strftime('%Y-%m-%d') if 'order_date' in df.columns else df.get('order_date', pd.Series(['']*len(df)))
        dup_key = df[key_cols].astype(str).agg('|'.join, axis=1)
        df['_dup_key'] = dup_key
        dup_groups = df.groupby('_dup_key').size()
        duplicate_keys = dup_groups[dup_groups > 1].index.tolist()
        print(f"[TRANSACTIONS][Q8] Duplicate groups found: {len(duplicate_keys)}")
        qty_col = next((c for c in ('quantity','qty','order_quantity') if c in df.columns), None)
        if qty_col:
            grp_qty = df[df['_dup_key'].isin(duplicate_keys)].groupby('_dup_key')[qty_col].apply(lambda s: pd.to_numeric(s, errors='coerce').sum())
            erroneous_keys = grp_qty[grp_qty <= 1].index.tolist()
        else:
            erroneous_keys = duplicate_keys
        if erroneous_keys:
            flagged = df[df['_dup_key'].isin(erroneous_keys)].sort_values('_dup_key').head(500)
            save_df(flagged, 'Q8_flagged_possible_duplicates_sample.csv')
            df = df.drop_duplicates(subset=['_dup_key'], keep='first')
            print(f"[TRANSACTIONS][Q8] Kept first occurrence for {len(erroneous_keys)} duplicate keys (others flagged).")
        df.drop(columns=['_order_date_str','_dup_key'], errors='ignore', inplace=True)
    # Q9: price outliers (product-level median)
    price_col = 'final_amount' if 'final_amount' in df.columns else (next((c for c in df.columns if 'price' in c), None))
    if price_col:
        print("[TRANSACTIONS][Q9] Detecting price outliers")
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
        ref_col = 'product_id' if 'product_id' in df.columns else (next((c for c in df.columns if 'category' in c), None))
        if ref_col and ref_col in df.columns:
            med = df.groupby(ref_col)[price_col].median().rename('ref_median')
            df = df.merge(med, left_on=ref_col, right_index=True, how='left')
        else:
            df['ref_median'] = df[price_col].median()
        def auto_fix(row):
            p = row[price_col]; m = row['ref_median']
            if pd.isna(p) or pd.isna(m) or m<=0: return p, False, None
            if p > m * 50:
                for f in (10, 100, 1000):
                    cand = p / f
                    if m*0.5 <= cand <= m*5:
                        return cand, True, f
                return p, False, 'flag'
            return p, False, None
        fixes = df.apply(lambda r: auto_fix(r), axis=1, result_type='expand')
        fixes.columns = ['_price_new','_price_fixed','_price_factor']
        df = pd.concat([df, fixes], axis=1)
        auto_mask = df['_price_fixed'] == True
        if auto_mask.any():
            save_df(df.loc[auto_mask, ['product_id', price_col, '_price_new', '_price_factor']].head(200), 'Q9_auto_fixed_samples.csv')
            df.loc[auto_mask, price_col] = df.loc[auto_mask, '_price_new']
            print(f"[TRANSACTIONS][Q9] Auto-fixed {auto_mask.sum()} price rows.")
        flagged = df[df['_price_factor'] == 'flag']
        if not flagged.empty:
            save_df(flagged[[price_col,'product_id','ref_median']].head(500), 'Q9_flagged_manual_review.csv')
            print("[TRANSACTIONS][Q9] Some outliers flagged for manual review.")
        df.drop(columns=['ref_median','_price_new','_price_fixed','_price_factor'], errors=True, inplace=True)
    # Q10: payment methods
    pay_col = next((c for c in df.columns if 'payment' in c or ('pay' in c and 'method' in c)), None)
    if not pay_col:
        pay_col = next((c for c in df.columns if any(k in c for k in ['payment_method','payment','paymethod','mode_of_payment','pay_type'])), None)
    if pay_col:
        print("[TRANSACTIONS][Q10] Standardizing payment method:", pay_col)
        def map_pay(x):
            if pd.isna(x): return np.nan
            s = str(x).lower(); s = re.sub(r'[^a-z0-9 ]',' ', s)
            if any(k in s for k in ['upi','gpay','google pay','phonepe','paytm','bhim','tez','googlepay']): return 'UPI'
            if any(k in s for k in ['cash on delivery','cod','c.o.d']): return 'COD'
            if any(k in s for k in ['credit card','creditcard','cc','visa','mastercard','amex']): return 'Credit Card'
            if any(k in s for k in ['debit card','debitcard']): return 'Debit Card'
            if any(k in s for k in ['netbank','net banking','netbanking','internet banking','bank transfer','neft','rtgs']): return 'NetBanking'
            if any(k in s for k in ['wallet','paytm wallet','mobikwik']): return 'Wallet'
            if 'upi' in s and 'wallet' in s: return 'UPI/Wallet'
            return s.title().strip()
        df[pay_col + '_clean'] = df[pay_col].apply(map_pay)
        save_df(df[[pay_col, pay_col + '_clean']].drop_duplicates().head(1000), 'Q10_payment_mapping_sample.csv')
        df[pay_col] = df[pay_col + '_clean']; df.drop(columns=[pay_col + '_clean'], inplace=True)
    # finalize transaction output
    out_path = OUT_CLEAN / 'transactions_cleaned.csv'
    if date_col and date_col in df.columns:
        try:
            df[date_col] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d')
        except:
            pass
    df.to_csv(out_path, index=False)
    print("[TRANSACTIONS] Cleaned saved to:", out_path)
    return df

# -----------------------
# PRODUCTS CLEANING (apply Q1..Q10 analogs where relevant)
# -----------------------
def clean_products():
    if not PRODUCTS_RAW.exists():
        print("[PRODUCTS] RAW file not found:", PRODUCTS_RAW)
        return None
    print("[PRODUCTS] Loading RAW:", PRODUCTS_RAW)
    p = pd.read_csv(PRODUCTS_RAW, low_memory=False, dtype=str)
    p.columns = [c.strip().lower().replace(' ','_') for c in p.columns]

    # P1: product_id normalization (upper, trim) and handle missing ids
    id_cols = [c for c in ('product_id','sku','asin','id') if c in p.columns]
    if id_cols:
        pid = id_cols[0]
        print("[PRODUCTS][P1] Normalizing product id column:", pid)
        p[pid] = p[pid].astype(str).fillna('').str.strip().str.upper().replace({'':'__MISSING__'})
        # if missing generated id, create surrogate
        missing_mask = p[pid] == '__MISSING__'
        if missing_mask.any():
            p.loc[missing_mask, pid] = ['MISSING_' + str(i) for i in range(1, missing_mask.sum()+1)]
        # canonical product_id column name
        p.rename(columns={pid:'product_id'}, inplace=True)
    else:
        # create surrogate id
        print("[PRODUCTS][P1] No id-like column found; creating surrogate product_id")
        p.insert(0, 'product_id', ['PID_' + str(i) for i in range(1, len(p)+1)])

    # P2: product name cleaning (trim, remove duplicates' whitespace)
    name_col = next((c for c in p.columns if 'name' in c), None)
    if name_col:
        print("[PRODUCTS][P2] Cleaning product name:", name_col)
        p[name_col] = p[name_col].astype(str).fillna('').str.strip().replace({'':'UNKNOWN_PRODUCT'})
        p.rename(columns={name_col:'product_name'}, inplace=True)
    else:
        p['product_name'] = p['product_id']

    # P3: price fields (mrp, price) -> numeric
    price_candidates = [c for c in ('mrp','price','list_price','selling_price','original_price') if c in p.columns]
    for pc in price_candidates:
        print(f"[PRODUCTS][P3] Parsing price column: {pc}")
        p[pc + '_clean'] = p[pc].apply(parse_price_string)
    # create canonical price column
    if price_candidates:
        first_pc = price_candidates[0] + '_clean'
        p['price'] = p[first_pc]
    else:
        p['price'] = np.nan

    # P4: brand normalization
    brand_col = next((c for c in p.columns if 'brand' in c), None)
    if brand_col:
        print("[PRODUCTS][P4] Normalizing brand column:", brand_col)
        def std_brand(x):
            if pd.isna(x): return np.nan
            s = str(x).strip()
            s = re.sub(r'[^A-Za-z0-9 &\-]', '', s)
            s = re.sub(r'\s+', ' ', s).strip()
            return s.title()
        p['brand'] = p[brand_col].apply(std_brand)
    else:
        p['brand'] = np.nan

    # P5: category normalization (map common synonyms)
    cat_col = next((c for c in p.columns if 'category' in c), None)
    if cat_col:
        print("[PRODUCTS][P5] Standardizing product category:", cat_col)
        mapping = {
            'electronics':'Electronics','electronic':'Electronics','electronics & accessories':'Electronics',
            'mobiles':'Mobiles & Accessories','mobile':'Mobiles & Accessories',
            'fashion':'Fashion','clothing':'Fashion',
            'home & kitchen':'Home & Kitchen','home & living':'Home & Kitchen',
            'books':'Books','beauty & personal care':'Beauty & Personal Care'
        }
        def std_cat(x):
            if pd.isna(x): return np.nan
            s = re.sub(r'[^a-z0-9\s&]', ' ', str(x).lower())
            s = re.sub(r'\s+', ' ', s).strip()
            for k,v in mapping.items():
                if k in s: return v
            return s.title()
        p['category'] = p[cat_col].apply(std_cat)
    else:
        p['category'] = np.nan

    # P6: missing images / urls — normalize nulls (if image columns exist)
    img_cols = [c for c in p.columns if 'image' in c or 'image_url' in c or 'img' in c]
    for ic in img_cols:
        p[ic] = p[ic].replace({'nan':None, '':None}).astype(object)

    # P7: duplicates in products — dedupe by product_id, keep most complete row
    print("[PRODUCTS][P7] Deduplicating products by product_id")
    p['__null_count'] = p.isna().sum(axis=1)
    p.sort_values(['product_id','__null_count'], ascending=[True, True], inplace=True)
    p = p.drop_duplicates(subset=['product_id'], keep='first')
    p.drop(columns=['__null_count'], inplace=True)

    # P8: handle outlier prices per product (if price extremely large)
    if 'price' in p.columns:
        print("[PRODUCTS][P8] Handling extreme product prices")
        p['price'] = pd.to_numeric(p['price'], errors='coerce')
        median_price = p['price'].median(skipna=True)
        # if product price > 100x median, flag for review
        p['__price_flag'] = (p['price'] > median_price * 100)
        flagged = p[p['__price_flag']]
        if not flagged.empty:
            save_df(flagged[['product_id','product_name','price']].head(500), 'P8_flagged_product_price_outliers.csv')
        p.drop(columns=['__price_flag'], inplace=True)

    # P9: clean product ratings if present
    prod_rating_cols = [c for c in p.columns if 'rating' in c]
    for rc in prod_rating_cols:
        p[rc + '_clean'] = p[rc].apply(parse_rating_string)
    if prod_rating_cols:
        save_df(p[[c + '_clean' for c in prod_rating_cols]].head(200), 'P9_product_ratings_sample.csv')
        # replace
        for rc in prod_rating_cols:
            p[rc] = p[rc + '_clean']; p.drop(columns=[rc + '_clean'], inplace=True)

    # P10: final tidy + outputs
    out_path = OUT_CLEAN / 'products_cleaned.csv'
    p.to_csv(out_path, index=False)
    print("[PRODUCTS] Cleaned products saved to:", out_path)

    # Save product category mapping sample for QA
    if 'category' in p.columns:
        save_df(p[['product_id','product_name','category']].drop_duplicates().head(1000), 'P10_product_category_sample.csv')

    return p

# Run both cleaners
# -----------------------
if __name__ == '__main__':
    tx = clean_transactions()
    pr = clean_products()

    # write brief cleaning summary
    summary = {
        'transactions_rows': len(tx) if tx is not None else 0,
        'products_rows': len(pr) if pr is not None else 0,
        'q1_dates_sample_saved': (OUT / 'Q1_invalid_dates_sample.csv').exists(),
        'q2_price_sample_saved': (OUT / 'Q2_unparsed_price_sample.csv').exists(),
        'q3_ratings_sample_saved': (OUT / 'Q3_ratings_sample.csv').exists(),
        'q4_unusual_cities_saved': (OUT / 'Q4_unusual_cities_sample.csv').exists(),
        'q8_dup_flag_saved': (OUT / 'Q8_flagged_possible_duplicates_sample.csv').exists(),
        'q9_auto_fixed_saved': (OUT / 'Q9_auto_fixed_samples.csv').exists(),
        'p8_product_price_flagged': (OUT / 'P8_flagged_product_price_outliers.csv').exists()
    }
    pd.Series(summary).to_frame('value').to_csv(OUT / 'cleaning_practice_full_summary.csv')
    print("[DONE] cleaning_practice_full_summary saved to outputs/")
