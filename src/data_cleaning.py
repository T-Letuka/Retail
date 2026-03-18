
#%%
import numpy as np
import pandas as pd
import os 
import warnings

warnings.filterwarnings('ignore')

os.makedirs('../data/processed/' , exist_ok=True)

print('=' *55)
print('Retail Data Cleaning')
print('=' *55)
print('all imports successful.')
# %%
DATA_PATH = '../data/raw/'

file_map = {
    'payment': 'olist_order_payments_dataset.csv',
    'orders' : 'olist_orders_dataset.csv',
    'items' : 'olist_order_items_dataset.csv',
    'reviews' :  'olist_order_reviews_dataset.csv',
   'products' :  'olist_products_dataset.csv',
   'location' :  'olist_geolocation_dataset.csv',
   'customers' :  'olist_customers_dataset.csv',
'translation'  : 'product_category_name_translation.csv',
}

raw = {}

for name, filename in file_map.items():
    filepath = DATA_PATH + filename
    raw[name] = pd.read_csv(filepath)
    df = raw[name]
    print(f"{name:<15} {df.shape[0]:>7,} rows x {df.shape[1]:>2} cols")

print(f"total files loaded:{len(raw)}")
print('All files loaded')
#%% 
print('Section 2 -inspection ')

def inspect_tables(df,name):
    print(f"Table: {name.upper()}")
    print(f"Shape:{df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"{'='*55}")
    print(f"{'COLUMN':<40} {'DTYPE':<14} {'NULLS':>8} {'UNIQUE':>8}")
    print(f"{'-'*40} {'-'*14} {'-'*8} {'-'*8} ")
    for col in df.columns:
        dtype = str(df[col].dtype)
        nulls = df[col].isnull().sum()
        pct = nulls / len(df) * 100
        unique = df[col].nunique()
        null_str = f"{nulls:,} ({pct:.1f}%)"
        print(f"{col:<40} {dtype:<14} {null_str:>14} {unique:>8,}")
print(f"\n First 3 rows:")

for name, df in raw.items():
    inspect_tables(df, name)

# %%
print('Section 2 -inspection ')
print("="*60)
orders = raw['orders'].copy()

print(f"Starting shape {orders.shape}")

timestamp_cols = [
    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date',
]

for col in timestamp_cols:
    before_dtype = orders[col].dtype
    orders[col] = pd.to_datetime(orders[col], errors='coerce')
    after_dtype = orders[col].dtype
    null_count = orders[col].isnull().sum()
    print(f"{col}")
    print(f" dtype: {before_dtype} - {after_dtype} | {null_count:,}")

orders['order_status'] = orders['order_status'].str.lower().str.strip()

print(f"\n Final shape: {orders.shape}")
print(f"order_id unique values: {orders['order_id'].nunique():,}")
print(f"Order status breakdown:")
print(orders['order_status'].value_counts().to_string())

print('Order table Cleaned.yoh')



# %%
