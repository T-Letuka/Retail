
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
   'sellers': 'olist_sellers_dataset.csv'
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
print('Section 3 -Cleaning orders table')
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
print('Section 04 --Cleaning the Customers table')
print('='*50)

customers = raw['customers'].copy()

print(f'Starting shape: {customers.shape}')
print(f"unique_id for customers: {customers['customer_id'].nunique():,}")
print(f"customers_unique_id unique: {customers['customer_unique_id'].nunique():,}")

repeat_customers = customers['customer_id'].nunique() - customers['customer_unique_id'].nunique()
print(f"Repeated customers (ordered 2+ times): {repeat_customers:,}")



customers['customer_state'] = (
    customers['customer_state'].str.upper()
    .str.strip()
)

customers['customer_city'] = (
    customers['customer_city'].str.upper()
    .str.lower()
)

print(f"Top 10 states by customer count:")
print(customers['customer_state'].value_counts().head(10).to_string())
print(f"Final Shape: {customers.shape}")
print('Customers table cleaned')


# %%

print('Section 05 CLEANING ORDER TIMES TABLE')
print('='*60)

items = raw['items'].copy()

print(f"starting shape : { items.shape}")
print(f"order_id shape {items['order_id'].nunique():,}")
print(f"(vs {raw['orders']['order_id'].nunique():,} unique orders)")
print(f"Some orders have multiple items , this is expected")

items['shipping_limit_date']  = pd.to_datetime(
    items['shipping_limit_date'], errors='coerce'

)

print(f"Price stats")
print(f" min: {items['price'].min():>10.2f}")
print(f" max: {items['price'].max():>10.2f}")
print(f"mean: {items['price'].mean():>10.2f}")
print(f"Negative prices: {(items['price'] < 0).sum()}")
print(f"Zero prices: {(items['price'] == 0).sum()}")

extreme_prices = items['price'].quantile(0.999)
n_extreme = (items['price'] > extreme_prices).sum()
print(f"Prices above 99.9th percentile ({extreme_prices:.2f}) : {n_extreme}")

print(f"Final shape:{items.shape}")
print('Items table cleaned.')
# %%

print('Section 06 CLEANING PAYMENT TABLE')
print('='*60)

payments = raw['payment'].copy()

print(f"Starting shape: {payments.shape}")
print(f" order_id unique: {payments['order_id'].nunique():,}")
print(f"Rows per order_id (before aggregation:)")
rows_per_order = payments.groupby('order_id').size()
print(f"Orders with 2+ payments rows need to be aggregated")


multi_payment_orders = (rows_per_order > 1).sum()
print(f"Orders with multiple payments: {multi_payment_orders}")

payment_agg = (
    payments
    .groupby('order_id')
    .agg(
        payment_value= ('payment_value', 'sum'), 
        payment_type= ('payment_type', lambda x: x.mode()[0]),
        payment_installments = ('payment_installments', 'mean'),
        payment_count = ('payment_value', 'count'),
    ).reset_index()
)

print(f"   After aggregation shape: { payment_agg.shape}")
print(f"    Unique order_id updated : {payment_agg['order_id'].nunique():,}")
print(f"    Payment typer breakdown:")
print(payment_agg['payment_type'].value_counts().to_string())



# %%

print('Section 07 CLEANING REVIEWS TABLE')
print('='*60)

reviews = raw['reviews'].copy()

print(f" Starting shape for reviews : {reviews.shape}")
print(reviews.columns)


date_cols = ['review_creation_date' , 'review_answer_timestamp']
for col in date_cols:
    if col in reviews.columns:
        reviews[col] = pd.to_datetime(reviews[col], errors='coerce')

n_duplicates = reviews.duplicated(subset='order_id').sum()
print(f" Duplicated order_ids : {n_duplicates:,}")

if n_duplicates > 0:
    reviews = (
        reviews
        .sort_values('review_creation_date', ascending=True)
        .drop_duplicates(subset='order_id', keep='last')
    ) 
    print(f" After deduplication : {reviews.shape[0]:,} rows")

score_nulls = reviews['review_score'].isnull().sum()
comment_nulls = reviews['review_comment_message'].isnull().sum()

print(f"  Review score nulls are {score_nulls:,}({score_nulls/len(reviews)*100:.1f}%)")
print(f"  Comment Nulls  are {comment_nulls:,}({comment_nulls/len(reviews)*100:.1f}%)")
print('We are expecting a high rate of comment nulls, this is normal. I dont leave comment too')

print(f'Review score distrubition:')
print(reviews['review_score'].value_counts().sort_index().to_string())

reviews_slim = reviews[['order_id', 'review_score', 'review_comment_message']].copy()

print(f"   Keeping {reviews_slim.shape[1]} columns from the original reviews table")
print('Reviews table done')

# %%

print('Section 08 CLEANING PRODUCTS TABLE')
print('='*60)

products = raw['products'].copy()
translation = raw['translation'].copy()

print(f" Starting shape for products {products.shape}")
print(f" Starting shape for translation {translation.shape}")


cat_nulls = products['product_category_name'].isnull().sum()
print(f"  Missing category name {cat_nulls:,} ({cat_nulls/len(products)*100:.1f}%)")


products_en = products.merge(
    translation,
    on='product_category_name',
    how='left'
)

products_en['product_category_name_english'] = (
    products_en['product_category_name_english']
    .fillna('uncategorized')
)

untranslated = (products_en['product_category_name_english'] == 'uncategorized').sum()
print(f"Uncategorized after translation : {untranslated:,}")

products_slim =  products_en[['product_id', 'product_category_name_english']].copy()

print(f" Top 10 products ")
print(products_slim['product_category_name_english'].value_counts().head(10).to_string())
print(f" Final slim shape: {products_slim.shape}")
print(f" Products tables cleaned ")


# %%
print('Section 09 CLEANING SELLERS TABLE')
print('='*60)

sellers = raw['sellers'].copy()

print (f"  Starting shape {sellers.shape}")
print(sellers.columns)

sellers['seller_state'] = (
    sellers['seller_state']
    .str.upper()
    .str.strip()
)

sellers_slim = sellers[['seller_id', 'seller_state']].copy()
print(f" Top 10 sellers states")
print(sellers_slim['seller_state'].value_counts().head(10).to_string())
print(f"Final slim shape {sellers_slim.shape}")
print('Sellers table cleaned')