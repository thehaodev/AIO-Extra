import os
import re

import pandas as pd
import numpy as np

dtype = {
    'customer_id': str,
    'gender_cd': str,
    'postal_cd': str,
    'application_store_cd': str,
    'status_cd': str,
    'category_major_cd': str,
    'category_medium_cd': str,
    'category_small_cd': str,
    'product_cd': str,
    'store_cd': str,
    'prefecture_cd': str,
    'tel_no': str,
    'street': str
}

df_customer = pd.read_csv("../100_pandas/data/customer.csv", dtype=dtype)
df_category = pd.read_csv("../100_pandas/data/category.csv", dtype=dtype)
df_product = pd.read_csv("../100_pandas/data/product.csv", dtype=dtype)
df_receipt = pd.read_csv("../100_pandas/data/receipt.csv", dtype=dtype)
df_store = pd.read_csv("../100_pandas/data/store.csv", dtype=dtype)
df_geocode = pd.read_csv("../100_pandas/data/geocode.csv", dtype=dtype)


def p_001():
    print(df_receipt.head(10))


def p_002():
    data = df_receipt[['sales_ymd', 'customer_id', 'product_cd', 'amount']]
    print(data.head(10))


def p_003():
    data = df_receipt[['sales_ymd', 'customer_id', 'product_cd', 'amount']]
    data.rename(columns={"sales_ymd": "sales_date"}, inplace=True)
    print(data.head(10))


def p_004():
    data = df_receipt[['sales_ymd', 'customer_id', 'product_cd', 'amount']]
    row_customer_id = data[data["customer_id"] == 'CS018205000001']
    print(row_customer_id)


def p_005():
    data = df_receipt[['sales_ymd', 'customer_id', 'product_cd', 'amount']]
    row_customer_id = data[(data["customer_id"] == 'CS018205000001') & (data['amount'] >= 1000)]
    print(row_customer_id)


def p_006():
    data = df_receipt[['sales_ymd', 'customer_id', 'product_cd', 'quantity', 'amount']]
    row_customer_id = data[(data["customer_id"] == 'CS018205000001')
                           & ((data['amount'] >= 1000) | (data['quantity'] >= 5))]
    print(row_customer_id)


def p_007():
    data = df_receipt[['sales_ymd', 'customer_id', 'product_cd', 'quantity', 'amount']]
    row_customer_id = data[(data["customer_id"] == 'CS018205000001')
                           & ((data['amount'] >= 1000) & (data['amount'] <= 2000))]
    print(row_customer_id)


def p_008():
    data = df_receipt[['sales_ymd', 'customer_id', 'product_cd', 'amount']]
    row_customer_id = data[(data["customer_id"] == 'CS018205000001')
                           & (data["product_cd"] != 'P071401019')]
    print(row_customer_id)


def p_009():
    query = df_store.query('(prefecture_cd != "13") & (floor_area<900)')
    print(query)


def p_0010():
    data = df_store.loc[df_store['store_cd'].str.contains('^S14', flags=re.I, regex=True)]
    print(data.head(10))


def p_0011():
    data = df_customer.loc[df_customer['customer_id'].str.contains('01$', flags=re.I, regex=True)]
    print(data.head(10))


def p_0012():
    data = df_store.loc[df_store['address'].str.contains('Yokohama City[a-z]*', flags=re.I, regex=True)]
    print(data.head(10))


def p_0013():
    data = df_customer.loc[df_customer['status_cd'].str.contains('^[A-F]', flags=re.I, regex=True)]
    print(data.head(10))


def p_0014():
    data = df_customer.loc[df_customer['status_cd'].str.contains('[1-9]$', flags=re.I, regex=True)]
    print(data.head(10))


def p_0015():
    data = df_customer.loc[df_customer['status_cd'].str.contains('^[A-F].*[1-9]$', flags=re.I, regex=True)]
    print(data.head(10))


def p_0016():
    data = df_store.loc[df_store['tel_no'].str.contains(r'^\d{3}.*\d{4}$', flags=re.I, regex=True)]
    print(data.head(10))


def p_0017():
    data = df_customer.sort_values('birth_day', ascending=False)
    print(data.head(10))


def p_0018():
    data = df_customer.sort_values('birth_day', ascending=True)
    print(data.head(10))


def p_0019():
    df_receipt['rank'] = df_receipt['amount'].rank(method='min', ascending=False)
    top_ranked = df_receipt.sort_values('rank', ascending=False)
    result = top_ranked[['customer_id', 'amount', 'rank']]
    print(result.head(10))


def p_0020():
    df_receipt['rank'] = df_receipt['amount'].rank(method='dense', ascending=False)
    top_ranked = df_receipt.sort_values('rank', ascending=False)
    result = top_ranked[['customer_id', 'amount', 'rank']]
    print(result.head(10))


p_001()
p_002()
p_003()
p_004()
p_005()
p_006()
p_007()
p_008()
p_009()
p_0010()
p_0011()
p_0012()
p_0013()
p_0014()
p_0015()
p_0016()
p_0017()
p_0018()
p_0019()
