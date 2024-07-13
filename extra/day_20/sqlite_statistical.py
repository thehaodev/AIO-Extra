import sqlite3
import pandas as pd


connection = sqlite3.connect('database.sqlite')
cursor = connection.cursor()

create__table_statement = """
    CREATE TABLE STOCK (
        ID INTEGER PRIMARY KEY,
        NAME TEXT NOT NULL,
        BUY INTEGER NOT NULL,
        INVESTOR TEXT NOT NULL
    );    
"""
cursor.execute(create__table_statement)

insert_statement = """
    INSERT INTO STOCK VALUES
        (1, 'ACB', 29.45, 'Nguyen'),
        (2, 'VIC', 44.55, 'Nguyen'),
        (3, 'GMD', 74.30, 'Nguyen'),
        (4, 'ACB', 28.45, 'Vinh'),
        (5, 'VIC', 40.55, 'Vinh'),
        (6, 'GMD', 60.30, 'Vinh')
"""
cursor.execute(insert_statement)

sum_query = """
    SELECT SUM(BUY) AS total_buy FROM STOCK;
"""

group_query = """
    SELECT INVESTOR, NAME, MAX(BUY) AS MAX_PRICE
    FROM STOCK
    GROUP BY INVESTOR;
"""


data = pd.read_sql_query("SELECT * FROM STOCK", connection)
data_sum = pd.read_sql_query(sum_query, connection)
data_max_by_investor = pd.read_sql_query(group_query, connection)

print('=== ALL Data ===')
print(data)
print(data_sum)
print(data_max_by_investor)
