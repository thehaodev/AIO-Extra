import sqlite3
import pandas as pd

connection = sqlite3.connect('database.sqlite')
cursor = connection.cursor()

create__table_statement = """
    CREATE TABLE PRODUCT (
        ID INTEGER PRIMARY KEY,
        NAME TEXT NOT NULL,
        PRICE INTEGER NOT NULL
    );    
"""
cursor.execute(create__table_statement)

insert_statement = """
    INSERT INTO PRODUCT VALUES
        (1, 'iPhone 15', 18000000),
        (2, 'Galaxy Z-Fold 5', 30000000)
"""
cursor.execute(insert_statement)

update_statement = """
    UPDATE PRODUCT
    SET PRICE=50000000
    WHERE ID = 2;
"""
cursor.execute(update_statement)

delete_state_ment = """
    DELETE FROM PRODUCT
    WHERE NAME = 'iPhone 15'
"""
cursor.execute(delete_state_ment)


data = pd.read_sql_query("SELECT * FROM PRODUCT", connection)
print(data)
