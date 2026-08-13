"""Create a small retail database and answer business questions with SQL."""
import sqlite3

connection = sqlite3.connect(":memory:")
cursor = connection.cursor()

cursor.executescript("""
CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, city TEXT);
CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, order_date TEXT, amount REAL,
  FOREIGN KEY(customer_id) REFERENCES customers(customer_id));
INSERT INTO customers VALUES
 (1, 'Asha', 'Hyderabad'), (2, 'Ravi', 'Bengaluru'), (3, 'Meera', 'Hyderabad');
INSERT INTO orders VALUES
 (101, 1, '2025-01-10', 2200), (102, 2, '2025-01-15', 1850),
 (103, 1, '2025-02-04', 3100), (104, 3, '2025-02-18', 2750),
 (105, 2, '2025-03-09', 1200);
""")

queries = {
    "Revenue by city": """
        SELECT c.city, ROUND(SUM(o.amount), 2) AS revenue
        FROM orders o JOIN customers c ON c.customer_id = o.customer_id
        GROUP BY c.city ORDER BY revenue DESC
    """,
    "Top customers": """
        SELECT c.name, ROUND(SUM(o.amount), 2) AS lifetime_value
        FROM orders o JOIN customers c ON c.customer_id = o.customer_id
        GROUP BY c.name ORDER BY lifetime_value DESC
    """,
    "Monthly revenue": """
        SELECT substr(order_date, 1, 7) AS month, ROUND(SUM(amount), 2) AS revenue
        FROM orders GROUP BY month ORDER BY month
    """,
}

for title, query in queries.items():
    print(f"\n{title}")
    for row in cursor.execute(query):
        print(row)

connection.close()
