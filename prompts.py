# prompts.py

# ==========================================
# 1. SCHEMA DEFINITIONS (MOCKED DDL)
# ==========================================

# NOTE: The instructions emphasize prioritizing V_ views over base tables.
SALES_SCHEMA = """
-- Sales & Revenue Domain
CREATE TABLE customers (customer_id integer PRIMARY KEY, first_name text NOT NULL, last_name text, email text UNIQUE, phone text, address_line1 text, city text, state text, zip text, country text DEFAULT 'USA', customer_type text DEFAULT 'retail', created_at date);
CREATE TABLE employees (employee_id integer PRIMARY KEY, first_name text NOT NULL, last_name text NOT NULL, role text, department text, email text, hire_date date);
CREATE TABLE orders (order_id integer PRIMARY KEY, customer_id integer, employee_id integer, order_date date NOT NULL, required_date date, shipped_date date, status text NOT NULL DEFAULT 'pending', shipping_cost numeric DEFAULT 0.00, discount_pct numeric DEFAULT 0.00, notes text, warehouse_id integer);
CREATE TABLE order_items (order_item_id integer PRIMARY KEY, order_id integer NOT NULL, product_id integer NOT NULL, quantity integer NOT NULL, unit_price numeric NOT NULL, discount_pct numeric DEFAULT 0.00);
"""

INVENTORY_SCHEMA = """
-- Inventory Domain
CREATE TABLE categories (category_id integer PRIMARY KEY, name text NOT NULL, description text);
CREATE TABLE products (product_id integer PRIMARY KEY, sku text NOT NULL UNIQUE, name text NOT NULL, category_id integer NOT NULL, supplier_id integer NOT NULL, cost_price numeric NOT NULL, retail_price numeric NOT NULL, weight_kg numeric, dimensions text, material text, is_active integer DEFAULT 1);
CREATE TABLE warehouses (warehouse_id integer PRIMARY KEY, name text NOT NULL, city text, country text, capacity_m2 integer);
CREATE TABLE stock (stock_id integer PRIMARY KEY, product_id integer NOT NULL, warehouse_id integer NOT NULL, qty_on_hand integer NOT NULL DEFAULT 0, qty_reserved integer NOT NULL DEFAULT 0, reorder_point integer NOT NULL DEFAULT 10, reorder_qty integer NOT NULL DEFAULT 50, last_updated date);
"""

PROCUREMENT_SCHEMA = """
-- Procurement Domain
CREATE TABLE suppliers (supplier_id integer PRIMARY KEY, name text NOT NULL, contact_name text, phone text, email text, country text, lead_time_days integer);
CREATE TABLE purchase_orders (po_id integer PRIMARY KEY, supplier_id integer NOT NULL, employee_id integer, order_date date NOT NULL, expected_date date, received_date date, status text NOT NULL DEFAULT 'open', warehouse_id integer, notes text);
CREATE TABLE purchase_order_items (po_item_id integer PRIMARY KEY, po_id integer NOT NULL, product_id integer NOT NULL, quantity integer NOT NULL, unit_cost numeric NOT NULL);
"""

SCHEMA_MAPPING = {
    "sales": SALES_SCHEMA,
    "inventory": INVENTORY_SCHEMA,
    "procurement": PROCUREMENT_SCHEMA
}

ALL_SCHEMAS = f"{SALES_SCHEMA}\n{INVENTORY_SCHEMA}\n{PROCUREMENT_SCHEMA}"

# ==========================================
# 2. SYSTEM PROMPTS
# ==========================================

SQL_GENERATION_SYSTEM_PROMPT = """You are an expert PostgreSQL developer for a Business Intelligence system.
Your job is to generate highly optimized, valid PostgreSQL queries based on the provided schema and user query.

CRITICAL RULES:
1. You MUST FORBID queries against monetary columns (e.g., price, cost, unit_price, cost_price) if the user role is "Warehouse Admin". If they ask for this, generate a safe query that returns a single column `error_message` indicating "Akses ditolak: Admin Gudang tidak dapat melihat data finansial."
2. Do not include markdown formatting like ```sql in your final output.
3. Provide the exact response format defined below or the system will crash.

RESPONSE FORMAT (Valid JSON object):
{
  "sql": "SELECT ..."
}
"""

EXPLAINER_SYSTEM_PROMPT = """You are an expert Business Analyst data explainer and communicator.
You will be provided with a user's original query, the executed SQL query, and the JSON results from the database.

Your task is to:
1. Write a natural, professional business explanation of the data in INDONESIAN.
2. Provide a clean, well-formatted Markdown table representing the data beneath your explanation.
3. Your answer should be direct and insightful. Don't mention the internal database structure directly.
4. Provide the response adhering strictly to the required JSON structure.

RESPONSE FORMAT (Valid JSON object):
{
  "text": "<Your Indonesian business summary here>",
  "table": "<Markdown table of the raw data array>"
}
"""
