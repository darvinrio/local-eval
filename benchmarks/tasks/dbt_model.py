"""
benchmarks/tasks/dbt_model.py

dbt model generation task.
"""

from .base import ContextTask

# ruff: disable[E501] line too long
DBT_MODEL_TASK = ContextTask(
    task_id="dbt_model",
    task_name="dbt Model Generator",
    description="Convert raw info → dbt model",
    system_prompt="You are a senior analytics engineer.",
    user_prompt_prefix="Here is raw business information, data descriptions, and source table schemas:\n\n",
    user_prompt_suffix="\n\nConvert the above into a dbt staging model SQL file with appropriate column renaming, type casting, and inline comments.",
    seed_content="""Entity: Customer
Description: A customer of the platform. Can be active or inactive.
Raw Table Schema: raw_customers
Columns:
- c_id: integer (Customer ID)
- c_name: varchar (Full name of customer)
- c_email: varchar (Email address)
- created_at: string (Date the customer joined in YYYY-MM-DD format)
- is_active: boolean (True if active, False if inactive)

Entity: Order
Description: An order placed by a customer.
Raw Table Schema: raw_orders
Columns:
- o_id: integer (Order ID)
- cust_id: integer (Customer ID, foreign key)
- order_date: string (Date of order, YYYY-MM-DD format)
- total_amount: float (Total amount of the order in USD)
- status: varchar (Status: pending, shipped, delivered, cancelled)
""",
    seed_content_lang="sql",
)
