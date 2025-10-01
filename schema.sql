-- ==========================
-- DATABASE SCHEMA FOR ANALYTICS
-- ==========================

-- CUSTOMERS DIMENSION
CREATE TABLE IF NOT EXISTS customers (
  customer_id VARCHAR(50) PRIMARY KEY,
  customer_city VARCHAR(128),
  customer_state VARCHAR(128),
  customer_tier VARCHAR(64),
  customer_spending_tier VARCHAR(64),
  customer_age_group VARCHAR(64),
  is_prime_member TINYINT(1),
  lifetime_value DECIMAL(14,2),
  total_orders INT,
  avg_order_value DECIMAL(12,2)
) ENGINE=InnoDB;

-- PRODUCTS DIMENSION
CREATE TABLE IF NOT EXISTS products (
  product_id VARCHAR(50) PRIMARY KEY,
  product_name VARCHAR(255),
  category VARCHAR(128),
  subcategory VARCHAR(128),
  brand VARCHAR(128),
  base_price_2015 DECIMAL(12,2),
  weight_kg DECIMAL(8,3),
  rating DECIMAL(3,1),
  is_prime_eligible TINYINT(1),
  launch_year YEAR,
  model VARCHAR(128),
  price DECIMAL(12,2)
) ENGINE=InnoDB;

-- TIME DIMENSION
CREATE TABLE IF NOT EXISTS time_dimension (
  date_id DATE PRIMARY KEY,
  year SMALLINT,
  quarter TINYINT,
  month TINYINT,
  month_name VARCHAR(20),
  week_of_year TINYINT,
  day_of_month TINYINT,
  day_name VARCHAR(20),
  is_weekend TINYINT(1)
) ENGINE=InnoDB;

-- TRANSACTIONS FACT TABLE
CREATE TABLE IF NOT EXISTS transactions (
  transaction_id VARCHAR(50) PRIMARY KEY,
  order_date DATETIME NOT NULL,
  customer_id VARCHAR(50) NOT NULL,
  product_id VARCHAR(50) NOT NULL,
  product_name VARCHAR(255),
  category VARCHAR(128),
  subcategory VARCHAR(128),
  brand VARCHAR(128),
  original_price_inr DECIMAL(12,2),
  discount_percent DECIMAL(5,2),
  discounted_price_inr DECIMAL(12,2),
  quantity INT,
  subtotal_inr DECIMAL(12,2),
  delivery_charges DECIMAL(10,2),
  final_amount_inr DECIMAL(12,2),
  customer_city VARCHAR(128),
  customer_state VARCHAR(128),
  customer_tier VARCHAR(64),
  customer_spending_tier VARCHAR(64),
  customer_age_group VARCHAR(64),
  payment_method VARCHAR(64),
  delivery_days TINYINT,
  delivery_type VARCHAR(64),
  is_prime_member TINYINT(1),
  is_festival_sale TINYINT(1),
  festival_name VARCHAR(128),
  customer_rating DECIMAL(3,1),
  return_status VARCHAR(32),
  order_month TINYINT,
  order_year SMALLINT,
  order_quarter TINYINT,
  product_weight_kg DECIMAL(8,3),
  is_prime_eligible TINYINT(1),
  product_rating DECIMAL(3,1),
  final_amount DECIMAL(12,2),
  original_price_inr_clean DECIMAL(12,2),

  -- FKs
  FOREIGN KEY (product_id) REFERENCES products(product_id),
  FOREIGN KEY (customer_id) REFERENCES customers(customer_id),

  -- Indexes
  INDEX idx_order_date(order_date),
  INDEX idx_customer(customer_id),
  INDEX idx_product(product_id),
  INDEX idx_state(customer_state),
  INDEX idx_category(category),
  INDEX idx_payment(payment_method)
) ENGINE=InnoDB;
