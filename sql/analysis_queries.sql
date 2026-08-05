-- analysis_queries.sql
-- A set of analytical SQL queries used to explore fraud/dispute patterns
-- in the disputed_transactions table. Written for SQLite.

-- 1. Overall fraud rate
SELECT
    COUNT(*)                                   AS total_disputes,
    SUM(is_fraud)                              AS fraud_count,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) AS fraud_rate_pct
FROM disputed_transactions;

-- 2. Fraud rate by merchant category (highest risk first)
SELECT
    merchant_category,
    COUNT(*)                                   AS total_disputes,
    SUM(is_fraud)                              AS fraud_count,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) AS fraud_rate_pct,
    ROUND(AVG(amount), 2)                      AS avg_amount
FROM disputed_transactions
GROUP BY merchant_category
ORDER BY fraud_rate_pct DESC;

-- 3. Fraud rate by channel
SELECT
    channel,
    COUNT(*)                                   AS total_disputes,
    SUM(is_fraud)                              AS fraud_count,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) AS fraud_rate_pct
FROM disputed_transactions
GROUP BY channel
ORDER BY fraud_rate_pct DESC;

-- 4. Monthly fraud trend
SELECT
    strftime('%Y-%m', transaction_datetime)    AS month,
    COUNT(*)                                   AS total_disputes,
    SUM(is_fraud)                              AS fraud_count,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) AS fraud_rate_pct
FROM disputed_transactions
GROUP BY month
ORDER BY month;

-- 5. New accounts (< 60 days) vs. established accounts: fraud comparison
SELECT
    CASE WHEN account_age_days < 60 THEN 'New Account (<60 days)'
         ELSE 'Established Account (60+ days)' END AS account_segment,
    COUNT(*)                                   AS total_disputes,
    SUM(is_fraud)                              AS fraud_count,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) AS fraud_rate_pct
FROM disputed_transactions
GROUP BY account_segment;

-- 6. Domestic vs. foreign transactions
SELECT
    CASE WHEN country = 'PH' THEN 'Domestic' ELSE 'Foreign' END AS geography,
    COUNT(*)                                   AS total_disputes,
    SUM(is_fraud)                              AS fraud_count,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) AS fraud_rate_pct
FROM disputed_transactions
GROUP BY geography;

-- 7. Repeat offenders: customers with 2+ historical disputes flagged as fraud
SELECT
    customer_id,
    COUNT(*)                                   AS total_disputes,
    SUM(is_fraud)                              AS fraud_count
FROM disputed_transactions
GROUP BY customer_id
HAVING COUNT(*) >= 2 AND SUM(is_fraud) >= 1
ORDER BY fraud_count DESC, total_disputes DESC
LIMIT 20;

-- 8. Transaction amount distribution: fraudulent vs. legitimate
SELECT
    dispute_status,
    COUNT(*)             AS n,
    ROUND(AVG(amount),2) AS avg_amount,
    ROUND(MIN(amount),2) AS min_amount,
    ROUND(MAX(amount),2) AS max_amount
FROM disputed_transactions
GROUP BY dispute_status;

-- 9. Late-night transactions (12AM-4AM) fraud rate vs. rest of day
SELECT
    CASE WHEN CAST(strftime('%H', transaction_datetime) AS INTEGER) BETWEEN 0 AND 4
         THEN 'Late Night (12AM-4AM)' ELSE 'Rest of Day' END AS time_segment,
    COUNT(*)                                   AS total_disputes,
    SUM(is_fraud)                              AS fraud_count,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) AS fraud_rate_pct
FROM disputed_transactions
GROUP BY time_segment;

-- 10. Highest-risk segment: new accounts + online channel + high-risk category
SELECT
    COUNT(*)                                   AS total_disputes,
    SUM(is_fraud)                              AS fraud_count,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) AS fraud_rate_pct
FROM disputed_transactions
WHERE account_age_days < 60
  AND channel = 'Online'
  AND merchant_category IN ('Electronics', 'Gift Cards', 'Crypto/Investment');
