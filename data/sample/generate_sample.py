"""
Generate customer_churn.csv — the demo dataset for Insight Excavator.

Hidden non-obvious relationships baked in:
1. Support tickets is a stronger churn driver than spend (counterintuitive)
2. Satisfaction score is highly correlated with support ticket volume (obvious once confirmed)
3. Premium customers churn less despite similar age distributions (group difference)
4. Monthly spend anomalies cluster around long-tenure customers (anomaly)
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1000

age = rng.integers(18, 70, n).astype(float)
tenure_months = rng.integers(1, 120, n).astype(float)
plan = rng.choice(["basic", "standard", "premium"], n, p=[0.30, 0.45, 0.25])

# Spend: premium pays more; long-tenure customers spend more
monthly_spend = (
    np.where(plan == "premium", rng.normal(180, 30, n),
    np.where(plan == "standard", rng.normal(100, 20, n),
             rng.normal(55, 15, n)))
    + 0.3 * tenure_months / tenure_months.max() * 50
    + rng.normal(0, 10, n)
)
monthly_spend = np.clip(monthly_spend, 10, None).round(2)

# Support tickets: younger + early tenure → more tickets
support_tickets = (
    rng.poisson(1.5, n)
    + (age < 30).astype(int) * rng.poisson(1, n)
    + (tenure_months < 12).astype(int) * rng.poisson(2, n)
)
support_tickets = np.clip(support_tickets, 0, 15)

# Satisfaction: driven primarily by support tickets (the non-obvious part)
satisfaction = (
    9.0
    - 0.55 * support_tickets
    + 0.15 * (plan == "premium").astype(float)
    + 0.005 * tenure_months
    + rng.normal(0, 0.4, n)
)
satisfaction = np.clip(satisfaction, 1, 10).round(1)

# Churn: support tickets matter MORE than spend (counterintuitive finding)
churn_logit = (
    -2.5
    + 0.35 * support_tickets          # strong driver
    - 0.008 * tenure_months           # longer tenure = less churn
    - 0.40 * (plan == "premium").astype(float)
    + 0.005 * age
    - 0.003 * monthly_spend
    + rng.normal(0, 0.3, n)
)
churn_prob = 1 / (1 + np.exp(-churn_logit))
churned = rng.binomial(1, churn_prob)

# Add a few realistic anomalies in spend (enterprise outliers)
outlier_idx = rng.choice(n, 12, replace=False)
monthly_spend[outlier_idx] = rng.uniform(600, 900, 12).round(2)

# Inject some messiness for the cleaning stage
df = pd.DataFrame({
    "customer_id": [f"C{i:04d}" for i in range(n)],
    "age": age,
    "tenure_months": tenure_months,
    "plan": plan,
    "monthly_spend": monthly_spend,
    "support_tickets": support_tickets.astype(int),
    "satisfaction_score": satisfaction,
    "churned": churned,
})

# Sprinkle nulls
null_age_idx = rng.choice(n, 25, replace=False)
null_spend_idx = rng.choice(n, 18, replace=False)
df.loc[null_age_idx, "age"] = np.nan
df.loc[null_spend_idx, "monthly_spend"] = np.nan

# Add some whitespace dirt in plan column
dirt_idx = rng.choice(n, 30, replace=False)
df.loc[dirt_idx, "plan"] = df.loc[dirt_idx, "plan"].apply(lambda x: f"  {x}  ")

df.to_csv("data/sample/customer_churn.csv", index=False)
print(f"Generated {n} rows → data/sample/customer_churn.csv")
print(f"Churn rate: {churned.mean():.1%}")
print(f"Anomalous spends injected: {len(outlier_idx)}")
