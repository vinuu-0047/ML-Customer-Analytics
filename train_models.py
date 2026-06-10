
# PHASE 1: IMPORTS & SETUP
import os                  # Lets Python interact with your computer's operating system (to make folders)
import numpy as np         # The math engine; used to generate our random, synthetic data
import pandas as pd        # The data engine; lets us work with data in a tabular format (like Excel)
from sklearn.linear_model import LinearRegression      # Algorithm to predict a continuous number (Revenue)
from sklearn.tree import DecisionTreeClassifier        # Algorithm to predict a category (Churn: Yes/No)
import joblib              # A tool used to save (serialize) our trained models into actual files

# Check if a folder named 'ml_models' exists. If not, create it so we have a place to save our work.
os.makedirs('ml_models', exist_ok=True)

print("⚡ Generating synthetic behavioral data...")

# Setting a "seed" ensures that even though we generate random numbers, 
# we get the exact same "random" numbers every time we run the script.
np.random.seed(42) 
n_samples = 1000 # We are creating a fake database of 1,000 customers

# ==========================================
# PHASE 2: DATA GENERATION (FEATURES)
# These are the inputs our model will learn from.
# ==========================================

# 1. Initial purchase amount: Random numbers between $10 and $500
initial_value = np.random.uniform(10, 500, n_samples)

# 2. Monthly logins: Random whole numbers between 1 and 30 times a month
logins = np.random.randint(1, 30, n_samples)

# 3. Inactivity: How many days since they last opened the app (0 to 90 days)
days_inactive = np.random.randint(0, 90, n_samples)


# ==========================================
# PHASE 3: DATA GENERATION (TARGETS)
# These are the "answers" we want our models to learn to predict.
# ==========================================

# TARGET A: Lifetime Value (LTV) - How much money will they spend over time?
# We invent a mathematical rule: Higher initial value and more logins = higher LTV. 
# High inactivity hurts LTV. We also add a little random "noise" so it's not perfectly predictable.
ltv = (initial_value * 1.5) + (logins * 12) - (days_inactive * 2) + np.random.normal(0, 25, n_samples)
ltv = np.clip(ltv, 10, None) # Ensures nobody has an LTV less than $10

# TARGET B: Churn - Will they leave the platform? (1 = Yes, 0 = No)
# We invent a rule: The higher the days inactive and the lower the logins, the higher the chance of leaving.
# We use a mathematical function (sigmoid) to turn these factors into a probability between 0% and 100%.
churn_prob = 1 / (1 + np.exp(-(-3 + (days_inactive * 0.1) - (logins * 0.15))))
churn = (churn_prob > 0.5).astype(int) # If probability is over 50%, mark them as "1" (Churned).


# ==========================================
# PHASE 4: PREPARING THE DATA TABLE
# ==========================================

# We combine all our separate arrays into a single Pandas DataFrame (a structured table)
df = pd.DataFrame({
    'initial_value': initial_value,
    'logins': logins,
    'days_inactive': days_inactive,
    'ltv': ltv,
    'churn': churn
})

# X represents our inputs (Features)
X = df[['initial_value', 'logins', 'days_inactive']]

# y represents our outputs (Targets)
y_ltv = df['ltv']
y_churn = df['churn']


# ==========================================
# PHASE 5: TRAINING THE MACHINE LEARNING MODELS
# ==========================================
print("🤖 Training LTV and Churn models...")

# Model 1: Linear Regression (Predicting money)
# We show the model the inputs (X) and the answers (y_ltv). It figures out the math to connect them.
reg_model = LinearRegression()
reg_model.fit(X, y_ltv)

# Model 2: Decision Tree Classifier (Predicting Yes/No)
# It learns rules (e.g., "If days_inactive > 45 AND logins < 5, then Churn = 1").
# max_depth=4 stops the tree from getting too complex, and 'gini' is the mathematical method it uses to split data.
clf_model = DecisionTreeClassifier(max_depth=4, criterion='gini', random_state=42)
clf_model.fit(X, y_churn)


# ==========================================
# PHASE 6: SAVING THE MODELS
# ==========================================

# We take the "brains" that just learned the patterns and save them as files.
# Later, our Django web server will load these exact files to make live predictions!
joblib.dump(reg_model, 'ml_models/linear_model.pkl')
joblib.dump(clf_model, 'ml_models/tree_model.pkl')

print("💾 Success! Models saved to ml_models/ directory.")