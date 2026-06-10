from django.shortcuts import render
import joblib
import numpy as np
import pandas as pd  # <-- Added Pandas to fix the warning
import os

# Import the database table
from .models import PredictionHistory

# 1. Locate and load your trained machine learning models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG_MODEL_PATH = os.path.join(BASE_DIR, 'ml_models', 'linear_model.pkl')
CLF_MODEL_PATH = os.path.join(BASE_DIR, 'ml_models', 'tree_model.pkl')

reg_model = joblib.load(REG_MODEL_PATH)
clf_model = joblib.load(CLF_MODEL_PATH)

def dashboard_view(request):
    context = {}
    
    # 2. If a user clicks the "Execute Engine" button...
    if request.method == 'POST':
        # Capture the numbers they typed into the form
        val = float(request.POST.get('initial_value', 0))
        logins = int(request.POST.get('logins', 0))
        inactive = int(request.POST.get('days_inactive', 0))
        
        # Structure the data exactly how our models expect it (as a DataFrame)
        input_data = pd.DataFrame(
            [[val, logins, inactive]], 
            columns=['initial_value', 'logins', 'days_inactive']
        )
        
        # 3. Generate live predictions!
        predicted_ltv = reg_model.predict(input_data)[0]
        predicted_churn = clf_model.predict(input_data)[0]
        
        # 4. Package the results
        context['prediction_made'] = True
        context['input_values'] = {'val': val, 'logins': logins, 'inactive': inactive}
        context['predicted_ltv'] = round(predicted_ltv, 2)
        
        if predicted_churn == 1:
            context['churn_risk'] = "🚨 HIGH RISK (Likely to Churn)" 
        else:
            context['churn_risk'] = "✅ LOW RISK (Active User)"

        # 5. Save the exact input and output to the database
        PredictionHistory.objects.create(
            initial_value=val,
            logins=logins,
            days_inactive=inactive,
            predicted_ltv=context['predicted_ltv'],
            churn_risk=context['churn_risk']
        )

    # Send everything to our HTML page
    return render(request, 'analytics/dashboard.html', context)