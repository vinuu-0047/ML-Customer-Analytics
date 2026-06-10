from django.db import models

class PredictionHistory(models.Model):
    # What the user typed in
    initial_value = models.FloatField()
    logins = models.IntegerField()
    days_inactive = models.IntegerField()
    
    # What the AI predicted
    predicted_ltv = models.FloatField()
    churn_risk = models.CharField(max_length=50)
    
    # Exactly when this happened
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction: LTV ${self.predicted_ltv} | {self.churn_risk}"