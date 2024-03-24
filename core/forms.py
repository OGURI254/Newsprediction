from django import forms
class PredictionForm(forms.Form):
    news=forms.CharField(max_length=1000)
    