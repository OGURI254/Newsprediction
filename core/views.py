from django.shortcuts import render
from .forms import PredictionForm
import joblib
from django.conf import settings
import pandas as pd

import re
import string

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*','', text)
    return text

def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"



# Create your views here.
LR2 = joblib.load(settings.LR_PATH)
DT2=joblib.load(settings.DT_PATH)
VECTORIZATION=joblib.load(settings.VECTORIZATION_PATH)

def predict (request):
    form = PredictionForm()
    dt_prediction = None
    lr_prediction = None
    if request.method == 'POST':
        _form = PredictionForm(request.POST)
        if _form.is_valid():
            cd=_form.cleaned_data['news']
            testing_news = {"text": [cd]}
            new_def_test = pd.DataFrame(testing_news)
            new_def_test["text"] = new_def_test["text"].apply(wordopt)
            new_x_test = new_def_test["text"]
            new_xv_test = VECTORIZATION.transform(new_x_test)
 
            pred_LR = LR2.predict(new_xv_test)
            pred_DT = DT2.predict(new_xv_test)
            
            dt_prediction = output_lable(pred_DT[0])
            lr_prediction = output_lable(pred_LR[0])

            # print("\nLR Prediction: {} \nDT Prediction: {}".format(output_lable(pred_LR[0]), output_lable(pred_DT[0])))
           
    
    return render(request,'predict.html',{'form':form,'dt_prediction':dt_prediction,'lr_prediction':lr_prediction})

