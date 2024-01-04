#!/usr/bin/env python
# coding: utf-8

# In[6]:


from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pred_utils import (interpret_lime_results_v5, cf_explanations, get_data, 
                        visualize_counterfactuals_radar_plotly, 
                        visualize_counterfactuals_plotly, explain_counterfactual_percentage, 
                        generate_transaction_features, main_instance, cfe_instance)
import pandas as pd
import joblib
import time
import traceback

app = FastAPI()

# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to match your deployment environment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data
X_train_ad, y_train_ad, X_train_smote, y_train_smote, X_train_stomek, y_train_stomek = get_data()

class PredictionRequest(BaseModel):
    selected_model: str  # Renamed from model_selection
    step: int
    type: int
    amount: float
    oldbalanceOrg: float

@app.post("/api/predict")
async def predict(request: PredictionRequest):
    try:
        data = request.dict()

        # Validate and load model selection
        selected_model = data.get('selected_model', 'Random Forest')

        if selected_model == 'Random Forest':
            model_name = 'Random Forest'
            pipeline = joblib.load('rfc_adasyn_pipeline.pkl')
            X_train = X_train_ad
            y_train = y_train_ad
        elif selected_model == 'Gradient Boost':
            model_name = 'Gradient Boost'
            pipeline = joblib.load('gbc_smote_pipeline.pkl')
            X_train = X_train_smote
            y_train = y_train_smote
        elif selected_model == 'Neural Network':
            model_name = 'Neural Network'
            pipeline = joblib.load('mlp_smotetomek_pipeline.pkl')
            X_train = X_train_stomek
            y_train = y_train_stomek
        else:
            raise HTTPException(status_code=400, detail="Invalid model selection")

        # Prepare input data
        user_data = {
            'step': int(data.get('step', 0)),
            'type': int(data.get('type', 0)),
            'amount': float(data.get('amount', 0)),
            'oldbalanceOrg': float(data.get('oldbalanceOrg', 0)),
            'newbalanceOrig': float(data.get('oldbalanceOrg', 0)) + float(data.get('amount', 0))
        }

        df = pd.DataFrame([user_data])
        live_df = generate_transaction_features(df)

        # Make prediction
        pred_outcome = pipeline.predict(live_df)
        prediction = pred_outcome[0]

        # Interpret the prediction
        prediction_text = "Fraudulent" if prediction == 1 else "Not Fraudulent"

        # Explainability methods
        cf_as_dict, original_instance = cf_explanations(pipeline, X_train, y_train, live_df, model_name, total_CFs=1)
        lime_explanation_html = interpret_lime_results_v5(pipeline, X_train, live_df, model_name)
        radial_plot_html = visualize_counterfactuals_radar_plotly(cf_as_dict, original_instance)
        bar_chart_html = visualize_counterfactuals_plotly(original_instance, cf_as_dict)
        narrative_html = explain_counterfactual_percentage(original_instance, cf_as_dict)

        # Return JSON response
        return JSONResponse(content={
            'prediction_text': prediction_text,
            'lime_explanation_html': lime_explanation_html,
            'radar_plot_html': radial_plot_html,
            'bar_chart_html': bar_chart_html,
            'narrative_html': narrative_html
        })

    except Exception as e:
        print(traceback.format_exc())  
        raise HTTPException(status_code=500, detail=str(e))
        

