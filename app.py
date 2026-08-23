import gradio as gr
import pandas as pd
import xgboost as xgb
import openai
import os
import json
import math
import re
import traceback

# Get OpenAI key from environment variables (configured in HF Space Secrets)
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Load the trained model
xgb_model = xgb.XGBClassifier()
xgb_model.load_model('xgb_model.json')

# Hardcoded features from your notebook training
existing_xgb_features = ['num_procedures', 'time_in_hospital', 'number_inpatient', 'number_outpatient', 'number_emergency', 'num_lab_procedures', 'number_diagnoses', 'num_medications', 'insulin_Up', 'insulin_Steady', 'insulin_No', 'insulin_Down', 'max_glu_serum_Norm', 'max_glu_serum_None', 'max_glu_serum_>300', 'max_glu_serum_>200', 'gender_Unknown/Invalid', 'A1Cresult_Norm', 'A1Cresult_None', 'A1Cresult_>8', 'A1Cresult_>7', 'age_[0-10)', 'age_[10-20)', 'age_[20-30)', 'age_[30-40)', 'age_[40-50)', 'age_[50-60)', 'age_[60-70)', 'age_[70-80)', 'age_[80-90)', 'age_[90-100)', 'gender_Female', 'gender_Male', 'race_Asian', 'race_Caucasian', 'race_AfricanAmerican', 'race_Hispanic', 'race_Other', 'discharge_disposition_id_1', 'discharge_disposition_id_2', 'discharge_disposition_id_3', 'discharge_disposition_id_4', 'discharge_disposition_id_5', 'discharge_disposition_id_6', 'discharge_disposition_id_7', 'discharge_disposition_id_8', 'discharge_disposition_id_9', 'discharge_disposition_id_10', 'discharge_disposition_id_13', 'discharge_disposition_id_14', 'discharge_disposition_id_16', 'discharge_disposition_id_17', 'discharge_disposition_id_18', 'discharge_disposition_id_22', 'discharge_disposition_id_23', 'discharge_disposition_id_24', 'discharge_disposition_id_25', 'discharge_disposition_id_27', 'discharge_disposition_id_28', 'change_No', 'change_Ch']

# Extraction function using LLM
def extract_features_from_notes(notes):
    prompt = f'''
    You are a medical data extractor. Extract the following features from the clinical notes below.
    Return ONLY a valid JSON object with the exact keys. If a feature is not mentioned, use sensible defaults (0 for boolean/counts).
    Keys to extract: {existing_xgb_features}

    Important Discharge Disposition Context:
    - ID 1: Discharged to home
    - ID 11: Expired (deceased)
    - ID 19: Expired at home
    - ID 20: Expired in a medical facility
    Ensure if the notes state the patient expired or is deceased, the corresponding feature (e.g., discharge_disposition_id_11) is set to 1.

    Clinical Notes:
    {notes}
    '''
    raw_response = ""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You return strictly JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.0
        )
        raw_response = response['choices'][0]['message']['content'].strip()

        extracted_json = re.sub(r"^```json\s*", "", raw_response)
        extracted_json = re.sub(r"^```\s*", "", extracted_json)
        extracted_json = re.sub(r"\s*```$", "", extracted_json)
        extracted_json = extracted_json.strip()

        features_dict = json.loads(extracted_json)

        df_input = pd.DataFrame([features_dict])
        for col in existing_xgb_features:
            if col not in df_input.columns:
                df_input[col] = 0

        df_input = df_input[existing_xgb_features]
        return df_input, "Success"
    except Exception as e:
        return None, f"Exception: {str(e)}\n\nRaw Response:\n{raw_response}"

# Prediction function
def predict_risk(df_input):
    try:
        def sanitize_colnames(df):
            cols = df.columns
            new_cols = []
            for col in cols:
                new_col = re.sub(r'[^a-zA-Z0-9_]', '_', col)
                new_cols.append(new_col)
            df.columns = new_cols
            return df

        df_input_sanitized = sanitize_colnames(df_input.copy())
        proba_weighted = xgb_model.predict_proba(df_input_sanitized)[0][1]
        log_odds_weighted = math.log(proba_weighted / (1 - proba_weighted)) if 0 < proba_weighted < 1 else 0
        prediction = "Readmission (<30 days)" if proba_weighted >= 0.5 else "No Readmission"

        result_text = f"Weighted XGBoost Prediction: {prediction}\n"
        result_text += f"Probability: {proba_weighted:.4f}\n"
        result_text += f"Log Odds: {log_odds_weighted:.4f}\n"
        return result_text, proba_weighted
    except Exception as e:
        return f"Prediction Error:\n{traceback.format_exc()}", None

# Recommendation function using LLM
def generate_recommendation(notes, risk_score, df_extracted=None):
    if risk_score is None:
        return "Cannot generate recommendation without a valid risk score."

    if df_extracted is not None:
        deceased_cols = ['discharge_disposition_id_11', 'discharge_disposition_id_19', 'discharge_disposition_id_20']
        for col in deceased_cols:
            if col in df_extracted.columns and df_extracted[col].iloc[0] == 1:
                return "Patient has expired (deceased). No follow-up or readmission prevention recommendations are applicable."

    question = f"The patient has a readmission risk score of {risk_score:.2f}. Based on their clinical notes: '{notes}', and the disparities guidelines in the PDF and diagnosis codes, what are the recommended follow-ups?"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant providing follow-up recommendations based on patient risk and guidelines."},
                {"role": "user", "content": question}
            ],
            temperature=0.2
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Recommendation Error:\n{traceback.format_exc()}"

# Gradio Interface wrapper
def process_patient_notes(notes):
    df_extracted, extraction_status = extract_features_from_notes(notes)
    if df_extracted is None:
        return {"error": "Extraction failed"}, f"Extraction Error Log:\n{extraction_status}", "Extraction failed."

    prediction_results, proba = predict_risk(df_extracted)
    if proba is None:
        return df_extracted.to_dict(orient='records')[0], prediction_results, "Prediction failed."

    recommendations = generate_recommendation(notes, proba, df_extracted)
    extracted_dict = df_extracted.to_dict(orient='records')[0]

    return extracted_dict, prediction_results, recommendations

# Build Gradio App
with gr.Blocks(title="Diabetic Patient Readmission Risk & Recommendation") as app:
    gr.Markdown("## 🏥 Diabetic Patient Readmission Risk Predictor")
    gr.Markdown("Enter patient notes below. The app will extract features, predict readmission risk (log odds), and provide recommendations.")

    with gr.Row():
        with gr.Column():
            notes_input = gr.Textbox(lines=10, label="Medical Professional Notes", placeholder="E.g., 65-year-old African American patient...")
            submit_btn = gr.Button("Process Notes & Predict")

        with gr.Column():
            extracted_output = gr.JSON(label="Extracted Features (JSON) / Error Dict")
            prediction_output = gr.Textbox(label="Prediction & Log Odds (Weighted XGBoost)", lines=10)
            recommendation_output = gr.Textbox(lines=8, label="Follow-up Recommendations")

    submit_btn.click(fn=process_patient_notes, inputs=notes_input, outputs=[extracted_output, prediction_output, recommendation_output])

if __name__ == "__main__":
    app.launch()
