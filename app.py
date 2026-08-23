import streamlit as st
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


def calculate_financial_balance(intervention_cost, missed_readmission_cost):
    """Calculate costs and break-even values from the notebook hold-out results."""
    tn, fp, fn, tp = 15942, 2133, 1585, 680
    interventions = fp + tp
    actual_readmissions = fn + tp
    baseline_cost = actual_readmissions * missed_readmission_cost
    total_intervention_cost = interventions * intervention_cost
    total_missed_cost = fn * missed_readmission_cost
    total_model_cost = total_intervention_cost + total_missed_cost
    savings = baseline_cost - total_model_cost
    break_even_intervention = (tp * missed_readmission_cost) / interventions
    break_even_readmission = (interventions * intervention_cost) / tp
    roi = savings / total_intervention_cost if total_intervention_cost else float("nan")

    return {
        "Hold-out patients": tn + fp + fn + tp,
        "Actual readmissions": actual_readmissions,
        "True positives": tp,
        "False positives": fp,
        "False negatives": fn,
        "True negatives": tn,
        "Interventions": interventions,
        "Total intervention cost": total_intervention_cost,
        "Total missed readmission cost": total_missed_cost,
        "Total model cost": total_model_cost,
        "No-intervention baseline": baseline_cost,
        "Savings vs no intervention": savings,
        "ROI": roi,
        "Break-even intervention cost per flagged patient": break_even_intervention,
        "Break-even missed-readmission cost": break_even_readmission,
    }

# Build Streamlit App

st.title("🏥 Diabetic Patient Readmission Risk & Recommendation")

st.markdown(
    """
    Enter patient notes below. The app will extract features,
    predict readmission risk, and provide recommendations.
    """
)

notes_input = st.text_area(
    "Medical Professional Notes",
    height=250,
    placeholder="E.g., 65-year-old African American patient..."
)

if st.button("Process Notes & Predict"):

    extracted, prediction, recommendations = process_patient_notes(
        notes_input
    )

    st.subheader("Recommendations")
    st.write(recommendations)

    st.subheader("Prediction")
    st.text(prediction)

    st.subheader("Extracted Features")
    st.json(extracted)

st.divider()
st.header("Hold-out Set Financial Analysis")
st.caption("Uses the fine-tuned XGBoost confusion matrix from the notebook's 20% hold-out set.")

with st.form("financial_form"):
    intervention_cost = st.number_input(
        "Intervention cost per flagged patient",
        min_value=0.0,
        value=3000.0,
        step=100.0,
    )
    missed_readmission_cost = st.number_input(
        "Cost per missed readmission",
        min_value=0.0,
        value=15000.0,
        step=500.0,
    )
    calculate_financials = st.form_submit_button("Calculate Financials")

if calculate_financials:
    financials = calculate_financial_balance(intervention_cost, missed_readmission_cost)
    st.subheader("Financial Results")
    st.dataframe(
        pd.DataFrame(
            {"Metric": list(financials.keys()), "Value": list(financials.values())}
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.write(
        f"At these assumptions, the maximum break-even intervention cost is "
        f"**${financials['Break-even intervention cost per flagged patient']:,.2f}** "
        f"per flagged patient."
    )
