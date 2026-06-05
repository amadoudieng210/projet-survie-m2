from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import json

app = Flask(__name__)

# 1. Chargement et préparation des données
def init_model():
    chemin_fichier = "ProjetM2SID2026.xlsx"
    df = pd.read_excel(chemin_fichier, sheet_name="Donnees")
    
    # Nettoyage et types
    df['DUREE SUIVI Apres Traitement (mois)'] = pd.to_numeric(df['DUREE SUIVI Apres Traitement (mois)'], errors='coerce')
    df['DECES_NUM'] = df['DECES'].map({'OUI': 1, 'NON': 0})
    
    bool_cols = ['DIABETE', 'Metastases Hepatiques', 'Dénutrition']
    for col in bool_cols:
        if col in df.columns:
            df[col + '_NUM'] = df[col].map({'OUI': 1, 'NON': 0})
            
    if 'SEXE' in df.columns:
        df['SEXE_NUM'] = df['SEXE'].map({'M': 1, 'F': 0})
        
    if 'Traitement par chirurgie' in df.columns:
        df['Traitement par chirurgie_NUM'] = df['Traitement par chirurgie'].map({'OUI': 1, 'NON': 0})
        
    # Caractéristiques stables du modèle de Cox
    features = ['AGE', 'SEXE_NUM', 'hémoglobine', "Durée d'evolution des Symptom en Mois", 
                'DIABETE_NUM', 'Metastases Hepatiques_NUM', 'Dénutrition_NUM', 'Traitement par chirurgie_NUM']
                
    df_cox = df[features + ["DUREE SUIVI Apres Traitement (mois)", "DECES_NUM"]].dropna()
    
    cph = CoxPHFitter()
    cph.fit(df_cox, duration_col="DUREE SUIVI Apres Traitement (mois)", event_col="DECES_NUM")
    return cph

# Initialisation globale du modèle de Cox
model = init_model()

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_data = None
    score_risque = None
    
    if request.method == 'POST':
        # Récupération des données du formulaire HTML
        age = float(request.form['age'])
        sexe = int(request.form['sexe'])
        hemo = float(request.form['hemo'])
        sympt = float(request.form['sympt'])
        diabete = int(request.form['diabete'])
        metastase = int(request.form['metastase'])
        denutrition = int(request.form['denutrition'])
        chirurgie = int(request.form['chirurgie'])
        
        # Création du profil patient
        profil = pd.DataFrame([{
            'AGE': age, 'SEXE_NUM': sexe, 'hémoglobine': hemo,
            "Durée d'evolution des Symptom en Mois": sympt, 'DIABETE_NUM': diabete,
            'Metastases Hepatiques_NUM': metastase, 'Dénutrition_NUM': denutrition,
            'Traitement par chirurgie_NUM': chirurgie
        }])
        
        # Prédiction de la courbe de survie
        surv_prob = model.predict_survival_function(profil)
        
        # Structurer les données pour Chart.js (JavaScript)
        prediction_data = {
            "labels": list(surv_prob.index.astype(int)),
            "values": list(np.round(surv_prob.values.flatten(), 2))
        }
        score_risque = round(float(model.predict_partial_hazard(profil).values[0]), 4)

    return render_template('index.html', prediction_data=json.dumps(prediction_data), score_risque=score_risque)

if __name__ == '__main__':
    app.run(debug=True)