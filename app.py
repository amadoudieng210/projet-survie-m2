from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import json

app = Flask(__name__)

def init_model():
    chemin_fichier = "ProjetM2SID2026.xlsx"
    df = pd.read_excel(chemin_fichier, sheet_name="Donnees")
    
    # 1. Nettoyage des espaces dans les noms de colonnes
    df.columns = df.columns.str.strip()
    
    time_col = "DUREE SUIVI Apres Traitement (mois)"
    event_col = "DECES_NUM"
    
    # 2. Conversion et traitement de la variable de temps
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
    # Si le temps est manquant, on met la médiane du suivi
    df[time_col] = df[time_col].fillna(df[time_col].median())
    
    # 3. Encodage de l'événement cible
    df[event_col] = df['DECES'].map({'OUI': 1, 'NON': 0}).fillna(0).astype(int)
    
    # 4. Encodage robuste des variables explicatives
    def map_column(search_term, new_name, default_val=0):
        actual_col = [c for c in df.columns if search_term.lower() in c.lower()]
        if actual_col:
            # On mappe les chaînes textuelles
            df[new_name] = df[actual_col[0]].map({'OUI': 1, 'NON': 0, 'M': 1, 'F': 0})
            # Si la colonne d'origine contenait déjà des chiffres (0 ou 1), on les préserve
            df[new_name] = df[new_name].fillna(df[actual_col[0]])
            # S'il reste des NaN, on met la valeur par défaut
            df[new_name] = df[new_name].fillna(default_val).astype(int)
        else:
            # Si la colonne n'existe pas du tout, on la crée remplie de 0 pour éviter le plantage
            df[new_name] = default_val

    map_column('DIABETE', 'DIABETE_NUM', default_val=0)
    map_column('Metastases Hepatiques', 'Metastases Hepatiques_NUM', default_val=0)
    map_column('Dénutrition', 'Dénutrition_NUM', default_val=0)
    map_column('SEXE', 'SEXE_NUM', default_val=1)
    map_column('chirurgie', 'Traitement par chirurgie_NUM', default_val=1)
    
    # 5. Traitement des variables numériques directes (AGE et Hémoglobine)
    df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce')
    df['AGE'] = df['AGE'].fillna(df['AGE'].mean())
    
    # Recherche souple pour l'hémoglobine
    col_hemo = [c for c in df.columns if 'hémoglobine' in c.lower() or 'hemo' in c.lower()]
    if col_hemo:
        df['hémoglobine'] = pd.to_numeric(df[col_hemo[0]], errors='coerce')
    df['hémoglobine'] = df['hémoglobine'].fillna(df['hémoglobine'].mean())
    
    # Recherche souple pour la durée des symptômes
    col_sympt = [c for c in df.columns if 'sympt' in c.lower() or 'evolution' in c.lower()]
    if col_sympt:
        df["Durée d'evolution des Symptom en Mois"] = pd.to_numeric(df[col_sympt[0]], errors='coerce')
    df["Durée d'evolution des Symptom en Mois"] = df["Durée d'evolution des Symptom en Mois"].fillna(df["Durée d'evolution des Symptom en Mois"].median())

    # Liste des variables prédictives finales
    features = [
        'AGE', 'SEXE_NUM', 'hémoglobine', "Durée d'evolution des Symptom en Mois", 
        'DIABETE_NUM', 'Metastases Hepatiques_NUM', 'Dénutrition_NUM', 'Traitement par chirurgie_NUM'
    ]
    
    # Extraction finale du dataset (sans aucun NaN restant possible)
    df_cox = df[features + [time_col, event_col]].copy()
    
    cph = CoxPHFitter()
    cph.fit(df_cox, duration_col=time_col, event_col=event_col)
    return cph

# Entraînement initial sécurisé du modèle de Cox
model = init_model()

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_data = None
    score_risque = None
    
    if request.method == 'POST':
        # Récupération sécurisée du formulaire
        age = float(request.form.get('age', 65))
        sexe = int(request.form.get('sexe', 1))
        hemo = float(request.form.get('hemo', 11.0))
        sympt = float(request.form.get('sympt', 6))
        diabete = int(request.form.get('diabete', 0))
        metastase = int(request.form.get('metastase', 0))
        denutrition = int(request.form.get('denutrition', 0))
        chirurgie = int(request.form.get('chirurgie', 1))
        
        # Profil du patient simulé
        profil = pd.DataFrame([{
            'AGE': age, 'SEXE_NUM': sexe, 'hémoglobine': hemo,
            "Durée d'evolution des Symptom en Mois": sympt, 'DIABETE_NUM': diabete,
            'Metastases Hepatiques_NUM': metastase, 'Dénutrition_NUM': denutrition,
            'Traitement par chirurgie_NUM': chirurgie
        }])
        
        # Prédiction de la survie
        surv_prob = model.predict_survival_function(profil)
        
        prediction_data = {
            "labels": list(surv_prob.index.astype(int)),
            "values": list(np.round(surv_prob.values.flatten(), 2))
        }
        score_risque = round(float(model.predict_partial_hazard(profil).values[0]), 4)

    return render_template('index.html', prediction_data=json.dumps(prediction_data), score_risque=score_risque)

if __name__ == '__main__':
    app.run(debug=True)
