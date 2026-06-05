from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import json

app = Flask(__name__)

def init_model():
    chemin_fichier = "ProjetM2SID2026.xlsx"
    df = pd.read_excel(chemin_fichier, sheet_name="Donnees")
    
    # NETTOYAGE DES NOMS DE COLONNES : On enlève les espaces au début et à la fin
    df.columns = df.columns.str.strip()
    
    # 1. Forcer la colonne de durée en type numérique
    time_col = "DUREE SUIVI Apres Traitement (mois)"
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
    
    # 2. Encodage de la variable cible
    df['DECES_NUM'] = df['DECES'].map({'OUI': 1, 'NON': 0})
    
    # 3. Encodage dynamique des variables pour éviter les KeyErrors
    # On cherche les colonnes même s'il y a des petites variations de texte
    def map_column(search_term, new_name):
        actual_col = [c for c in df.columns if search_term.lower() in c.lower()]
        if actual_col:
            df[new_name] = df[actual_col[0]].map({'OUI': 1, 'NON': 0, 'M': 1, 'F': 0})
            return True
        return False

    map_column('DIABETE', 'DIABETE_NUM')
    map_column('Metastases Hepatiques', 'Metastases Hepatiques_NUM')
    map_column('Dénutrition', 'Dénutrition_NUM')
    map_column('SEXE', 'SEXE_NUM')
    map_column('chirurgie', 'Traitement par chirurgie_NUM')
    
    # Variables explicatives stables utilisées pour le modèle de Cox
    features = [
        'AGE', 'SEXE_NUM', 'hémoglobine', "Durée d'evolution des Symptom en Mois", 
        'DIABETE_NUM', 'Metastases Hepatiques_NUM', 'Dénutrition_NUM', 'Traitement par chirurgie_NUM'
    ]
    
    # Suppression des valeurs manquantes sur les lignes du modèle
    df_cox = df[features + [time_col, 'DECES_NUM']].dropna()
    
    cph = CoxPHFitter()
    cph.fit(df_cox, duration_col=time_col, event_col='DECES_NUM')
    return cph

# Initialisation du modèle de Cox
model = init_model()

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_data = None
    score_risque = None
    
    if request.method == 'POST':
        # Récupération des données du formulaire
        age = float(request.form['age'])
        sexe = int(request.form['sexe'])
        hemo = float(request.form['hemo'])
        sympt = float(request.form['sympt'])
        diabete = int(request.form['diabete'])
        metastase = int(request.form['metastase'])
        denutrition = int(request.form['denutrition'])
        chirurgie = int(request.form['chirurgie'])
        
        # Création du profil du patient simulé
        profil = pd.DataFrame([{
            'AGE': age, 'SEXE_NUM': sexe, 'hémoglobine': hemo,
            "Durée d'evolution des Symptom en Mois": sympt, 'DIABETE_NUM': diabete,
            'Metastases Hepatiques_NUM': metastase, 'Dénutrition_NUM': denutrition,
            'Traitement par chirurgie_NUM': chirurgie
        }])
        
        # Prédiction de la courbe de survie
        surv_prob = model.predict_survival_function(profil)
        
        prediction_data = {
            "labels": list(surv_prob.index.astype(int)),
            "values": list(np.round(surv_prob.values.flatten(), 2))
        }
        score_risque = round(float(model.predict_partial_hazard(profil).values[0]), 4)

    return render_template('index.html', prediction_data=json.dumps(prediction_data), score_risque=score_risque)

if __name__ == '__main__':
    app.run(debug=True)
