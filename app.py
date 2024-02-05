# app.py

import pickle
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import RandomOverSampler

# Load the data
genes = pd.read_csv("sfari_genes.csv")

# Drop unnecessary columns
columns_to_drop = ['status', 'chromosome', 'number-of-reports', 'gene-name', 'ensembl-id', 'gene-score', 'genetic-category']
genes = genes.drop(columns=columns_to_drop)

# Encode gene symbols as dummy variables
genes_encoded = pd.get_dummies(genes, columns=['gene-symbol'])

# Features (X) excluding the 'syndromic' column
X = genes_encoded.drop(columns='syndromic')

# Labels (y)
y = genes_encoded['syndromic']

# Convert to binary classification (1 for syndromic, 0 for non-syndromic)
y_binary = (y == 1).astype(int)

# Resample the dataset
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y_binary)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialize the classifier
classifiers = {
    'XGBoost': XGBClassifier(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier()
}

# Train and evaluate each classifier on the resampled data
for clf_name, clf in classifiers.items():
    # Train the classifier
    clf.fit(X_train, y_train)

# Save each trained classifier to a pickle file
for clf_name, clf in classifiers.items():
    filename = f"{clf_name}_model.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(clf, file)
    print(f"{clf_name} model saved to {filename}")

# Streamlit App
st.title("Autism gene classification Web App")

# Sidebar for user input
gene_symbol = st.sidebar.text_input("Enter a gene symbol:")

if st.sidebar.button("Check Gene"):
    # Check if the gene symbol exists in the data
    if gene_symbol in genes['gene-symbol'].values:
        # Extract the corresponding row from the dataframe
        gene_info = genes[genes['gene-symbol'] == gene_symbol]

        # Make predictions using the loaded models
        xgb_prediction = classifiers['XGBoost'].predict(gene_info.drop(columns='syndromic'))
        svm_prediction = classifiers['SVM'].predict(gene_info.drop(columns='syndromic'))
        rf_prediction = classifiers['Random Forest'].predict(gene_info.drop(columns='syndromic'))

        # Display predictions
        st.write(f"XGBoost Prediction: {xgb_prediction[0]}")
        st.write(f"SVM Prediction: {svm_prediction[0]}")
        st.write(f"Random Forest Prediction: {rf_prediction[0]}")

    else:
        st.warning("The gene symbol does not exist in the data.")
