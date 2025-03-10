import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import RFE
import numpy as np

# Function to load and prepare data (same as the existing one)
def btts(start, end):
    df_btts = pd.DataFrame()
    inter = pd.DataFrame()
    for i in range(start, end):
        url = f"https://fixturedownload.com/feed/json/epl-{i}"
        inter = pd.read_json(url)
        df_btts = pd.concat([df_btts, inter], ignore_index=True)
    df_btts['BTTS'] = (df_btts['HomeTeamScore'] > 0) & (df_btts['AwayTeamScore'] > 0)
    btts_summary = df_btts.groupby('RoundNumber')['BTTS'].agg(['sum', 'count'])
    btts_summary['BTTS_Rate'] = btts_summary['sum'] / btts_summary['count']
    return df_btts

# Function to create dataset (same as the existing one)
def datasetCreation(df):
    df['DateUtc'] = pd.to_datetime(df['DateUtc'], format='ISO8601')
    df['Day of Week'] = df['DateUtc'].dt.day_name()
    df['Time of Day'] = df['DateUtc'].dt.hour
    df['Time of Day'] = df['Time of Day'].apply(lambda x: 'Afternoon' if 12 <= x < 18 else ('Evening' if x >= 18 else 'Morning'))
    df[['Day of Week','Time of Day']] = df[['Day of Week','Time of Day']].astype('category')
    recent_teams = pd.unique(
        pd.concat([
            df.loc[df['DateUtc'].dt.year == df['DateUtc'].dt.year.max(), 'HomeTeam'],
            df.loc[df['DateUtc'].dt.year == df['DateUtc'].dt.year.max(), 'AwayTeam']
        ])
    )
    df = df[df['HomeTeam'].isin(recent_teams) & df['AwayTeam'].isin(recent_teams)]
    df_past = df[df['HomeTeamScore'].notna()]
    df_past['Both Teams Score'] = (df_past['HomeTeamScore'] > 0) & (df_past['AwayTeamScore'] > 0)
    return df_past

# Logistic Regression Model Fitting Function
def LogRegModelFitting(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)
    model = LogisticRegression(max_iter=1000, random_state=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1, y_test, y_pred, model

# Feature Selection and Model Fitting
def bestFeatures(Z):
    X = Z.drop(columns=['Both Teams Score_True', 'Both Teams Score_False', 'BTTS'])
    y = Z['Both Teams Score_True']
    accuracy, precision, recall, f1, y_test, y_pred, model = LogRegModelFitting(X, y)
    selector = RFE(model, n_features_to_select=10)
    selector.fit(X, y)
    selected_features = X.columns[selector.support_]
    return selected_features, model

# Prediction Function
def predictions(df_future, selected_features, model, df_results):
    X_new = df_future[selected_features]
    y_new_pred = model.predict(X_new)
    df_final = df_results
    df_final['BTTS Prediction'] = y_new_pred
    return df_final

# Streamlit User Interface
st.title("BTTS Prediction Dashboard")

col1, col2, col3 = st.columns(3)

# Visualization of BTTS distribution
st.sidebar.header("Visualizations")
if st.sidebar.checkbox("Show BTTS Distribution Plot"):
    df_btts = btts(2023, 2025)  # Or use uploaded data
    btts_summary = df_btts.groupby('RoundNumber')['BTTS'].agg(['sum', 'count'])
    btts_summary['BTTS_Rate'] = btts_summary['sum'] / btts_summary['count']
    plt.figure(figsize=(10, 5))
    sns.barplot(x=btts_summary.index, y=btts_summary['BTTS_Rate'], palette='viridis')
    plt.title('Proportion of Matches with Both Teams Scoring per Round')
    plt.xlabel('Round Number')
    plt.ylabel('BTTS Rate')
    plt.ylim(0, 1)
    with col1:
        st.pyplot(plt)

number = st.slider("Pick a number", 0, 100)
st.balloons()