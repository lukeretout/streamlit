import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import RFE
import numpy as np

def btts(start, end):
    df_btts = pd.DataFrame()
    inter = pd.DataFrame()
    for i in range(start, end):
        url = f"https://fixturedownload.com/feed/json/epl-{i}"
        inter = pd.read_json(url)
        df_btts = pd.concat([df_btts, inter], ignore_index = True)
    # Add a column to indicate if both teams scored
    df_btts['BTTS'] = (df_btts['HomeTeamScore'] > 0) & (df_btts['AwayTeamScore'] > 0)


    # Calculate BTTS summary per round
    btts_summary = df_btts.groupby('RoundNumber')['BTTS'].agg(['sum', 'count'])
    btts_summary['BTTS_Rate'] = btts_summary['sum'] / btts_summary['count']


    # Count BTTS per round
    btts_summary = df_btts.groupby('RoundNumber')['BTTS'].sum().reset_index()

    # Add a column for the BTTS ratio as a string (e.g., '7:3' if 7 games had BTTS)
    btts_summary['BTTS Ratio'] = btts_summary['BTTS'].astype(str) + ':' + (10 - btts_summary['BTTS']).astype(str)

    # Count how often each BTTS ratio occurs
    ratio_counts = btts_summary['BTTS Ratio'].value_counts().reset_index()
    ratio_counts.columns = ['BTTS Ratio', 'Count']

    # Plot the distribution of BTTS ratios
    #plt.figure(figsize=(10, 6))
    #sns.barplot(x='BTTS Ratio', y='Count', data=ratio_counts, palette='viridis')
    #plt.title('Distribution of BTTS Outcomes per Round')
    #plt.xlabel('BTTS : No BTTS')
    #plt.ylabel('Number of Rounds')
    #plt.xticks(rotation=45)
    #plt.show()
    return df_btts

def datasetCreation(df):

    # Formatting the date to datetime
    df['DateUtc'] = pd.to_datetime(df['DateUtc'], format='ISO8601')

    # Extract the day of the week (0 = Monday, 6 = Sunday)
    df['Day of Week'] = df['DateUtc'].dt.day_name()

    # Determine if it is an afternoon or evening game
    df['Time of Day'] = df['DateUtc'].dt.hour
    df['Time of Day'] = df['Time of Day'].apply(lambda x: 'Afternoon' if 12 <= x < 18 else ('Evening' if x >= 18 else 'Morning'))

    #Set HomeTeam and AwayTeam for Nottingham Forest
    df['HomeTeam'] = df['HomeTeam'].replace({'Nott\'m Forest': 'Nottingham Forest'}).astype('category')
    df['AwayTeam'] = df['AwayTeam'].replace({'Nott\'m Forest': 'Nottingham Forest'}).astype('category')
    # Setting Home Team, Away Time, Day of Week and Time of Day to be categories. This will save memory and can be used later for one-hot encoding
    df[['Day of Week','Time of Day']] = df[['Day of Week','Time of Day']].astype('category')


        
    # Find the most recent season
    most_recent_year = df['DateUtc'].dt.year.max()

    # Find teams from the most recent season
    recent_teams = pd.unique(
        pd.concat([
            df.loc[df['DateUtc'].dt.year == most_recent_year, 'HomeTeam'],
            df.loc[df['DateUtc'].dt.year == most_recent_year, 'AwayTeam']
        ])
    )

    # Filter out matches involving teams not in the most recent season
    df = df[
        df['HomeTeam'].isin(recent_teams) & df['AwayTeam'].isin(recent_teams)
    ]

    # Removing all future matches
    df_past = df[df['HomeTeamScore'].notna() & (df['HomeTeamScore'] != '')]
    df_future = df[(df['HomeTeamScore'].isna())]

    # Setting both of the scores to be ints
    df_past['HomeTeamScore'] = df_past['HomeTeamScore'].astype(int)
    df_past['AwayTeamScore'] = df_past['AwayTeamScore'].astype(int)
    

    # Creating a new column for whether both teams have scored or not and setting it as a category
    df_past['Both Teams Score'] = (df_past['HomeTeamScore'] > 0) & (df_past['AwayTeamScore'] > 0)
    df_past['Both Teams Score'] = df_past['Both Teams Score'].astype('category')

    grouping = df_past.groupby('RoundNumber')['Both Teams Score'].value_counts().unstack()
    # Adding an empty column to df_future to ensure that it will be the same shape later on for predictions.
    df_future['Both Teams Score'] = None
    df_results = df_future

    scoreMatch = df_past[['RoundNumber','DateUtc','HomeTeamScore','AwayTeamScore']]

    
    df_past = df_past.drop(columns=['MatchNumber','RoundNumber','DateUtc','Location','AwayTeamScore','HomeTeamScore','Group'])
    df_past = pd.get_dummies(df_past)

    df_future = df_future.drop(columns=['MatchNumber','Location', 'RoundNumber', 'DateUtc','Group','AwayTeamScore','HomeTeamScore'])
    df_future = pd.get_dummies(df_future)
    
    return df_past, df_future, df_results, grouping, scoreMatch

def countplot(data):
    # Simple count plot for visualization
    sns.countplot(data=data, x='Both Teams Score_True')
    plt.title('Count plot of BTTS')
    plt.xlabel('Both Teams Score')
    print(data['Both Teams Score_True'].value_counts(normalize=True))

def LogRegModelFitting(X, y):
    

    # Try without scaling
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.20, random_state=10)

    # Initiate a base model
    model = LogisticRegression(max_iter=1000, random_state=10)

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Use the fitted model to make a prediction
    y_pred = model.predict(X_test)
    
    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return accuracy, precision, recall, f1, y_test, y_pred, model

def bestFeatures(Z):
    
    X = Z.drop(columns=['Both Teams Score_True','Both Teams Score_False','BTTS'])
    y = Z['Both Teams Score_True']
    
    accuracy, precision, recall, f1, y_test, y_pred, model = LogRegModelFitting(X,y)
   
    # Feature Importance
    features = pd.Series(model.coef_[0],index=X.columns)
    sorted_features=features.sort_values()
    #plt.figure(figsize=(20, 20))
    #sorted_features.plot(kind='barh')
    #plt.title('Feature Importance in the Logistic Regression Model')
    #plt.show()
    #df_past.columns[:-2]
    results = []
    for i in range(1,50):
        model = LogisticRegression(max_iter=1000, random_state=10)
        selector = RFE(model, n_features_to_select=i)
        selector.fit(X, y)
        selected_features = X.columns[selector.support_]

        accuracy, precision, recall, f1, y_test, y_pred, model = LogRegModelFitting(X[selected_features],y)
        results.append({'Iteration':i,'Accuracy':accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1} )

    results_df = pd.DataFrame(results)
    
    #plt.figure(figsize=(10, 5))
    #sns.lineplot(x='Iteration', y='Accuracy', data=results_df, marker='o', color='b',label='Accuracy')
    #sns.lineplot(x='Iteration', y='Precision', data=results_df, marker='o', color='g',label='Precision')
    #sns.lineplot(x='Iteration', y='Recall', data=results_df, marker='o', color='r',label='Recall')
    #sns.lineplot(x='Iteration', y='F1', data=results_df, marker='x', color='b',label='F1')
    #plt.title("Accuracy, Precision, Recall & F1 Over Iterations")
    #plt.xlabel("Iteration")
    #plt.ylabel("Value")
    #plt.show()

    # Calculate row sums and find the index of the maximum sum
    optimal_index_sum = results_df.sum(axis=1).idxmax()

    #print("\nOptimal Index (Max Sum):", optimal_index_sum)
    #print(results_df.loc[optimal_index_sum])

    # Calculate the min value per row and find the index of the maximum min
    optimal_index_min = results_df.min(axis=1).idxmax()

    #print("\nOptimal Index (Highest Minimum):", optimal_index_min)
    #print(results_df.loc[optimal_index_min])

    model = LogisticRegression(max_iter=1000, random_state=10)
    selector = RFE(model, n_features_to_select=optimal_index_min+1)
    selector.fit(X, y)
    selected_features = X.columns[selector.support_]

    accuracy, precision, recall, f1, y_test, y_pred, model = LogRegModelFitting(X[selected_features],y)

    # Print the results
    #print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    #print("Classification report:\n", classification_report(y_test, y_pred))
    return results_df, model, selected_features

def predictions(df_future, selected_features, model, df_results):
    
    # Assume X_new is your new data (same features as X)
    X_new = df_future[selected_features]

    y_new_pred = model.predict(X_new)
    #print(y_new_pred.shape)
    
    y_new_probs = model.predict_proba(X_new)
    #print(y_new_probs.shape)
    
    #df_final = df_results.reset_index(drop=True)
    df_final = df_results
    #print(df_final.shape)
    df_final['BTTS Prediction'] = y_new_pred

    # Reset index if necessary
    df_final = df_final.reset_index(drop=True)

    # Convert to DataFrame and concatenate
    y_new_prob_df = pd.DataFrame(y_new_probs, columns=['% False', '% True'])
    y_new_prob_df = y_new_prob_df.reset_index(drop=True)

    # Now concatenate
    df_final1 = pd.concat([df_final, y_new_prob_df], axis=1)
    df_final1 = df_final1.drop(columns=['MatchNumber','Group','HomeTeamScore','AwayTeamScore','BTTS','Both Teams Score'])

    return df_final1
# Streamlit User Interface
st.title("BTTS Prediction Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    start = st.selectbox("Start Year", list(range(2020, 2026)))
with col2:
    end = st.selectbox("End Year", list(range(start+1, 2026)))

#df_btts = btts(start, end)

#df_past, df_future,df_results, grouping, scoreMatch = datasetCreation(df_btts)


#results_df, model, selected_features = bestFeatures(df_past)
#df_final1 = predictions(df_future, selected_features, model, df_results)

#st.write(df_final1.head())
st.slider("Select the number of rows to display", 1, 10, 5)
