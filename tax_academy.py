import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('TTA.csv',header=0,sep=';')
status = df['Progress'].unique().tolist()
#print(status)
st.logo("cropped-The-Tax-Academy-Logo.png",size="large")
st.set_page_config(layout="wide", page_title='The Tax Academy', page_icon=':moneybag:')
st.title('The Tax Academy')

st.expander('Welcome to the Tax Academy', expanded=True)

tab1, tab2 = st.tabs(['Prisons', 'Tax Returns'])

with tab1:
    st.write('Prisons')
with tab2:
    st.write(df)
    
# Group by "Progress" and count occurrences
progress_counts = df.groupby('Progress').size()
cmap = [(243/256,108/256,53/256,1),(255/256,223/256,186/256,1),(88/256,89/256,91/256,1)]
# Plot pie chart
#plt.figure(figsize=(8, 6))
plt.pie(progress_counts, labels=progress_counts.index, autopct='%1.1f%%',colors=cmap)
plt.title('Tax Return Status')
col1, col2, col3 = st.columns(3)
with col2:
    st.pyplot(plt)
with col3:
    st.write('Tax Returns Remaining', (df['Progress'] == 'Open').sum())
with col1:
    st.image('cropped-The-Tax-Academy-Logo.png')

