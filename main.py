# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:00:22 2023

@author: Vijay B.
"""

import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_extras.colored_header import colored_header
import datetime
import pickle


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style.css")



with st.sidebar:
    st.title("Sensor Data Analysis ")
    
    st.markdown('''Write Project Description''')
    
def get_sample_numbers(df):
    timestamp_test = []
    len_test =[]
    for x in list(df):
        timestamp_test.append(x[0])
        len_test.append(len(x[1]))
    return (list(range(1,len(len_test)+1)),len_test)


tab1, tab2 = st.tabs(["Original dataset - Analysis", "ML based Error prediction"])



def run_models(test):
    #Generating the Predicted reference for Acceleration_x error
    accel_x_model_path =  'acceleration_x.pkl'
    with open(accel_x_model_path , 'rb') as f:
        accel_x_model = pickle.load(f)
    accel_x_pred_df=test[['Accel_Y','Accel_X','Gyro_Z']]
    accel_x_pred_df.rename(columns={'Gyro_Z':'feature_Gyro_Z','Accel_X':'feature_Accel_X','Accel_Y':'feature_Accel_Y'},inplace=True)
    pred_erro=accel_x_model.predict(accel_x_pred_df[['feature_Accel_Y','feature_Accel_X','feature_Gyro_Z']])
    accel_x_pred_df['pred_ref_accel_x'] = accel_x_pred_df['feature_Accel_X'].values-pred_erro ## Ref = Test - error 

    #Generating the Predicted reference for Acceleration_y error
    accel_x_model_path =  'reg_accel_y.pkl'
    with open(accel_x_model_path , 'rb') as f:
        accel_y_model = pickle.load(f)
    
    final_df = test.copy()
    final_df.rename(columns={'Accel_Y':'feature_Accel_Y','Accel_X':'feature_Accel_X'},inplace=True)
    accel_y_pred_df=final_df[['feature_Accel_Y','feature_Accel_X']]
    pred_erro=accel_y_model.predict(accel_y_pred_df[['feature_Accel_Y','feature_Accel_X']])
    accel_y_pred_df['pred_ref_accel_y'] = accel_y_pred_df['feature_Accel_Y'].values-pred_erro.reshape(1,-1)[0]

    #Generating the Predicted reference for Acceleration_z error
    accel_x_model_path =  'reg_accel_z.pkl'
    with open(accel_x_model_path , 'rb') as f:
        accel_z_model = pickle.load(f)
        
    final_df = test.copy()
    final_df.rename(columns={'Accel_Y':'feature_Accel_Y','Gyro_Z':'feature_Gyro_Z'},inplace=True)
    accel_z_pred_df=final_df[['Accel_Z','feature_Accel_Y','feature_Gyro_Z']]
    pred_erro=accel_z_model.predict(accel_z_pred_df[['feature_Accel_Y','feature_Gyro_Z']])
    accel_z_pred_df['pred_ref_accel_z'] = final_df['Accel_Z'].values-pred_erro.reshape(1,-1)[0]

    #Generating the Predicted reference for gyro_x_error

    accel_x_model_path =  'reg_gyro_x_pred.pkl'
    with open(accel_x_model_path , 'rb') as f:
        gyro_x = pickle.load(f)
        
    final_df = test.copy()
    final_df.rename(columns={'Gyro_X':'feature_Gyro_X'},inplace=True)
    gyro_x_pred_df=final_df[['feature_Gyro_X']]
    pred_erro=gyro_x.predict(gyro_x_pred_df[['feature_Gyro_X']])
    gyro_x_pred_df['pred_ref_gyro_x'] = gyro_x_pred_df['feature_Gyro_X'].values-pred_erro.reshape(1,-1)[0]

    #Generating the Predicted reference for gyro_y_error
    
    accel_x_model_path =  'reg_gyro_y.pkl'
    with open(accel_x_model_path , 'rb') as f:
        gyro_y = pickle.load(f)
    
    final_df = test.copy()
    final_df.rename(columns={'Gyro_Y':'feature_Gyro_Y'},inplace=True)
    gyro_y_pred_df=final_df[['feature_Gyro_Y']]
    pred_erro=gyro_y.predict(gyro_y_pred_df['feature_Gyro_Y'].values.reshape(-1,1))
    gyro_y_pred_df['pred_ref_gyro_y'] = gyro_y_pred_df['feature_Gyro_Y']-pred_erro.reshape(1,-1)[0]
    

    
    #Generating the Predicted reference for gyro_z_error
    
    accel_x_model_path =  'regr_gyro_z.pkl'
    with open(accel_x_model_path , 'rb') as f:
        gyro_z = pickle.load(f)
        
    final_df = test.copy()
    final_df.rename(columns={'Gyro_Z':'feature_Gyro_Z'},inplace=True)
    gyro_z_pred_df=final_df[['feature_Gyro_Z']]
    pred_erro=gyro_z.predict(gyro_z_pred_df['feature_Gyro_Z'].values.reshape(-1,1))
    gyro_z_pred_df['pred_ref_gyro_z'] = gyro_z_pred_df['feature_Gyro_Z']-pred_erro.reshape(1,-1)[0]

    predicted_reference_value_df =  pd.DataFrame()
    predicted_reference_value_df['pred_ref_accel_x']=accel_x_pred_df['pred_ref_accel_x'].values
    predicted_reference_value_df['pred_ref_accel_y']=accel_y_pred_df['pred_ref_accel_y'].values
    predicted_reference_value_df['pred_ref_accel_z']=accel_z_pred_df['pred_ref_accel_z'].values
    predicted_reference_value_df['pred_ref_gyro_x']=gyro_x_pred_df['pred_ref_gyro_x'].values
    predicted_reference_value_df['pred_ref_gyro_y']=gyro_y_pred_df['pred_ref_gyro_y'].values
    predicted_reference_value_df['pred_ref_gyro_z']=gyro_z_pred_df['pred_ref_gyro_z'].values
    
    return predicted_reference_value_df



with tab1:
    uploaded_file = st.file_uploader('Upload test dataset in .xlsx format', type='.xlsx', accept_multiple_files=False, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    response_container_qa = st.container()
    
    print('upload file first:',uploaded_file)
    if "uploaded_file_s" not in st.session_state:
        st.session_state.uploaded_file_s = False
        
    if uploaded_file or st.session_state.uploaded_file_s:
        st.session_state.uploaded_file_s = True
      
        if uploaded_file is not None:
           with st.spinner('Analysing your data..'):
                total_data=[]
                dataframe = pd.read_excel(uploaded_file)
                dataframe['Timestamp'] = dataframe['Timestamp'].apply(lambda x : datetime.datetime.fromtimestamp(x).strftime('%H:%M:%S.%f'))
                dataframe.set_index('Timestamp',inplace=True)
                dataframe.sort_index(inplace=True,ascending = True)
                dataframe.index = pd.to_datetime(dataframe.index)
                st.write('Analysing Number of samples in given dataset per second')
                samples_cal_df = dataframe.resample('1s')
                dataframe_l1=dataframe.resample('1s').mean()
                timestamp_test,len_test=get_sample_numbers(samples_cal_df)
                st.plotly_chart(px.line(x=timestamp_test,y=len_test,title='Number of samples in given second for test setup').update_layout(xaxis_title='sample number',yaxis_title='Number of samples'))
                colored_header(label='', description='', color_name='yellow-40')
                st.markdown('''Statistical summary of the given dataset''')
                st.dataframe(dataframe_l1.describe().style.highlight_max(axis=0))
                colored_header(label='', description='', color_name='yellow-40')
                st.plotly_chart(px.histogram(dataframe_l1,nbins=30,title='Data Distribution of the dataset'),use_container_width=True)
                colored_header(label='', description='', color_name='yellow-40')
                
                
with tab2:
    with st.container():
        if st.button('Run ML models on input data'):
           with st.spinner('Running ML on Uploaded dataset'):
               predicted_reference_value_df= run_models(dataframe_l1)   
               colored_header(label='', description='', color_name='yellow-40')
               result = st.container()
               with result:
                   st.plotly_chart(px.line(dataframe_l1,x=dataframe_l1.index,y=dataframe_l1.columns,title='Actual test data'),use_container_width=True)
                   st.plotly_chart(px.line(predicted_reference_value_df,x=predicted_reference_value_df.index,y=predicted_reference_value_df.columns,title='Corrected test data/Predicted reference data'),use_container_width=True)
                   colored_header(label='', description='', color_name='yellow-40')
                   st.markdown('''Statistical summary of the Prediction''')
                   st.plotly_chart(px.histogram(predicted_reference_value_df,nbins=20,title='Data Distribution of Predictions'),use_container_width=True)
    
