# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:00:22 2023

@author: Vijay Budhewar
"""
#importing all the necessary modules 
import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_extras.colored_header import colored_header
import datetime
import pickle

#The sidebar on the main streamlit app comes from st.sidebar defined below
with st.sidebar:
    st.title("Sensor Data Analysis ")
    
    st.markdown('''Write Project Description''')

#fuction to get the number of samples from test and the reference dataset
def get_sample_numbers(df):
    timestamp_test = []
    len_test =[]
    for x in list(df):
        timestamp_test.append(x[0])
        len_test.append(len(x[1]))
    return (list(range(1,len(len_test)+1)),len_test)

#Defining total number of streamlit app tabs , with theier names
tab1, tab2,tab3 = st.tabs(["Original dataset - Analysis", "ML based Error prediction","Compare results"])

#function to run the models- regression models to predict errors from different sesors 
def run_models(test):

    #Below code is copy pasted from developed notebook of Utilising models - the documentation for this has been already submitted 
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

    #Creating a dataframe to merge all the predicted reference values
    predicted_reference_value_df =  pd.DataFrame()
    predicted_reference_value_df['pred_ref_accel_x']=accel_x_pred_df['pred_ref_accel_x'].values
    predicted_reference_value_df['pred_ref_accel_y']=accel_y_pred_df['pred_ref_accel_y'].values
    predicted_reference_value_df['pred_ref_accel_z']=accel_z_pred_df['pred_ref_accel_z'].values
    predicted_reference_value_df['pred_ref_gyro_x']=gyro_x_pred_df['pred_ref_gyro_x'].values
    predicted_reference_value_df['pred_ref_gyro_y']=gyro_y_pred_df['pred_ref_gyro_y'].values
    predicted_reference_value_df['pred_ref_gyro_z']=gyro_z_pred_df['pred_ref_gyro_z'].values
    predicted_reference_value_df.index = final_df.index
    return predicted_reference_value_df


#Defining Tab1 - Original dataset - Analysis

with tab1:

    #Uploading the test  file
    uploaded_file = st.file_uploader('Upload test dataset in .xlsx format', type='.xlsx', accept_multiple_files=False, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    response_container_qa = st.container()
    
    print('upload file first:',uploaded_file)
    if "uploaded_file_s" not in st.session_state:
        st.session_state.uploaded_file_s = False
    #Streamlit supporting code to maintain state variables 
    if uploaded_file or st.session_state.uploaded_file_s:
        st.session_state.uploaded_file_s = True
      
        if uploaded_file is not None:
           with st.spinner('Analysing your data..'):
                total_data=[]
               #Below code is copy pasted from developed notebooks
                dataframe = pd.read_excel(uploaded_file)
                dataframe['Timestamp'] = dataframe['Timestamp'].apply(lambda x : datetime.datetime.fromtimestamp(x).strftime('%H:%M:%S.%f'))
                dataframe.set_index('Timestamp',inplace=True)
                dataframe.sort_index(inplace=True,ascending = True)
                dataframe.index = pd.to_datetime(dataframe.index)
                st.write('Analysing Number of samples in given dataset per second')
                samples_cal_df = dataframe.resample('1s')
                st.session_state.dataframe_l1=dataframe.resample('1s').mean()
               #getting number of samples from defined functions for test dataset
                timestamp_test,len_test=get_sample_numbers(samples_cal_df)
               #plotting the datasets on Tab1
                st.plotly_chart(px.line(x=timestamp_test,y=len_test,title='Number of samples in given second for test setup').update_layout(xaxis_title='sample number',yaxis_title='Number of samples'))
                colored_header(label='', description='', color_name='yellow-40')
                st.markdown('''Statistical summary of the given dataset''')
                st.dataframe(st.session_state.dataframe_l1.describe().style.highlight_max(axis=0))
                colored_header(label='', description='', color_name='yellow-40')
                st.plotly_chart(px.histogram(st.session_state.dataframe_l1,nbins=30,title='Data Distribution of the dataset'),use_container_width=True)
                colored_header(label='', description='', color_name='yellow-40')
                
#Defining Tab2-ML based Error prediction
with tab2:
    #Defining streamlit state variables 
    if "predicted_reference_value_df" not in st.session_state:
        st.session_state.predicted_reference_value_df = False
        
    #Defining the Tab2 in a container 
    with st.container():
        if st.button('Run ML models on input data'): #Definging the name of the button on Tab2
           with st.spinner('Running ML on Uploaded dataset'):
               st.session_state.predicted_reference_value_df= run_models(st.session_state.dataframe_l1)   #Calling run_model to generate predictions for each sensor 
               colored_header(label='', description='', color_name='yellow-40')
               result = st.container()
               with result:
                   #plotting the required plots using plotly on Tab2
                   st.plotly_chart(px.line(st.session_state.dataframe_l1,x=st.session_state.dataframe_l1.index,y=st.session_state.dataframe_l1.columns,title='Actual test data'),use_container_width=True)
                   st.plotly_chart(px.line(st.session_state.predicted_reference_value_df,x=st.session_state.predicted_reference_value_df.index,y=st.session_state.predicted_reference_value_df.columns,title='Corrected test data/Predicted reference data'),use_container_width=True)
                   colored_header(label='', description='', color_name='yellow-40')
                   st.markdown('''Statistical summary of the Prediction''')
                   st.plotly_chart(px.histogram(st.session_state.predicted_reference_value_df,nbins=20,title='Data Distribution of Predictions'),use_container_width=True)
#Defining Tab3 - Compare results
with tab3:
    #Uploading dataset for reference 
    uploaded_file_ref = st.file_uploader('Upload reference dataset in .xlsx format', type='.xlsx', accept_multiple_files=False, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    
    with st.container():
        st.write('To compare the results form ML model , please upload the reference data here')
        #Defining streamlit state variables for Tab3 
        if "uploaded_file_ref" not in st.session_state:
            st.session_state.uploaded_file_ref = False
        
        if uploaded_file_ref or st.session_state.uploaded_file_ref:
            st.session_state.uploaded_file_ref = True
        
            if uploaded_file_ref is not None:
               with st.spinner('Analysing your data..'):
                   #Below code is copy pasted from developed notebook
                   ref_df = pd.read_excel(uploaded_file_ref)
                   ref_df['Timestamp'] = ref_df['Timestamp'].apply(lambda x : datetime.datetime.fromtimestamp(x).strftime('%H:%M:%S.%f'))
                   ref_df.set_index('Timestamp',inplace=True)
                   ref_df.sort_index(inplace=True,ascending = True)
                   ref_df.index = pd.to_datetime(ref_df.index)
                   ref_df_sample=ref_df.resample('1s')
                   ref_df_l1=ref_df.resample('1s').mean()
                   ref_df_l1.columns = ['Gyro_X_ref','Gyro_Y_ref','Gyro_Z_ref','Accel_X_ref','Accel_Y_ref','Accel_Z_ref']
                   
                   
                   
                   
                  # lets merge/join them together on timestamp for visualizing the differences between test setup and reference setup -- Im maaping the test data to reference data
                   cumm_data = st.session_state.dataframe_l1.merge(ref_df_l1,how='inner',left_on = st.session_state.dataframe_l1.index ,right_on = ref_df_l1.index)
                   cumm_data.rename(columns = {'key_0':'Timestamp'},inplace=True)
                   cumm_data.set_index('Timestamp',inplace=True)
                   cumm_data.sort_index(inplace=True,ascending = True)
                # lets merge/join them together on timestamp for visualizing the differences between predicted reference and reference setup -- Im maaping the predicted reference data to reference data
                   pred_cumm_data = st.session_state.predicted_reference_value_df.merge(ref_df_l1,how='inner',left_on = st.session_state.predicted_reference_value_df.index ,right_on = ref_df_l1.index)
                   pred_cumm_data.rename(columns = {'key_0':'Timestamp'},inplace=True)
                   pred_cumm_data.set_index('Timestamp',inplace=True)
                   pred_cumm_data.sort_index(inplace=True,ascending = True)
                                       
                   
                   colored_header(label='', description='', color_name='yellow-40')
                   timestamp_test_ref,len_test_ref=get_sample_numbers(ref_df_sample)
                   st.plotly_chart(px.line(x=timestamp_test_ref,y=len_test_ref,title='Number of samples in given second for test setup').update_layout(xaxis_title='sample number',yaxis_title='Number of samples'))
            #Creating 6 buttons for 6 sensors on Tab3
            col1, col2, col3,col4,col5,col6 = st.columns(6)
            resp_cont=st.container()

            #Creating the dataframes from the various generated data -> predicted ref , ref and test dataset for each sensor 
            with col1:
                if st.button('Accel x'):
                    with resp_cont:
                        st.write('The features used for Acceleration X error reduction are :"Accel_Y","Accel_X","Gyro_Z"')
                        error_after_pred=pred_cumm_data['Accel_X_ref'].values-pred_cumm_data['pred_ref_accel_x'].values
                        error_before_pred=cumm_data['Accel_X_ref'].values-cumm_data['Accel_X'].values
                        error_df_accel_x=pd.DataFrame({'Error before prediction':error_before_pred,'Error after prediction':error_after_pred})
                        st.dataframe(error_df_accel_x.describe().style.highlight_max(axis=0))
                        st.markdown('''The predictions perfomed here are based on the ML model trained on historical data of error with an R2 of 0.58 and MSE of 0.006''')
                        #with st.button('Plot error distribution'):
                        st.plotly_chart(px.histogram(error_df_accel_x,nbins=20,title='Data Distribution of Errors'),use_container_width=True)
                        
            with col2:
                 if st.button('Accel y'):
                     with resp_cont:
                         st.write('The features used for Acceleration Y error reduction are :"Accel_Y","Accel_X"')
                         error_after_pred=pred_cumm_data['Accel_Y_ref'].values-pred_cumm_data['pred_ref_accel_y'].values
                         error_before_pred=cumm_data['Accel_Y_ref'].values-cumm_data['Accel_Y'].values
                         error_df_accel_y=pd.DataFrame({'Error before prediction':error_before_pred,'Error after prediction':error_after_pred})
                         st.dataframe(error_df_accel_y.describe().style.highlight_max(axis=0))
                         st.markdown('''The predictions perfomed here are based on the ML model trained on historical data of error with an R2 of 0.8 and MSE of 0.0012''')
                         #with st.button('Plot error distribution'):
                         st.plotly_chart(px.histogram(error_df_accel_y,nbins=20,title='Data Distribution of Errors'),use_container_width=True)
            with col3:
                 if st.button('Accel z'):
                     with resp_cont:
                         st.write('The features used for Acceleration Z error reduction are :"Accel_Y","Gyro_Z"')
                         error_after_pred=pred_cumm_data['Accel_Z_ref'].values-pred_cumm_data['pred_ref_accel_z'].values
                         error_before_pred=cumm_data['Accel_Z_ref'].values-cumm_data['Accel_Z'].values
                         error_df_accel_z=pd.DataFrame({'Error before prediction':error_before_pred,'Error after prediction':error_after_pred})
                         st.dataframe(error_df_accel_z.describe().style.highlight_max(axis=0))
                         st.markdown('''The predictions perfomed here are based on the ML model trained on historical data of error with an R2 of 0.53 and MSE of 0.03''')
                         #with st.button('Plot error distribution'):
                         st.plotly_chart(px.histogram(error_df_accel_z,nbins=20,title='Data Distribution of Errors'),use_container_width=True)
            with col4:
                 if st.button('Gyro x'):
                     with resp_cont:
                         st.write('The features used for  Gyro X error reduction is :"Gyro_X"')
                         error_after_pred=pred_cumm_data['Gyro_X_ref'].values-pred_cumm_data['pred_ref_gyro_x'].values
                         error_before_pred=cumm_data['Gyro_X_ref'].values-cumm_data['Gyro_X'].values
                         error_df_gyro_x=pd.DataFrame({'Error before prediction':error_before_pred,'Error after prediction':error_after_pred})
                         st.dataframe(error_df_gyro_x.describe().style.highlight_max(axis=0))
                         st.markdown('''The predictions perfomed here are based on the ML model trained on historical data of error with an R2 of 0.92 and MSE of 0.0002''')
                         #with st.button('Plot error distribution'):
                         st.plotly_chart(px.histogram(error_df_gyro_x,nbins=20,title='Data Distribution of Errors'),use_container_width=True)
            with col5:
                 if st.button('Gyro y'):
                     with resp_cont:
                         st.write('The features used for Gyro Y error reduction is :"Gyro_X"')
                         error_after_pred=pred_cumm_data['Gyro_Y_ref'].values-pred_cumm_data['pred_ref_gyro_y'].values
                         error_before_pred=cumm_data['Gyro_Y_ref'].values-cumm_data['Gyro_Y'].values
                         error_df_gyro_y=pd.DataFrame({'Error before prediction':error_before_pred,'Error after prediction':error_after_pred})
                         st.dataframe(error_df_gyro_y.describe().style.highlight_max(axis=0))
                         st.markdown('''The predictions perfomed here are based on the ML model trained on historical data of error with an R2 of 0.99 and MSE of 0.0073''')
                        # with st.button('Plot error distribution'):
                         st.plotly_chart(px.histogram(error_df_gyro_y,nbins=20,title='Data Distribution of Errors'),use_container_width=True)
            with col6:
                 if st.button('Gyro z'):
                     with resp_cont:
                         st.write('The features used for Gyro Z error reduction is :"Gyro_Z"')
                         error_after_pred=pred_cumm_data['Gyro_Z_ref'].values-pred_cumm_data['pred_ref_gyro_z'].values
                         error_before_pred=cumm_data['Gyro_Z_ref'].values-cumm_data['Gyro_Z'].values
                         error_df_gyro_z=pd.DataFrame({'Error before prediction':error_before_pred,'Error after prediction':error_after_pred})
                         st.dataframe(error_df_gyro_z.describe().style.highlight_max(axis=0))
                         st.markdown('''The predictions perfomed here are based on the ML model trained on historical data of error with an R2 of 0.95 and MSE of -0.00034''')
                         #with st.button('Plot error distribution'):
                         st.plotly_chart(px.histogram(error_df_gyro_z,nbins=20,title='Data Distribution of Errors'),use_container_width=True)
            colored_header(label='', description='', color_name='yellow-40')
            st.plotly_chart(px.line(pred_cumm_data,x=pred_cumm_data.index,y=pred_cumm_data.columns,title='Actural reference vs predicted Reference'),use_container_width=True)
