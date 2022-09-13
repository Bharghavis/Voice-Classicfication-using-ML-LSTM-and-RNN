import streamlit as st
import numpy as np
import tensorflow as ty
import os,urllib
import librosa #to extract speech features
def main():
   #print(cv2.__version__)
   selected_box=st.sidebar.selectbox(
   'Choose an option..',
   ('LSTM_and_RNN_Voice_Classification_using_ML','view source code')
   )
   if selected_box=='LSTM_and_RNN_Voice_Classification_using_ML':
       st.sidebar.success('To try by yourself by adding an audio file.')
       application()
   if selected_box=='view source code':
       st.code(get_file_content_as_string("appSER.py"))

@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url='https://github.com/Bharghavis/Voice-Classicfication-using-ML-LSTM-and-RNN'
    response=urllib.request.urlopen(url)
    return response.read().decode('utf-8')

@st.cache(show_spinner=False)
def load_model():
    model=tf.keras.models.load_model('mymodel.h5')
    return model

def application():
    models_load_state=st.text('/n loading models..')
    model = load_model()
    models_load_state.text('\n Models loading..complete')

    files_to_be_uploaded=st.files_to_be_uploader("Choose an audio...",type="wav")

    if files_to_be_uploaded:
        st.audio(files_to_be_uploaded,format='audio/wav')
        st.sucess('Emotion of the audio is' + predict(model,files_to_be_uploaded))

def extract_mfcc(wav_file_name):
    #This function extracts mfcc features and obtains the mean of each dimension
    #input: path_to_wav_file
    #output: mfcc_features'''
    y,sr=librosa.load(wav_file_name)
    mfccs=np.mean(librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40).T,axis=0)
    return mfcc

def predict(model,wav_filepath):
    emotions={1:'neutral',2:'calm',3:'happy',4:'sad',5:'angry',6:'fearful',7:'disgust',8:'suprised'}
    test_point=extract_mfcc(wav_filepath)
    test_point=np.reshape(test_point,newshape=(1,40,1))
    predictions=model.predict(test_point)
    print(emotions[np.argmax(predictions[0]) + 1])

    return emotions[np.argmax(predictions[0])+ 1]
  
 if __name__ == "__main__" :
   main()
