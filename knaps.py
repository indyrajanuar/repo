import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from streamlit_option_menu import option_menu
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

with st.sidebar:
    choose = option_menu("Linear Regression (Polynomial)", ["Home", "Dataset", "Prepocessing", "Predict", "Help"],
                             icons=['house', 'table', 'boxes', 'boxes', 'check2-circle'],
                             menu_icon="app-indicator", default_index=0,
                             styles={
            "container": {"padding": "5!important", "background-color": "10A19D"},
            "icon": {"color": "orange", "font-size": "25px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#3D5656"},
        }
        )
if choose=='Home':
    st.markdown('<h1 style = "text-align: center;"> Prediksi Harga Rumah</h1>', unsafe_allow_html = True)
    logo = Image.open('makam.jpeg')
    st.image(logo, caption='')
    st.write('<p style = "text-align: justify;">Rumah merupakan salah satu kebutuhan pokok manusia, selain sandang dan pangan, rumah juga berfungsi sebagai tempat tinggal dan berfungsi untuk melindungi dari gangguan iklim dan makhluk hidup lainnya[1]. Tak kalah buruknya dengan emas, rumah pun bisa dijadikan sebagai sarana investasi masa depan karena pergerakan harga yang berubah dari waktu ke waktu, dan semakin banyak orang yang membutuhkan hunian selain kedekatan dengan tempat kerja, pusat perkantoran dan pusat bisnis, transportasi. dll tentunya akan cepat mempengaruhi harga rumah tersebut[2].</p>', unsafe_allow_html = True)
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write("Dr. Indah Agustien Siradjuddin")

if choose=='Dataset':
    st.markdown('<h1 style = "text-align: center;"> Data Harga Rumah </h1>', unsafe_allow_html = True)
    df = pd.read_csv('https://raw.githubusercontent.com/Shintaalya/repo/main/HARGA%20RUMAH%20JAKSEL.csv')
    df
    st.markdown('<h1 style = "text-align: center;"> Fitur Dataset: </h1><ol type = "1" style = "text-align: justify; background-color: #f2a916; padding: 30px; border-radius: 20px;"><p>Dataset ini diambil dari kaggle.com</p><li><i><b>HARGA</b></i> = harga dari rumah</li><li><i><b>LT</b></i> = Jumlah Luas Tanah</li><li><i><b>LB</b></i> = Jumlah Luas Bangunan</li><li><i><b>JKT</b></i> = Jumlah Kamar Tidur</li><li><i><b>JKM</b></i> = Jumlah Kamar Mandi</li><li><i><b>GRS</b></i> = Ada / Tidak Ada</li><li><i><b>KOTA</b></i> = Nama Kota</li></ol>', unsafe_allow_html = True)

if choose=='Prepocessing':
    st.markdown('<h1 style = "text-align: center;"> Prediksi Harga Rumah</h1>', unsafe_allow_html = True)
    st.write("Dari 7 Fitur")
    logo = Image.open('dataset.png')
    st.image(logo, caption='')
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write("Diseleksi menjadi 2 Fitur")
    logo = Image.open('dataset2.png')
    st.image(logo, caption='')
   
    
    
if choose=='Predict':
    st.markdown('<h1 style = "text-align: center;"> Prediksi Harga Rumah</h1>', unsafe_allow_html = True)
    logo = Image.open('plot_mape.png')
    st.image(logo, caption='')
    pilih_LT = st.selectbox(
        'Pilih LT',
        ('1', '2', '3' , '4', '5', '6', '7', '8', '9', '10'))
    btn = st.button('Prediksi')
    if btn:
        df = pd.read_csv('https://raw.githubusercontent.com/Shintaalya/repo/main/HARGA%20RUMAH%20JAKSEL.csv')
        X = df['Bulan']
        y = df['Jumlah']
        X = X.values.reshape(-1, 1)
        y = y.values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,shuffle=False)
        # Melatih model
        model.fit(X_train, y_train)
        # Membuat prediksi pada data testing
        # y_pred = model.predict(X_test)
      
        y_pred = model.predict([[b]])
        hasil = int(y_pred[0])
        st.write('Prediksi Harga', b,'sebanyak :', hasil, 'pengunjung')

if choose=='Help':
    st.markdown('<h1 style = "text-align: center;"> Panduan : </h1><ol type = "1" style = "text-align: justify; background-color: #f2a916; padding: 30px; border-radius: 20px;"><li><i><b>Cara View Dataset</b></i> <ol type = "a"><li>Masuk ke sistem</li><li>Pilih menu dataset</li></ol></li><li><i><b>Cara Prediksi Harga</b></i> <ol type = "a"><li>Pilih menu predict</li><li>Pilih bulan</li><li>Klik tombol prediksi</li></ol></li></ol>', unsafe_allow_html = True)
