import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier



st.title("LINEAR REGRESSION (POLYNOMIAL) ")
st.write("##### Dr. Indah Agustien Siradjuddin, S.Kom., M.Kom ")
st.write("==============================================================")

data_set_description, modeling, implementation = st.tabs(["Data Set Description", "Modeling", "Implementation"])

with data_set_description:
    st.write("""# Data Set Description """)
    st.write("# Description ")
    st.write("Data Set Ini Adalah : Klasifikasi Harga Rumah di JakSel dan Tebet")
    st.write("""Dataset Harga Rumah merupakan daftar harga rumah yang terbagi menjadi 2 data, yaitu data harga rumah daerah Jaksel dan data harga rumah daerah Tebet. Data diambil dan dikumpulkan dari beberapa website penjualan seperti rumah123.com""")
    st.write("""Terdapat 7 kolom """)
    st.write("""Yaitu :

1. HARGA = harga dari rumah.
2. LT = jumlah luas tanah.
3. LB = jumlah luas bangunan.
4. JKT = jumlah kamar tidur.
5. JKM = jumlah kamar mandi.
6. GRS = ada/tidak ada
7. KOTA = nama kota.
    """)
    
    st.
    st.write("Link Dataset pada kaggle : https://www.kaggle.com/datasets/wisnuanggara/daftar-harga-rumah")
    st.write("Link github Aplikasi : https://github.com/Shintaalya/repo")

with modeling:
    st.write("masih kosong")
    
with implementation:
    with st.form("my_form"):
        st.subheader("Implementasi")
        Age = st.number_input('Masukkan umur (Age) : ')
        Gender = st.number_input('Masukkan jenis kelamin berupa angka 1 : Laki-laki, 0 : Perempuan (Gender) : ')
        Total_Bilirubin = st.number_input('Masukkan total bilirubin dalam darah - Berupa angka desimal (Total_Bilirubin) : ')
        Direct_Bilirubin = st.number_input('Masukkan direct bilirubin - Berupa angka desimal (Direct_Bilirubin) : ')
        Alkaline_Phosphotase = st.number_input('Masukkan Alkaline phosphotase - Berupa angka desimal (Alkaline_Phosphotase) : ')
        Alamine_Aminotransferase = st.number_input('Masukkan Alamine Aminotransferase - Berupa angka desimal (Alamine_Aminotransferase) : ')
        Aspartate_Aminotransferase = st.number_input('Masukkan Aspartate Aminotransferase - Berupa angka desimal (Aspartate_Aminotransferase) : ')
        Total_Protiens = st.number_input('Masukkan Total Protiens - Berupa angka desimal (Total_Protiens) : ')
        Albumin = st.number_input('Masukkan ALbumin - Berupa angka desimal (Albumin) : ')
        Albumin_And_Globulin_Ratio = st.number_input('Masukkan Albumin dan Globulin Ratio - Berupa angka desimal (Albumin_And_Globulin_Ratio) : ')
        
        model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi?',
                ('Gaussian Naive Bayes', 'K-NN', 'Decision Tree'))

        prediksi = st.form_submit_button("Submit")
        if prediksi:
            inputs = np.array([
                Age,
                Gender,
                Total_Bilirubin,
                Direct_Bilirubin,
                Alkaline_Phosphotase,
                Alamine_Aminotransferase,
                Aspartate_Aminotransferase,
                Total_Protiens,
                Albumin,
                Albumin_And_Globulin_Ratio,
                
            ])

            df_min = X.min()
            df_max = X.max()
            input_norm = ((inputs - df_min) / (df_max - df_min))
            input_norm = np.array(input_norm).reshape(1, -1)

            if model == 'Gaussian Naive Bayes':
                mod = gaussian
            if model == 'K-NN':
                mod = knn 
            if model == 'Decision Tree':
                mod = dt

               
            input_pred = mod.predict(input_norm)


            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Pemodelan :', model)

            st.write(input_pred)
