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
    choose = option_menu("Prediksi Harga Rumah", ["Home", "Dataset", "Predict", "Help"],
                             icons=['house', 'table', 'boxes','check2-circle'],
                             menu_icon="app-indicator", default_index=0,
                             styles={
            "container": {"padding": "5!important", "background-color": "10A19D"},
            "icon": {"color": "orange", "font-size": "25px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#3D5656"},
        }
        )
if choose=='Home':
    st.markdown('<h1 style = "text-align: center;"> Wisata pesarean Syaichona Kholil di Bangkalan</h1>', unsafe_allow_html = True)
    logo = Image.open('makam.jpg')
    st.image(logo, caption='')
    st.write('<p style = "text-align: justify;"> Wisata Makam Muhammad Syaikhona Kholil Bangkalan adalah salah satu tempat wisata yang berada di Kampung Senenan, Desa Kemayoran, Kecamatan Bangkalan, Kabupaten Bangkalan, ujung Barat Pulau Madura, Jawa Timur, Indonesia. Wisata Makam Muhammad Syaikhona Kholil Bangkalan adalah tempat wisata yangramai dengan wisatawan pada hari biasa maupun hari liburan. Tempat ini sangat indah dan bisa memberikan sensasi yang berbeda dengan aktivitas kita sehari hari. Wisata Makam Muhammad Syaikhona Kholil Bangkalan memiliki pesona sejarah yang sangat menarik untuk dikunjungi. Sangat di sayangkan jika anda berada di kota Bangkalan tidak mengunjungi wisata Makam Muhammad Syaikhona Kholil Bangkalan yang mempunyai keindahan yang tiada duanya tersebut. Wisata Makam Muhammad Syaikhona Kholil Bangkalan sangat cocok untuk mengisi kegiatan liburan anda, apalagi saat liburan panjang seperti libur nasional, ataupun hari ibur lainnya. Keindahan wisata Makam Muhammad Syaikhona Kholil Bangkalan ini sangatlah baik bagi anda semua yang berada di dekat atau di kejauhan untuk merapat mengunjungi tempat wisata Makam Muhammad Syaikhona Kholil di kota Bangkalan</p>', unsafe_allow_html = True)

if choose=='Dataset':
    st.markdown('<h1 style = "text-align: center;"> Data jumlah pengunjung pariwisata Syaichona Kholil </h1>', unsafe_allow_html = True)
    df = pd.read_csv('https://raw.githubusercontent.com/AriAndiM/dataset/main/data-pariwisata-syaikhona.csv')
    df
    st.markdown('<h1 style = "text-align: center;"> Fitur Dataset: </h1><ol type = "1" style = "text-align: justify; background-color: #f2a916; padding: 30px; border-radius: 20px;"><p>Dataset didapatkan dari Dinas Pemuda Olahraga Dan Pariwisata Kabupaten Bangkalan. Data diambil pada tahun 2010-2022</p><li><i><b>Bulan</b></i> merupakan bulan pengunjung datang ke wisata.</li><li><i><b>Jumlah</b></i> merupakan jumlah pengunjung wisata di setiap bulan.</li></ol>', unsafe_allow_html = True)

if choose=='Predict':
    st.markdown('<h1 style = "text-align: center;"> Prediksi jumlah pengunjung wisata pesarean Syaichona Kholil di Bangkalan</h1>', unsafe_allow_html = True)
    logo = Image.open('plot_mape.png')
    st.image(logo, caption='')
    pilih_bulan = st.selectbox(
        'Pilih Bulan',
        ('Januari', 'Februari', 'Maret' , 'April', 'Mei', 'Juni', 'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember'))
    btn = st.button('Prediksi')
    if btn:
        df = pd.read_csv('https://raw.githubusercontent.com/AriAndiM/dataset/main/data-pariwisata-syaikhona.csv')
        X = df['Bulan']
        y = df['Jumlah']
        X = X.values.reshape(-1, 1)
        y = y.values.reshape(-1, 1)

        # Membuat objek model Linear Regression
        model = LinearRegression()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,shuffle=False)
        # Melatih model
        model.fit(X_train, y_train)
        # Membuat prediksi pada data testing
        # y_pred = model.predict(X_test)
        bulan = pilih_bulan
        if bulan=='Januari':
            b = 1
        elif bulan=='Februari':
            b = 2
        elif bulan=='Maret':
            b = 3
        elif bulan=='April':
            b = 4
        elif bulan=='Mei':
            b = 5
        elif bulan=='Juni':
            b = 6
        elif bulan=='Juli':
            b = 7
        elif bulan=='Agustus':
            b = 8
        elif bulan=='September':
            b = 9
        elif bulan=='Oktober':
            b = 10 
        elif bulan=='November':
            b = 11
        elif bulan=='Desember':
            b = 12

        y_pred = model.predict([[b]])
        hasil = int(y_pred[0])
        st.write('Prediksi pengunjung pada bulan', b,'sebanyak :', hasil, 'pengunjung')

if choose=='Help':
    st.markdown('<h1 style = "text-align: center;"> Panduan : </h1><ol type = "1" style = "text-align: justify; background-color: #f2a916; padding: 30px; border-radius: 20px;"><li><i><b>Cara View Dataset</b></i> <ol type = "a"><li>Masuk ke sistem</li><li>Pilih menu dataset</li></ol></li><li><i><b>Cara Prediksi Pengunjung</b></i> <ol type = "a"><li>Pilih menu predict</li><li>Pilih bulan</li><li>Klik tombol prediksi</li></ol></li></ol>', unsafe_allow_html = True)
