# Upload dataset
from google.colab import files
files.upload()

# Import library yang dibutuhkan
import pandas as pd
import numpy as np
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt
import math

# Memunculkan 5 data teratas dari dataset yang telah diupload
dataset= pd.read_csv("CORONAFIX (1).csv")
dataset.head()

# Untuk mengetahui jumlah data pada dataset
print("#jumlah dataset saya : "+str(len(dataset.index)))

# Analisis deskriptif terhadap dataset yang dipilih
dataset.describe()

# Menampilkan grafik plot. Koordinat X merupakan Jumlah Kasus sedangkan koordinat Y merupakan Jumlah Kematian
dataset.plot(x='Kasus', y='Kematian', style='o')
plt.title('Pengaruh Jumlah Kasus terhadap Angka Kematian pada Bulan April di 20 Negara')
plt.xlabel('Jumlah Kasus Bulan April')
plt.ylabel('Jumlah Kematian Bulan April')
plt.show()

# Menentukan variabel independen (sumbu X) yaitu Kasus dan menentukan variabel dependen (sumbu Y) yaitu Kematian.
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Pembagian data menjadi 2 bagian yaitu untuk data training (training set) dan data test (test set) yaitu 80% untuk data training dan 20% untuk data test.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Mengimpor class LinearRegression dari library yang dibutuhkan untuk membuat model regresi. kemudian membuat objek regressor sebagai fungsi dari LinearRegression dan membuat model regresi untuk data training.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
print(regressor.intercept_)
print(regressor.coef_)
y_pred = regressor.predict(x_test)
dataframe = pd.DataFrame({'Data Sebenarnya' : y_test, 'Data Prediksi' : y_pred})
dataframe

# Menampilkan grafik plot dari data Kasus dan Kematian
sns.pairplot(dataset)

# Menampilkan hasil dari Test Set
plt.scatter(x_test, y_test, color = 'blue')
plt.plot(x_train, regressor.predict(x_train), color = 'black')
plt.title('Cases Vs Death on April (Test Set)')
plt.xlabel('Jumlah Kasus')
plt.ylabel('Jumlah Kematian')
plt.show()

# Menampilkan hasil dari Training Set
plt.scatter(x_train, y_train, color = 'blue')
plt.plot(x_train, regressor.predict(x_train), color = 'black')
plt.title('Cases Vs Death on April (Training Set)')
plt.xlabel('Jumlah Kasus')
plt.ylabel('Jumlah Kematian')
plt.show()
