import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Input file CSV dari pengguna
st.sidebar.header("Upload Dataset Anda")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

def load_data(file):
    data = pd.read_csv(file)
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data['Month'] = data['Date'].dt.month
    return data

if uploaded_file is not None:
    data = load_data(uploaded_file)
else:
    data = load_data("dataset/supermarket_sales - Sheet1.csv")

# Streamlit UI
st.title("Prediksi Penjualan Supermarket dengan KNN & SVM")

# Menampilkan dataset awal
st.write("Dataset Awal", data.head())

# Visualisasi distribusi produk
st.subheader("Distribusi Produk")
fig, ax = plt.subplots(figsize=(8, 6))
data['Product line'].value_counts().plot(kind='bar', color='skyblue', ax=ax)
ax.set_title("Distribusi Produk")
ax.set_xlabel("Product Line")
ax.set_ylabel("Jumlah Penjualan")
plt.xticks(rotation=45)
st.pyplot(fig)

# Pilihan model
model_option = st.selectbox("Pilih Model untuk Menampilkan Hasil", ["KNN", "SVM"])

# Membaca hasil prediksi dan rekomendasi dari file CSV berdasarkan pilihan model
if model_option == "KNN":
    predictions_df = pd.read_csv('./result/knn/predictions_and_recommendations_knn.csv')
    metrics_df = pd.read_csv('./result/knn/model_metrics_knn.csv')
elif model_option == "SVM":
    predictions_df = pd.read_csv('./result/svm/predictions_and_recommendations_svm.csv')
    metrics_df = pd.read_csv('./result/svm/model_metrics_svm.csv')

# Visualisasi Prediksi
st.subheader("Prediksi Penjualan dan Rekomendasi Restock")
fig1, ax1 = plt.subplots(figsize=(10, 6))
width = 0.4
x = np.arange(len(predictions_df['Product line']))

bars1 = ax1.bar(x - width/2, predictions_df['Predicted_Quantity'], 
                 color='blue', label='Predicted Quantity', width=width)
bars2 = ax1.bar(x + width/2, predictions_df['Recommended_Quantity'], 
                 color='orange', alpha=0.5, label='Recommended Quantity', width=width)

ax1.set_title('Prediksi dan Rekomendasi Restock')
ax1.set_xlabel('Product Line')
ax1.set_ylabel('Quantity')
ax1.set_xticks(x)
ax1.set_xticklabels(predictions_df['Product line'], rotation=45)
ax1.legend()

for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    height1 = bar1.get_height()
    height2 = bar2.get_height()
    ax1.annotate(f'{int(height1)}', xy=(bar1.get_x() + bar1.get_width() / 2, height1),
                 xytext=(0, 3), textcoords="offset points",
                 ha='center', va='bottom')
    ax1.annotate(f'{int(height2)}', xy=(bar2.get_x() + bar2.get_width() / 2, height2),
                 xytext=(0, 3), textcoords="offset points",
                 ha='center', va='bottom')

st.pyplot(fig1)
