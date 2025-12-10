import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_csv('Mall_Customers.csv')

df.rename(index=str, columns={
    'Annual Income (k$)': 'income',
    'Spending Score (1-100)': 'score'
}, inplace=True)

x = df.drop(['CustomerID', 'Gender'], axis=1)

st.header("isi dataset")
st.write(x)

# menampilkan panah elbow
clusters = []
for i in range(1,11):
    km = KMeans(n_clusters=i).fit(x)
    clusters.append(km.inertia_)

fig, ax=plt.subplots(figsize=(12,8))
sns.lineplot(x=list(range(1,11)), y=clusters, ax=ax)
ax.set_title('mencari elbow')
ax.set_xlabel('clusters')
ax.set_ylabel('inertia')

ax.annotate('Possible elbow point', xy=(3,140000), xytext=(3,50000),xycoords='data',
            arrowprops=dict(arrowstyle='-|>', connectionstyle='arc3', color='blue',lw=2))
ax.annotate('Possible elbow point', xy=(5,80000), xytext=(5,150000),xycoords='data',
            arrowprops=dict(arrowstyle='-|>', connectionstyle='arc3', color='blue',lw=2))

st.pyplot(fig)

st.sidebar.subheader("Nilai jumlah K")
clust = st.sidebar.slider("Pilih jumlah cluster :", 2,10,3,1)

def k_means(n_clust):
    kmean = KMeans(n_clusters=n_clust).fit(x)
    x['Labels'] = kmean.labels_

    fig2 = plt.figure(figsize=(10,8))  # Simpan figure ke variabel
    sns.scatterplot(x='income', y='score', hue='Labels', style='Labels', size='Labels', data=x, palette=sns.color_palette('hls',n_clust))

    for label in x['Labels'].unique():
        plt.annotate(label,
            (x[x['Labels']==label]['income'].mean(),
            x[x['Labels']==label]['score'].mean()), 
            horizontalalignment='center',
            verticalalignment='center',
            size=20,
            weight='bold',
            color='black')
    st.header('Cluster plot')
    st.pyplot(fig2)  # Gunakan fig2, bukan kosong
    st.write(x)

k_means(clust)

