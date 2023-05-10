#!/usr/bin/env python
# coding: utf-8

# # Customer Segmentation of a Wholesale Distributor Using K-means: Analysis of Purchase Profiles.

# In[1]:


# Import libs

# Data Manipulation and Visualization
import sklearn
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib as m
import matplotlib.pyplot as plt
import locale

# Machine Learning
from sklearn.cluster import KMeans
from sklearn import metrics
 
# This command allows the plots to be displayed directly in the output cells of the Jupyter notebook.
get_ipython().run_line_magic('matplotlib', 'inline')


# # Import Dataset

# In[2]:


# The data set refers to clients of a wholesale distributor. 
# It includes the annual spending in monetary units (m.u.) on diverse product categories
# https://archive.ics.uci.edu/ml/datasets/Wholesale+customers#


# In[3]:


# Formatting values 
locale.setlocale(locale.LC_ALL, 'pt_PT.UTF-8')

# Import Dataset
df_sales = pd.read_csv("C:/Users/antun/OneDrive/Documentos/Projetos/clusterizacao_distribuidor_atacadista/dados_clientes_atacado.csv")


# ### Data dictionary
# 
# * Fresh: annual spending (in monetary units) on fresh products;
# * Milk: annual spending (in monetary units) on milk products;
# * Grocery: annual spending (in monetary units) on grocery products;
# * Frozen: annual spending (in monetary units) on frozen products;
# * Detergents_Paper: annual spending (in monetary units) on detergents and paper products;
# * Delicassen: annual spending (in monetary units) on delicatessen products;
# * Channel: customers' channel, where 1 is Horeca (Hotel/Restaurant/Cafe) and 2 is Retail channel (nominal)
# * Region: customers' region, where 1 is Lisbon, 2 is Porto, and 3 is other regions.

# # Exploratory Data Analysis

# In[4]:


# Rows and columns
df_sales.shape


# In[5]:


# First rows
df_sales.head()


# In[6]:


# Data Types
df_sales.dtypes


# In[7]:


# Null values
df_sales.isnull().sum()


# In[8]:


# Column statistics
df_sales.describe()


# In[9]:


# Distribution between sales channels and region
sns.set_theme(style="ticks", palette="pastel")
plt.figure(1, figsize=(8,5))
ax = sns.countplot(y='Channel', hue='Region', data=df_sales)
ax.set_xlabel('Count')
ax.set_ylabel('Channel')

# Adding data values to bars
for i in ax.containers:
    ax.bar_label(i, label_type='edge')

plt.show()


# In[10]:


# Mean sales by channel
mean_channel = df_sales.groupby('Channel').mean()
mean_channel.iloc[:,1:]


# In[11]:


# Mean sales by region
mean_channel = df_sales.groupby('Region').mean()
mean_channel.iloc[:,1:]


# # Outliers

# In[12]:


# Checking for outliers
plt.boxplot(df_sales.iloc[:,2:])
plt.show()


# In[13]:


# Treating outliers using IQR (Interquartile Range)
df_sales2 = df_sales
for col in df_sales2.columns:
    q1 = df_sales2[col].quantile(0.25)
    q3 = df_sales2[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df_sales2 = df_sales2.loc[((df_sales2[col] >= lower_bound) & (df_sales2[col] <= upper_bound))]
df_sales2


# In[14]:


#  Checking for outliers after treatment
plt.boxplot(df_sales2.iloc[:,2:])
plt.show()

# I chose to use this new dataset to avoid possible distortions in the segmentation. 
# However, it is possible to repeat the process to further reduce the outliers. 
# It is important to remember that outliers can negatively influence data segmentation, 
# as the K-means algorithm is sensitive to these values, which can result in distorted clusters and incorrect information about the distribution of the data.


# In[15]:


# Distribution between cales channels and region
sns.set_theme(style="ticks", palette="pastel")
plt.figure(1, figsize=(8,5))
ax = sns.countplot(y='Channel', hue='Region', data=df_sales2)
ax.set_xlabel('Count')
ax.set_ylabel('Channel')

# Adding data values to bars
for i in ax.containers:
    ax.bar_label(i, label_type='edge')

plt.show()


# In[16]:


# Mean sales by channel
mean_channel = df_sales2.groupby('Channel').mean()
mean_channel.iloc[:,1:]


# In[17]:


# Mean sales by region
mean_channel = df_sales2.groupby('Region').mean()
mean_channel.iloc[:,1:]


# # Creating the model

# In[18]:


# Variable selection
X1 = df_sales2


# #### Defining number of clusters with the Elbow method

# In[19]:


# Inertia
inertia = []

# Loop to test K values.
for n in range(2 , 11):
    modelo = (KMeans(n_clusters = n,
                     init = 'k-means++', 
                     n_init = 10,
                     max_iter = 300, 
                     tol = 0.0001,  
                     random_state = 111, 
                     algorithm = 'elkan'))
    modelo.fit(X1)
    inertia.append(modelo.inertia_)

# Plot
plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(2 , 11) , inertia , 'o')
plt.plot(np.arange(2 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of clusters') , plt.ylabel('Inertia')
plt.show()


# In[20]:


# Creating the model with 3 clusters.
model_1 = (KMeans(n_clusters = 3,
                      init = 'k-means++', 
                      n_init = 10 ,
                      max_iter = 300, 
                      tol = 0.0001,  
                      random_state = 111  , 
                      algorithm = 'elkan') )

# Treinamento do modelo
model_1.fit(X1)

# Labels
labels_1 = model_1.labels_

# Centróides
centroids_1 = model_1.cluster_centers_


# In[21]:


# Shape labels
labels_1.shape


# In[22]:


# Typo
type(labels_1)


# In[23]:


# Converting the array to a dataframe
df_labels = pd.DataFrame(labels_1)


# In[24]:


df_sales2


# In[25]:


# Index
df_sales2_idx = df_sales2.reset_index()
df_labels_idx = df_labels.reset_index()

# Merging df_sales2 and the labels (clusters) found by the model.
df_result = df_sales2_idx.merge(df_labels_idx, left_index = True, right_index = True)

# rename clusters column
df_result.rename(columns = {0:"cluster"}, inplace = True)

df_result


# #### Results

# In[26]:


# Distribuição entre os Canais de Atendimento
sns.set_theme(style="ticks", palette="pastel")
plt.figure(1, figsize=(8,5))
ax = sns.countplot(y='Channel',data=df_result)
ax.set_xlabel('customer count')
ax.set_ylabel('Channel')

# Adicionando os valores dos dados nas barras
for i in ax.containers:
    ax.bar_label(i, label_type='edge')

plt.show()


# In[27]:


# Total sales by channel
sales_channel = df_result.iloc[:,1:9].groupby('Channel').sum()

# total sum 
sales_channel = sales_channel.assign(Total=sales_channel.sum(axis=1))

sales_channel.iloc[:, 1:]


# In[28]:


# Total sales by Region
sales_region = df_result.iloc[:,1:9].groupby('Region').sum()

# total sum 
sales_region = sales_region.assign(Total=sales_region.sum(axis=1))

sales_region.iloc[:, 1:]


# In[29]:


# Distribution between cales channels and region
sns.set_theme(style="ticks", palette="pastel")
plt.figure(1, figsize=(8,5))
ax = sns.countplot(y='Channel', hue='Region', data=df_result)
ax.set_xlabel('Count')
ax.set_ylabel('Channel')

# Adding data values to bars
for i in ax.containers:
    ax.bar_label(i, label_type='edge')

plt.show()


# ## Result by cluster

# In[30]:


md_cluster = df_result.groupby('cluster').mean()

md_cluster['Fresh'] = md_cluster['Fresh'].map(locale.currency)
md_cluster['Milk'] = md_cluster['Milk'].map(locale.currency)
md_cluster['Grocery'] = md_cluster['Grocery'].map(locale.currency)
md_cluster['Frozen'] = md_cluster['Frozen'].map(locale.currency)
md_cluster['Detergents_Paper'] = md_cluster['Detergents_Paper'].map(locale.currency)
md_cluster['Delicassen'] = md_cluster['Delicassen'].map(locale.currency)

md_cluster.iloc[:,3:9]

