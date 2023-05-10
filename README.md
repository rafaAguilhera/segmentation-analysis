# Segmentation using K-means: Purchase Profiles Analysis.
Customer segmentation is a widely used method in marketing. According to Siqueira (1999), segmentation aims to better understand markets, identify niches and opportunities, and thus achieve a stronger competitive position.

In this article, we will address customer segmentation using the K-means machine learning algorithm. For this study, we will use a dataset available in the UCI Machine Learning repository. The dataset records the annual purchase value of 440 customers of a wholesale distributor in various categories of products.

[UCI Link](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers#)

## Exploring the dataset

To start, it is necessary to read the CSV file and then perform an exploratory analysis on the data.

![image](https://github.com/rafaAguilhera/segmentation-analysis/assets/119620977/8641c20f-d267-4413-8a00-938b21e4cbf8)

The dataset consists of 440 observations and 8 columns. No duplicates or null values were identified in this database.
To detect outliers, I used the boxplot, which is a quick way to identify outliers.

![image](https://github.com/rafaAguilhera/segmentation-analysis/assets/119620977/faf96100-df99-4676-83fe-64e27ed0cff8)


In this analysis, we will use the Interquartile Range (IQR) method to remove data that are below the first quartile and above the third quartile, keeping only the records that make up the 50% of the central data in our database.

```` 
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
````

After removing outliers, the dataset now contains 318 observations and has fewer outliers than the original. I chose to use this new data set to avoid possible segmentation distortions. However, you can repeat the process to further reduce the outliers. It is important to remember that outliers can negatively influence data segmentation, as the K-means algorithm is sensitive to these values, which can result in distorted clusters and incorrect information about data distribution.
Below is the boxplot generated with the new database.

![image](https://github.com/rafaAguilhera/segmentation-analysis/assets/119620977/8fccb2dc-f235-4a87-835d-98f676dcf36a)

When analyzing the new set of data, we observed that around 72% of customers who purchased from the company belong to channel 1, which includes commercial establishments, hotels, restaurants and cafes. These customers spent a total of €4,673,699.00, with an average of €20,122.60 per customer over the year. On the other hand, channel 2, which represents direct sales to customers, corresponds to approximately 28% of customers and generated a total of €2,925,876.00 in purchases, with an average spending per customer of €34,021.81.

Based on the analyzed data, it is possible to conclude that channel 2 customers have a higher average ticket compared to channel 1 customers. This may indicate an opportunity for the company to explore ways to increase the representativeness of direct sales to customers, aiming at a increased revenue and profitability.

We could expand this analysis to identify other insights, but for the purpose of this study, let's stick with clustering. For this, we will use the sklearn library and the KMeans algorithm.
First we will define the ideal number of groups. The result of the code below will be a graph showing the Elbow curve, which will allow us to identify the ideal number of groups for clustering. In the specific case of this study, we chose to choose 3 groups.

````
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
````
![image](https://github.com/rafaAguilhera/segmentation-analysis/assets/119620977/169d9405-5c80-4c60-98d0-0c109e45ea07)

Now we can create and train our model. The code below creates a clustering model using the KMeans algorithm with 3 groups, where the model's initialization and training parameters are defined. Then, the model is trained with the X1 matrix and labels are generated for each instance and centroids for each group.

* The parameters used for creating the model are:
* n_clusters: the desired number of clusters, in this case, 3.
* init: the method used to initialize the cluster centroids, in this case, 'k-means++', which is a technique that aims to initialize the centroids more efficiently than the random method.
* n_init: the number of times the algorithm will be run with different initial centroids.
* max_iter: the maximum number of iterations the algorithm will perform in each run.
* tol: the tolerance for the convergence of the algorithm.
* random_state: the seed used to generate random numbers and ensure that the model is trained in the same way every time it is run.
* algorithm: the method used to calculate the clusters, in this case, 'elkan', which is an optimized version of the KMeans algorithm.

````
# Variable selection
X1 = df_sales2

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
````
We will analyze the results of each cluster based on the average sales for each product category, presented in the following table: 

![image](https://github.com/rafaAguilhera/segmentation-analysis/assets/119620977/4d3dafbc-12c0-4339-b896-f0b1801aa4c5)

We can conclude that group zero is composed of customers who have a higher frequency of purchases of dairy, grocery, and cleaning products, which suggests opportunities for cross-selling. Additionally, it was possible to observe that the customers who buy the most fresh products are also the ones who buy the most frozen products. Based on this analysis, we can suggest to the wholesale supplier to direct their marketing actions towards these groups, aiming to increase sales by offering promotions and discounts to stimulate the purchase of these items together.

We have come to the end of this article, but before we close, I would like to thank you for reading this far. I hope you enjoyed the content and, who knows, maybe even learned something new! If you have any questions or want to exchange ideas on the subject, don't hesitate to leave a comment. See you next time!

[Article Link](https://www.linkedin.com/posts/aguilhera_activity-7060314634280460289-T5Uu?utm_source=share&utm_medium=member_desktop)
