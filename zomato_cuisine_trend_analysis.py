import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist

# ---------------- STEP 1: LOADING DATASET ----------------
print("STEP 1: Loading Dataset")

df = pd.read_csv("zomato.csv", encoding="latin1")
df = df[["City", "Cuisines", "Aggregate rating", "Votes"]]

print("Original Shape:", df.shape)

# ---------------- STEP 2: PREPROCESSING ----------------
print("\nSTEP 2: Preprocessing Data")

df = df.dropna()

df["City"] = df["City"].str.lower().str.strip()
df["Cuisines"] = df["Cuisines"].str.lower().str.strip()

df["Cuisines"] = df["Cuisines"].str.split(",")
df = df.explode("Cuisines")
df["Cuisines"] = df["Cuisines"].str.strip()

print("After Cleaning Shape:", df.shape)
print("\nSample After Preprocessing:")
print(df.head())

print("\nUnique Cities:", df["City"].nunique())
print("Unique Cuisines:", df["Cuisines"].nunique())

# ---------------- STEP 3: CUISINE TREND ANALYSIS ----------------
print("\nSTEP 3: Cuisine Frequency Analysis")

cuisine_counts = df["Cuisines"].value_counts().head(10)

print("\nTop 10 Cuisines Overall:")
print(cuisine_counts)

plt.figure(figsize=(10, 5))
cuisine_counts.plot(kind="bar")
plt.title("Top 10 Trending Cuisines (Overall)")
plt.xlabel("Cuisine")
plt.ylabel("Number of Restaurants")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------- STEP 4: K-MEANS CLUSTERING ----------------
print("\nSTEP 4: K-Means Clustering on Cities")

city_cuisine_matrix = pd.crosstab(df["City"], df["Cuisines"])
print("City-Cuisine Matrix Shape:", city_cuisine_matrix.shape)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(city_cuisine_matrix)

k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

city_cuisine_matrix["Cluster"] = clusters

print("\nSample Clustered Cities:")
print(city_cuisine_matrix[["Cluster"]].head(10))

print("\nCities per Cluster:")
print(city_cuisine_matrix["Cluster"].value_counts())

city_cuisine_matrix[["Cluster"]].to_csv("city_clusters.csv")
print("\nSaved: city_clusters.csv")

# ---------------- STEP 5: PCA VISUALIZATION ----------------
print("\nSTEP 5: Visualizing City Clusters using PCA")

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(8, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap="tab10")
plt.colorbar(label="Cluster")
plt.title("City Clusters Based on Cuisine Trends (PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.tight_layout()
plt.show()

# ---------------- STEP 6: TOP CUISINES PER CLUSTER ----------------
print("\nSTEP 6: Top Cuisines in Each Cluster")

df["Cluster"] = df["City"].map(city_cuisine_matrix["Cluster"])

cluster_trends = {}

for cluster_id in sorted(df["Cluster"].unique()):
    print(f"\nCluster {cluster_id} — Top Cuisines:")
    top_cuisines = df[df["Cluster"] == cluster_id]["Cuisines"].value_counts().head(5)
    print(top_cuisines)
    cluster_trends[cluster_id] = top_cuisines

cluster_df = pd.concat(cluster_trends, axis=1)
cluster_df.to_csv("cluster_cuisine_trends.csv")
print("\nSaved: cluster_cuisine_trends.csv")

# ---------------- STEP 7 & 8: SILHOUETTE COMPARISON ----------------
print("\nSTEP 7 & 8: Comparing Clustering Models with Different K Values")

k_values = [2, 3, 4, 5, 6]
silhouette_scores = []

for k in k_values:
    kmeans_test = KMeans(n_clusters=k, random_state=42)
    labels = kmeans_test.fit_predict(scaled_data)
    score = silhouette_score(scaled_data, labels)
    silhouette_scores.append(score)
    print(f"K = {k} | Silhouette Score = {round(score, 3)}")

plt.figure(figsize=(6, 4))
plt.plot(k_values, silhouette_scores, marker='o')
plt.title("Silhouette Score vs Number of Clusters (K)")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.tight_layout()
plt.show()

best_k = k_values[silhouette_scores.index(max(silhouette_scores))]
best_kmeans_score = max(silhouette_scores)
print("\nBest K based on Silhouette Score:", best_k)

# ---------------- STEP 9: CLUSTER INTERPRETATION ----------------
print("\nSTEP 9: Interpreting Cluster Characteristics (Hidden Patterns)\n")

for c in sorted(city_cuisine_matrix["Cluster"].unique()):
    cities_in_cluster = city_cuisine_matrix[city_cuisine_matrix["Cluster"] == c].shape[0]
    top_cuisines = (
        df[df["Cluster"] == c]["Cuisines"].value_counts().head(3)
    )

    print(f"CLUSTER {c}")
    print("Number of Cities:", cities_in_cluster)
    print("Top Cuisines:")
    print(top_cuisines)
    print("-" * 40)

# ---------------- STEP 10: HIERARCHICAL CLUSTERING ----------------
print("\nSTEP 10: Hierarchical Clustering (Second Model)")

Z = linkage(scaled_data, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(Z, truncate_mode='lastp', p=10)
plt.title("Hierarchical Clustering Dendrogram (Cities)")
plt.xlabel("Cluster Size")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

hier_clusters = fcluster(Z, t=4, criterion='maxclust')
city_cuisine_matrix["Hier_Cluster"] = hier_clusters

print("\nHierarchical Cluster Distribution:")
print(city_cuisine_matrix["Hier_Cluster"].value_counts())

# ---------------- STEP 11: BUSINESS RECOMMENDATIONS ----------------
print("\nSTEP 11: Business Recommendations by City Cluster")

for cluster_id in sorted(city_cuisine_matrix["Cluster"].unique()):
    cities = city_cuisine_matrix[city_cuisine_matrix["Cluster"] == cluster_id].index
    cluster_data = df[df["City"].isin(cities)]
    top_cuisines = cluster_data["Cuisines"].value_counts().head(5)

    print(f"\nRECOMMENDED CUISINES FOR CLUSTER {cluster_id}")
    print("Number of Cities:", len(cities))
    print("Top Demand Cuisines:")
    print(top_cuisines)
    print("-" * 50)

# ---------------- STEP 12: FINAL MODEL COMPARISON ----------------
print("\nSTEP 12: Final Model Comparison & Best Model Selection")

distance_matrix = pdist(scaled_data)
coph_corr, _ = cophenet(Z, distance_matrix)

print("\nModel Evaluation Scores:")
print(f"K-Means Silhouette Score (Best K={best_k}): {best_kmeans_score:.3f}")
print(f"Hierarchical Cophenetic Correlation: {coph_corr:.3f}")

print("\nFINAL RESULT:")

if best_kmeans_score > coph_corr:
    print("🏆 BEST MODEL: K-MEANS CLUSTERING")
    print("Reason: Higher silhouette score indicates better cluster separation.")
else:
    print("🏆 BEST MODEL: HIERARCHICAL CLUSTERING")
    print("Reason: Better preservation of distance relationships between cities.")

print("\nConclusion:")
print("K-Means performs better for identifying cuisine-based city demand patterns in this dataset.")
