from sklearn.datasets import make_moons
from sklearn.decomposition import KernelPCA
from matplotlib import pyplot as plt  

X, y = make_moons(n_samples= 100,random_state= 1)

k_pca = KernelPCA(n_components=2, kernel='rbf', gamma=15)

X_kpca = k_pca.fit_transform(X)

plt.scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], color='red', marker='^', alpha= 0.5)
plt.scatter(X_kpca[y==1, 0], X_kpca[y==1, 1], color='blue', marker='^', alpha= 0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.show()