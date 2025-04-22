import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time
latent_rep = np.load('results/VAE_latent_vectors_64.npy')
num_samples = 0
indices = np.random.choice(latent_rep.shape[0],size=num_samples)
latent_rep_sampled = latent_rep[indices]
# Sample from Gaussian
latent_dim = latent_rep.shape[1]
num_gaussian_sample = 2000
gaussian_samples = np.random.randn(num_gaussian_sample, latent_dim)
combined_samples = np.vstack([latent_rep_sampled, gaussian_samples])
print(f"训练样本数量: {latent_rep_sampled.shape[0]}")
print(f"高斯样本数量: {gaussian_samples.shape[0]}")
print(f"合并样本形状: {combined_samples.shape}")



print("开始3D t-SNE降维...")

start_time = time.time()
tsne_3d = TSNE(n_components=3, random_state=42, verbose=1)
latent_3d_combined = tsne_3d.fit_transform(combined_samples)
print(f"3D t-SNE完成，耗时: {time.time() - start_time:.2f}秒")
print(f"降维后形状: {latent_3d_combined.shape}")

latent_3d = latent_3d_combined[:num_samples]
gaussian_3d = latent_3d_combined[num_samples:]

# 可视化
plt.figure(figsize=(10, 8))
plt.scatter(latent_3d[:, 0], latent_3d[:, 1], s=5, c='blue', alpha=0.7)
plt.scatter(gaussian_3d[:, 0], gaussian_3d[:, 1], s=5, c='red', alpha=0.7)
plt.title('t-SNE Visualization of Sampled Latent Space (2000 points)')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.tight_layout()
plt.savefig('results/tsne_2d_visualization_sampled.png', dpi=300)
plt.show()



# 3D可视化
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter_train = ax.scatter(
    latent_3d[:, 0], 
    latent_3d[:, 1], 
    latent_3d[:, 2],
    s=3, 
    alpha=0.6,
    c='blue',  # 使用索引作为颜色
    cmap='training data'
)

scatter_gaussian = ax.scatter(
    gaussian_3d[:, 0],
    gaussian_3d[:, 1],
    gaussian_3d[:, 2],
    s=3,
    alpha=0.6,
    c='red',
    label='Gaussian sample'
)
ax.set_title('3D t-SNE Visualization of Latent Space')
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.set_zlabel('t-SNE Dimension 3')
plt.tight_layout()
plt.savefig('results/tsne_3d_visualization.png', dpi=300)
plt.show()