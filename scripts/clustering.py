# import npy vector clustering
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

data_dir = Path(__file__).resolve().parent.parent / "data" / "z"

selected_folder = ["crawl-0.4-0-d", "move-ego--90-4", "jump-2", "raisearms-h-h", "rotate-x--5-0.8", "lieonground-down", "headstand"]

# ==========================================
# 1. 載入資料 (Load .npy files and assign labels for folder distinction)
# ==========================================
all_vectors = []
folder_labels = []  # 紀錄向量屬於哪個資料夾（用於畫圖顏色區分）

for folder_idx, folder_name in enumerate(selected_folder):
    target_path = data_dir / folder_name
    
    if not target_path.exists():
        print(f"Warning: Folder {target_path} does not exist, skipping...")
        continue

    # 搜尋底下所有 .npy 檔案
    npy_files = list(target_path.glob("*.npy"))
    
    for npy_file in npy_files:
        vec = np.load(npy_file)
        
        # 確保向量維度為 2D (N, D)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
            
        all_vectors.append(vec)
        folder_labels.append(np.full(vec.shape[0], folder_idx))

if not all_vectors:
    raise ValueError("No .npy files loaded. Please check your data_dir path and folder names.")

# 將所有載入的陣列串接起來
X = np.vstack(all_vectors)
folder_labels = np.concatenate(folder_labels)

print(f"Total loaded samples: {X.shape[0]}, Feature dimension: {X.shape[1]}")

# ==========================================
# 2. KMeans 聚類 (KMeans Clustering)
# ==========================================
num_clusters = len(selected_folder)  # 預設分群數等於 folder 數量（可自由調整）
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

# ==========================================
# 3. PCA 降維至 2D (PCA Projection to 2D)
# ==========================================
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X)
centers_2d = pca.transform(kmeans.cluster_centers_)  # 將 KMeans 中心點也降維至 2D

# ==========================================
# 4. 繪製視覺化圖表 (Visualization)
# ==========================================
plt.figure(figsize=(10, 8))

# 定義顏色對應（使用 tab10 / Set1 等清晰調色盤）
colors = plt.cm.tab10(np.linspace(0, 1, len(selected_folder)))

# 按資料夾（folder）依序畫出數據點，確保同一個 folder 擁有相同顏色
for folder_idx, folder_name in enumerate(selected_folder):
    mask = (folder_labels == folder_idx)
    plt.scatter(
        X_2d[mask, 0],
        X_2d[mask, 1],
        c=[colors[folder_idx]],
        label=folder_name,
        alpha=0.6,
        edgecolors='none',
        s=30
    )

# 標示出 KMeans 的聚類中心點 (Centers)
plt.scatter(
    centers_2d[:, 0],
    centers_2d[:, 1],
    c='black',
    marker='X',
    s=200,
    linewidths=2,
    label='Cluster Centers (KMeans)'
)

plt.title("2D PCA Projection of Vectors (Colored by Folder Source)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 圖例放在右側
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# 顯示圖表或儲存圖片
plt.savefig("clustering_result.png", dpi=300)