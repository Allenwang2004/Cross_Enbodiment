# import npy vector clustering
import re
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

data_root = Path(__file__).resolve().parent.parent / "data"
sources = {
    "z": data_root / "z",
    "retargeting_z": data_root / "retargeting_z",
}


def natural_key(path: Path):
    """把檔名尾端的數字當成整數排序，避免 _10 排在 _9 前面。"""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", path.stem)]


def prefix_of(folder_name: str) -> str:
    """取資料夾名稱第一段當作群組（例如 move-ego-0-2 -> move）。"""
    return folder_name.split("-")[0]


# ==========================================
# 1. 挑選資料夾：同樣開頭的（例如所有 move-*）只取一組
# ==========================================
group_to_folder = {}
for src_dir in sources.values():
    if not src_dir.exists():
        continue
    for folder in sorted(p for p in src_dir.iterdir() if p.is_dir()):
        group_to_folder.setdefault(prefix_of(folder.name), folder.name)

selected_folder = [group_to_folder[g] for g in sorted(group_to_folder)]
print(f"Selected folders ({len(selected_folder)}): {selected_folder}")

# ==========================================
# 2. 載入資料：每個 .npy 取整條軌跡的平均 z
# ==========================================
all_vectors = []
folder_labels = []   # 該向量屬於哪個資料夾（顏色）
source_labels = []   # 該向量來自 z 還是 retargeting_z（標記形狀）

for src_idx, (src_name, src_dir) in enumerate(sources.items()):
    if not src_dir.exists():
        print(f"Warning: Source {src_dir} does not exist, skipping...")
        continue

    for folder_idx, folder_name in enumerate(selected_folder):
        target_path = src_dir / folder_name

        if not target_path.exists():
            print(f"Warning: Folder {target_path} does not exist, skipping...")
            continue

        npy_files = sorted(target_path.glob("*.npy"), key=natural_key)
        if not npy_files:
            print(f"Warning: No .npy under {target_path}, skipping...")
            continue

        # 每個檔案是一次個別的 retargeting，取整條軌跡的平均 z。
        # 最後一幀是 goal embedding（tracking_inference 的滑動窗只剩一幀），
        # 語意上跟 data/z 的 reward embedding 不同；平均後再投回半徑
        # sqrt(d) 的球面，才跟 project_z / data/z 在同一個空間。
        for npy_file in npy_files:
            vec = np.load(npy_file)
            if vec.ndim == 1:
                vec = vec.reshape(1, -1)

            z = vec.mean(axis=0)
            z = z / np.linalg.norm(z) * np.sqrt(z.shape[0])
            all_vectors.append(z)
            folder_labels.append(folder_idx)
            source_labels.append(src_idx)

if not all_vectors:
    raise ValueError("No .npy files loaded. Please check your data_dir path and folder names.")

X = np.vstack(all_vectors)
folder_labels = np.asarray(folder_labels)
source_labels = np.asarray(source_labels)

print(f"Total loaded samples: {X.shape[0]}, Feature dimension: {X.shape[1]}")

# ==========================================
# 3. PCA 降維至 2D（用全部資料 fit，三張圖共用同一個投影空間）
# ==========================================
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X)

# ==========================================
# 4. 繪製視覺化圖表 (Visualization)
# ==========================================
out_dir = Path(__file__).resolve().parent.parent / "data"
colors = plt.cm.tab10(np.linspace(0, 1, len(selected_folder)))
source_names = list(sources)
markers = {0: "o", 1: "^"}  # z / retargeting_z


def plot(src_indices, title, out_name, with_source_in_label):
    plt.figure(figsize=(10, 8))

    for src_idx in src_indices:
        for folder_idx, folder_name in enumerate(selected_folder):
            mask = (folder_labels == folder_idx) & (source_labels == src_idx)
            if not mask.any():
                continue
            label = folder_name
            if with_source_in_label:
                label = f"{folder_name} ({source_names[src_idx]})"
            plt.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                c=[colors[folder_idx]],
                marker=markers.get(src_idx, "s"),
                label=label,
                alpha=0.8,
                edgecolors="none",
                s=80,
            )

    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = out_dir / out_name
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


# (1) 兩個來源疊在一起
plot(
    range(len(source_names)),
    "2D PCA of Mean-z Vectors (Color: Folder, Marker: Source)",
    "clustering_result_all.png",
    with_source_in_label=True,
)

# (2)(3) 每個來源各一張
for src_idx, src_name in enumerate(source_names):
    plot(
        [src_idx],
        f"2D PCA of Mean-z Vectors - {src_name}",
        f"clustering_result_{src_name}.png",
        with_source_in_label=False,
    )
