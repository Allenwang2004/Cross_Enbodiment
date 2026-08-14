# `scripts/` 檔案總覽

27 個腳本，橫跨三代設計。這份文件按**流程階段**整理，並標明每個檔案**現在還能不能跑**。

判讀狀態標記：

| 標記 | 意思 |
|---|---|
| 🟢 **現役** | bilevel pipeline 現在還會用到，或產出物是 `model/bilevel/config.py` 指向的東西 |
| 🟡 **可跑但已被取代** | 程式沒壞，但它的角色已由 `model/bilevel/` 裡的模組接手 |
| 🔴 **需先重建輸入** | 依賴 `data/retargeted_motion/`，那個目錄**已經不存在**（retargeting 改成 runtime 可微函數了）。要跑得先用 `qpos_retarget.py` 重新生一份 |
| ⚫ **廢棄** | 前提已消失，直接跑會炸 |

---

## 1. 資料生成（來源動作）

### 🟢 `metamotivo_motion_rollout.py`
整條 pipeline 的**源頭**。用 Meta Motivo 的 FB-CPR 預訓練模型在 HumEnv 裡跑出人形動作。

- `--z-mode reward`：從 humenv 的獎勵名稱（`move-ego-0-2`、`jump-2`…）零樣本推出 context vector `z`
- `--z-mode random`：`z ~ N(0,I)` 投影到單位球，動作沒有語意
- `--tasks-file docs/humenv_all_tasks_official.txt`：批次跑完 54 個 task × 10 trials

輸出 → `data/origin_motion/<task>/<task>_<trial>.npz`（key `qpos`）、`data/z/<task>/`、`outputs/robot_motion_video/`（只錄每個 task 的第一次）。

> ⚠️ **執行緒數是數值的一部分。** torch 按 thread 數切 matmul，改變 float32 的歸約順序，模擬會放大這個差異。`--threads 32`（預設）bit-exact 但 155 ms/step；`--threads 1` 快 11 倍（14 ms/step）但只有 ~1e-5 的漂移。

輸出 → `data/origin_action/<task>/`、`outputs/origin_action/regen_report.csv`。

### 🟢 `split_tasks.py`
把 `docs/humenv_all_tasks_official.txt` 切 train/test。**在 task 層級切，不在 row 層級**——同一個 `reward_name` 的 10 個 trial 動作高度相關（同一個 z 分布），按 row 切會讓近乎重複的動作洩漏到測試集，把分數灌水。`model/bilevel/data.py:352` 用的是同一套 43/11 切法。

### 🟡 `clustering.py`
`data/z/` 裡的 z 向量做 KMeans + PCA 降到 2D 畫圖，看不同 task 的 z 有沒有分開。純探索用，硬編了 7 個資料夾名稱，輸出 `clustering_result.png`。

---

## 2. 身體生成與致動器校正

這一段的**因果鏈很重要**，不是四支獨立工具：

```
scale_robot.py  ──產生 13 具身體──►  assets/robots/
       │                                   │
       │  （11 具用了 --no-actuator-scale） │
       ▼                                   ▼
                              audit_bodies.py 發現：13 具裡只有 8 具
                              連自己的 rest pose 都撐不住
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
        regen_bodies.py（用預測的負載模型）          calibrate_actuators.py（用量測的力矩需求）
        → assets/robots_scaled/  ❌ 反而更糟          → assets/robots_calib/  ✅ 現在用這個
```

### 🟢 `scale_robot.py`
從 `assets/robots/adult/robot.xml` 生出縮放變體。腿 / 手臂 / 軀幹 / 頭四組各有獨立的**長度**與**粗細**縮放。body/joint/actuator 的名稱和順序完全不動，所以預訓練的 Meta Motivo 模型還能用（obs/action 維度不變）。

縮放後會跑一次 FK 自動修正 Pelvis 根部高度，讓腳在預設姿勢還是踩在 z=0。輸出 `assets/robots/<label>/robot.xml` + `parameter.json`。

### 🟢 `audit_bodies.py` — Stage 0 / R1 關卡
**這支必須在任何訓練之前跑。** 13 具身體裡有 12 具沿用**成人的** gainprm/biasprm/forcerange，但質量從 38 kg（child）到 210 kg（heavy）——只有 `elderly` 被重算過。

量的是**準靜態下界**：對 N 個真實參考幀（p=0 的 naive retarget），設 `qvel = qacc = 0` 後跑 `mj_inverse`，得到「純粹把這個姿勢**撐住**」所需的力矩，再算

```
headroom_j = forcerange_j / |qfrc_inverse_j|      （≥ 1 才可行）
```

連不動都撐不住的幀，追蹤起來更不可能。結果：`heavy` 0.03、`short_stocky` 0.04、`short_limbed` 0.09、`giant` 0.48、`pear_shaped` 0.96。撐不住自己的身體只會貢獻梯度噪音，所以**訓練用哪 10 具身體是這份 audit 決定的，不是標籤清單決定的**。

輸出 → `outputs/body_audit*.json`。

**實測反而更糟**：它把每具身體的 headroom 拉向成人的比值，而不是給出餘裕——`heavy` 只到 0.11（還差 10 倍），`child` 反而從 7.54 掉到 3.00、`petite` 從 1.41 掉到 0.65。淨損兩具可用身體，一具也沒救回。結論是**力矩需求根本不在 rest pose**。

輸出寫到 `assets/robots_scaled/`（刻意不覆蓋，避免讓既有 checkpoint 的比較數字失效）。

### 🟢 `calibrate_actuators.py` — 成功的那條路（`config.robots_dir` 指的就是它）
不預測，**量測**。對一批真實參考姿勢算出每個致動器實際要出的靜態力矩，然後把致動器尺寸訂到「這具身體在同一段動作上，擁有和成人一樣的餘裕」：

```
tau_b[j]        = |qfrc_inverse[j]| 在參考姿勢上的高百分位（預設 P90）
h_adult[j]      = forcerange_adult[j] / tau_adult[j]      成人自己的餘裕
forcerange_b[j] = tau_b[j] * h_adult[j]
```

再加一個地板條件，保證 rest pose 至少有 `--min-rest`（預設 3.0）的餘裕。

> 這是一個**建模決定**而不是 bug fix，而且是有辯護理由的：更重的人本來就有按比例更強的肌肉。跨形態研究要的是身體因**幾何**而不同，不是某些身體天生殘廢。

關鍵細節：`gainprm[0]`、`biasprm[0]`、`biasprm[1]` **三個一起**乘同一個 k。因為伺服的平衡角 `q* = −(g·ctrl + b0)/b1`，三個同乘會讓 q* 不變，所以 `ctrl ∈ [−1,1] ⟺ qpos ∈ jnt_range` 這個恆等式仍精確成立——那是 bilevel 的 BC target 和可行性懲罰依賴的前提。

輸出 → `assets/robots_calib/`（12 具，`heavy` 被排除）。

### 🟢 `write_body_splits.py`
把權威的 train/test/unused 切分寫進每具身體的 `parameter.json`。

修的是一個**過期標記**：commit 9495329 給 `pear_shaped` 和 `teen` 標了 `"split": "val"`，但從來沒有任何程式讀過這個欄位——而 Stage 0 audit 選出來的切分把這兩具放在 **train**。一個沒人讀、又跟真實設定矛盾的欄位，正是幾年後會被人當真的東西，所以這裡把它變成活的。`model/bilevel/data.py:load_body` 會 assert 檔案和 config 一致，兩邊不能再默默漂開。

### 🟢 `export_skeleton_json.py`
把 MJCF 在 rest pose（qpos=0）下每個 body 的世界座標變換 + geom 匯出成 JSON，給 Blender 建 armature 用。

重點在**不要自己手刻 MJCF 的四元數組合/正規化規則**——讓 MuJoCo 自己 `mj_forward` 算出權威的 rest-pose 世界變換，只把攤平的結果交出去。輸出 `assets/robots/<label>/skeleton.json`。

### 🟡 `make_child_origact_xml.py`
組一具混血身體：child 的縮小幾何/慣量 + 成人**未縮放**的原始致動器。給 `model/train_explore.py` 那個純運動學探索實驗用，目的是把「child 的形態在成人級力量驅動下能不能學會維持物理合理」從致動器縮放的影響裡隔離出來。

實作很簡單（兩個 XML 的 actuator 名稱/joint/順序完全相同，整塊 `<actuator>` 換掉就好），但會先 assert 這個恆等式。輸出 `assets/robots/child/robot_origact.xml`。

---

## 3. Retargeting（舊的離線版）

> **這一整節已經被 `model/bilevel/retarget.py` 取代。** 現在的 retargeting 是 runtime、參數化、可微分的——那正是重新設計的核心（見 [retarget.md](retarget.md)）。這兩支保留下來，是因為 **p = 0 精確重現 `qpos_retarget.py:91` 的 `retarget_qpos`**，`tests/test_retarget.py` 拿它當黃金標準在 assert。

### 🟡 `qpos_retarget.py`
兩具**拓撲完全相同**的 MJCF 之間的直接 qpos 空間 retargeting（同樣的 body 名稱、joint 型別/軸/宣告順序，只有 `body_pos` 不同）。

因為拓撲相同，非根部的每個 qpos 值本身就是相對於自己 parent 的關節角，**原封不動複製過去就重現同樣的關節姿態**，跟節段長度無關。只有 free joint 的根部（`qpos[0:3]`，骨盆世界座標）要調——那是為**來源**骨架比例錄下的絕對平移。做法是按兩具骨架各自 rest pose 的骨盆高度比等比縮放。

它自己承認的限制：單一純量的高度縮放**只在站姿附近才正確**。對 crawl / headstand / lieonground / split / crouch 這類非站立片段，「rest 時的根部高度」與「這個姿勢下的根部高度」的比值在非等比縮放的骨架之間不一樣。實測 540 段：只做 naive 縮放，**83.7% 的片段平均最低點低於 −0.01 m**（成人自己是 +0.0065 m）。這就是 `retarget.py` 把逐幀地面修正換成可學 `dz_root` 的原因。

### 🟡 `qpos_retarget_ik.py`
在上面之上加一層**鎖腳 IK**，專治 footskate（不是提升一般 retargeting 精度）。

問題很具體：同樣的關節角配上不同縮放的根部軌跡，會讓腳在**應該踩住不動**的支撐相滑動。實測 `move-ego--90-2`，naive retarget 讓支撐相的每幀腳滑量變 5 倍（0.0039 → 0.0191 m/frame）。

做法：從**來源**動作偵測支撐相（低高度 + 低水平速度）→ naive retarget → 把該腳的世界座標鎖在區間第一幀落點 → 對那條腿的 9 個 hip/knee/ankle DOF 解 damped-least-squares IK 拉回來 → 區間邊界做幾幀 crossfade 避免跳動。擺動相保持 naive 不動。

---

## 4. 可行性檢查（不需要訓練，純檢驗參考動作）

### 🟢 `audit_origin_motion_feasibility.py` — 這一類裡唯一現役的
用**上層自己的 `C` 項**（`model/bilevel/semantics.py:reference_feasibility`）稽核 `data/origin_motion/` 裡的每一段片段。它問的問題跟 humenv 的 reward 不同、而且對這個專案更根本：

```
humenv reward:  「機器人這個任務做得好不好？」        （需要模擬）
上層的 C     :  「這個參考對這具身體來說到底問不問得出口？」（純運動學，不模擬、不用策略、不用獎勵模型）
```

在 **p = 0**（naive retarget）下評估，所以每個數字都是「來源動作 + 目標骨架」的固有性質，中間沒有任何學到的東西。這使它成為**上層必須超越的 baseline**。`adult` 那一列是對照組：來源身體的 retarget 是恆等映射，所以那裡的 C 是地板值，不是 retargeting 造成的傷害。

除了 C 的五項，也報 C 本身不報的可解讀指標：至少有一個非法關節的幀比例（那個 95.8% 的頭條數字）、以**弧度**而非 ctrl 單位表示的違規幅度、逐關節違規率、以 mm 表示的穿透深度。次要區塊報 `S`：在 p=0 時 `s_pose` 恆為 0、`s_heading ~1e-32`（角度和四元數是逐字複製的），但 `s_reach`/`s_contact`/`s_froude` 不是零——它們量的是**單純換身體**在任何修正之前就造成的動作扭曲。

輸出 → `outputs/qpos_audit/{per_clip.csv, per_joint.csv, summary.json}`。

### 🟢 `torque_capability_check.py`
測一具身體的致動器設計本身能不能站——**完全不牽涉**凍結的 Metamotivo actor、任何 z0、任何人類動捕參考。

- `--mode static`：在 rest pose 用 `mj_inverse` 算抗重力所需的廣義力，比對 forcerange。純靜力學，沒有控制器可以當藉口。
- `--mode stand`：真的 `mj_step`，用重力補償 + PD 控制器試著把身體維持在自己的 rest pose。

> 這支的註解記錄了一個**踩過的坑**：`--mode stand` 一開始用 `mj_inverse` 配上猜測的浮動基底目標加速度，那是**錯的**——強迫未致動的 free joint 加速度為 0，解出來的「所需力矩」不對應任何實際可達的力。改用 `qfrc_bias`（只問「現在這一刻抵消重力/科氏力要多少」）就避開了，因為那個量與 free joint 無關、恆有良好定義。

---

## 5. 物理追蹤實驗（三代路線的實驗紀錄）

### 🟡 `infer_z_from_qpos.py`
從一段 qpos 反推 `z`（`tracking_inference` / `goal_inference`），可選擇跟已知的來源 `z0` 比對。qpos → obs 是靠把 qpos 灌進 HumEnv 實例（`set_physics`）再讀 `get_obs()["proprio"]`。

有用的地方是它可以直接比「同一具身體」與「retarget 到別具身體」的推論退化程度。輸出逐步 `z`、與 z0 的 cosine 曲線圖與 CSV。

---

## 6. 視覺化

### 🟢 `rollout_video.py` — 現在看訓練結果就是用這支
拿一個 checkpoint、一段指定片段、一具身體，跑完整段長 rollout，輸出左邊參考、右邊實體機器人的對照 mp4 + 逐步追蹤數字。

兩個關鍵設計：

- **共用 `model/bilevel/longeval.rollout_one`** —— 和訓練迴圈每 100 iteration 算 `long/` 指標用的是**同一個函式**，所以影片下面印的數字和 W&B 曲線上的數字不可能漂開。
- **兩個刻意跟訓練不同的地方**，都是為了讓影片呈現「真正會交付的東西」：
  - `--wrench 0`（預設關）—— 外部根部力矩是訓練用的拐杖（proposal R5）。誠實的影片是不開外力的那支。
  - **整段片段** —— 訓練只看 0.8 s 的 window。局部好、全局不連貫的動作在那個長度看不出來，在這裡一目了然。

### 🟢 `render_qpos_playback.py`
純運動學播放（`mj_forward`，不碰物理）把 qpos npz 渲成 mp4，可以同時渲第二個 qpos 並排比較。其他腳本會 `import render_qpos` 重用它。

### 🟢 `compare_videos.py`
把兩支**已經渲好**的 mp4 並排或上下疊起來。跟上一支的差別是它不重新渲染，只讀影格拼接。

---

## 7. 廢棄

### `build_dataset.py`
bilevel 路線**完全不用**這支。`model/bilevel/data.py` 直接讀 `data/origin_motion/`、`data/z/` 和 `assets/robots_calib/*/parameter.json`：沒有 manifest、沒有 build 步驟、磁碟上也沒有那份 78 MB 的重複資料。

它的兩個前提都已經消失：

1. `retargeted_motion` 這個概念**已被移除**（retargeting 現在是 p 的 runtime 可微函數），它要複製的檔案和寫進 manifest 的 key 都不存在了
2. `MORPHOLOGY_SRC`（`:44`）指向 `assets/robots/robot_child_parameter.json`，那個路徑在 9495329 的資產重構裡被移除，所以現在跑會直接在那裡崩

留著只是為了能重建 legacy 的 `model/train.py` baseline。

---

## 附錄：常見任務對照

| 我想做什麼 | 跑哪支 |
|---|---|
| 生一批新的來源動作 | `metamotivo_motion_rollout.py --tasks-file docs/humenv_all_tasks_official.txt` |
| 加一具新身體 | `scale_robot.py` → `audit_bodies.py` → `calibrate_actuators.py` → `write_body_splits.py` |
| 確認某具身體撐不撐得住參考動作 | `audit_bodies.py`（快、準靜態）或 `torque_capability_check.py --mode static` |
| 知道 p=0 的 baseline 有多差 | `audit_origin_motion_feasibility.py` |
| 看訓練出來的策略實際表現 | `rollout_video.py --ckpt ... --task ... --body ...` |
| 改了 `config.py` 的 train/test 切分 | `write_body_splits.py`（不跑的話 `data.py:load_body` 會 assert 失敗） |
| 重建舊的 `data/retargeted_motion/` | `qpos_retarget.py`（或加鎖腳 IK 的 `qpos_retarget_ik.py`） |
