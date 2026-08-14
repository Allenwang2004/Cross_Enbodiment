# Cross-Embodiment Bilevel Training — 整體設計

> 本文件依 `new.md` 的兩階段計畫，對現有 code base 提出完整重寫設計。
> 所有數字都可追溯：註明 `file:line` 或本次實測命令。
>
> **第一次看這個專案？先讀 §0。** 後面的章節預設你已經知道 §0 裡的名詞。

---

## 0. 給第一次看的人

### 0.1 這個專案在解什麼問題

有一個現成的 AI 叫 **Meta Motivo**，它能操控一個模擬的人形機器人做各種動作（走路、跳、爬、倒立）。
它很強，但有個限制：**它只在「一種身材」上訓練過** —— 一個 71.8 公斤、身高比例標準的成人骨架。

把同一個 AI 原封不動裝到別的身材上（38 公斤的小孩、110 公斤的巨人），它會跌倒。
不是因為它笨，而是因為它學到的控制策略是針對那一具身體的比例與力量算出來的。

> **這個專案的目標：讓同一個 AI 在各種身材上都能動，而且不重新訓練那個 AI。**

不重新訓練是硬性條件 —— 那個模型很貴、很難重現，而且我們要的是「適配」而不是「重做」。
所以我們在它外面加幾個小網路去修正它的輸入與輸出，原模型整個凍結。

### 0.2 三個角色

整個系統只有三個東西在互動，先認清楚它們：

| 角色 | 是什麼 | 動不動 |
|---|---|---|
| **凍結的 AI**（Meta Motivo） | 吃「機器人現在的狀態」+ 一個 256 維的「我要做什麼」向量 `z`，吐出 69 個關節的控制訊號 | **完全不動** |
| **參考動作**（reference motion） | 「這一秒該擺什麼姿勢」的逐幀腳本。從成人身上錄下來的 540 段動作 | 由**上層**修正 |
| **機器人**（13 種身材之一） | MuJoCo 裡的物理模擬體，跟成人拓樸完全相同、只是肢段長短粗細不同 | 由**下層**驅動 |

### 0.3 什麼是 retargeting，為什麼它是核心

540 段動作是在**成人**身上錄的。要拿它當小孩的目標，得先「換算」到小孩身上 —— 這個換算就叫 **retargeting**。

原本的做法只有**一個數字**：把根部（骨盆）的世界座標乘上身高比。關節角度直接照抄。

這很粗糙，而且我們量出了一個關鍵事實：

> **95.8% 的參考影格，至少有一個關節角度超出目標身體的物理極限。**

也就是說參考動作不只是「不精準」，而是**根本做不到**。MuJoCo 會默默把超限的角度夾住，
所以無論下層怎麼訓練，追蹤誤差都有一個永遠消不掉的地板。

> ⚠️ **後續實測把這條的份量調降了（2026-08-10）。** 95.8% 是「至少一個關節越界」的
> 影格比例，聽起來很嚇人，但**越界的幅度**才是重點：實際只有 **6.5% 的 (影格, 關節)**
> 越界，而且平均只超出 **0.6°**。夾住這種幅度對追蹤誤差幾乎沒有影響。
> 上層真正要修的不是關節限位，而是**參考動作陷進地板裡**（p=0 時 88% 的影格
> 穿地 12–16 mm，見 §8.2 Stage 1 的記錄）。這段留著是因為它仍然是上層存在的
> 理由之一，但它**不是**最主要的那個。

**這就是為什麼需要「上層」** —— 讓參考動作本身也變成可以學習調整的東西。

### 0.4 為什麼要分兩層

```
上層（Upper）：修正參考動作，讓它在這具身體上真的做得到
     ↓ 提供目標
下層（Lower）：用強化學習訓練機器人去追上那個參考動作
```

**下層**是標準的強化學習：機器人動一步、拿到獎勵（追得準嗎？姿勢自然嗎？還站著嗎？）、調整策略。

**上層**問的是不一樣的問題：「如果機器人怎麼練都追不上，是不是參考動作本身有問題？」

### 0.5 兩層互相打架 —— 本文件最大的篇幅在處理這件事

這個設計有一個**天生的陷阱**，而且它很致命：

> 上層的目標是「縮小機器人實際做出來的動作 與 參考動作 之間的差距」。
> 但上層**同時有權修改參考動作**。
> 那麼縮小差距最省力的方法，不是讓機器人變好，而是 —— **把參考動作改成機器人本來就在做的樣子。**

這叫**退化（degeneracy）**。退化之後 loss 曲線會漂亮地往下掉，但機器人什麼也沒學會，
而且原本的動作語意（走路變成走路、跳變成跳）全毀了。

本文件 §3 整節都在防這件事，用三層防禦：

1. **結構上讓它做不到** —— 上層只能輸出 36 個全域修正數字，而且每個都被 `tanh` 硬夾在小範圍內。
   36 個數字要同時蓋住 540 段動作 × 10 種身材，容量上不可能把它們全變成站姿。
2. **目標函數上讓它不划算** —— 額外加一項「參考動作不准偏離原始動作太多」的懲罰，
   權重比例刻意設成 3:1，代表參考動作**最多只能往機器人漂移 25%**。
3. **診斷上讓它藏不住** —— 同時記錄「對修正後參考的誤差」和「對原始參考的誤差」。
   只有前者降 = 在作弊。

### 0.6 名詞與符號對照表

後面章節大量使用這些符號與名詞，**全部集中在這一節**，後面不會再重複解釋。
先看前兩張（符號、下層三個網路），其餘五張當字典查即可。

| 符號 | 唸法／全名 | 是什麼 |
|---|---|---|
| **p** | p | **上層**的參數。一個小網路的權重，決定參考動作怎麼被修正 |
| **φ** | phi | **下層**的參數。三個小網路（見下）的權重，決定機器人怎麼動 |
| **β** | beta | **身材描述向量**，8 個數字：腿/手/軀幹/頭 的「長度倍率」與「粗細倍率」 |
| **z** | — | 256 維的「意圖向量」。凍結 AI 吃這個決定要做什麼動作。每段動作有一個 `z0` |
| **z_β** | z-beta | 經過調整、餵給凍結 AI 的 `z`。`z_β = z0 + 小修正` |
| **u** | — | p 網路吐出的 36 個原始數字，經過硬 tanh box 後變成實際的 retarget 參數 |
| **τ** | tau | 機器人實際跑出來的軌跡（trajectory） |
| **G** | Gap | 機器人實際動作 **vs** 修正後參考動作 的差距。上層想縮小它 |
| **S** | Semantic fidelity | 修正後參考 **vs** 原始動作 的偏離。防退化的主力，上層想縮小它 |
| **C** | Feasibility Cost | 參考動作在這具身體上**合不合法**（關節超限、穿地、腳滑）。上層想縮小它 |
| **H** | Horizon | 一個訓練片段的長度 = **24 步**（0.8 秒 @ 30 Hz） |
| **RSI** | Reference State Initialization | 每個片段開始時，直接把機器人擺成參考動作的姿勢再開始跑 |
| **TTSA** | Two-Timescale Stochastic Approximation | 「兩層用不同速度更新」的做法：下層每輪都更新，上層每 10 輪才更新一次 |
| **ES** | Evolution Strategy | 一種不需要梯度的估計法。用在「模擬器不可微分、算不出梯度」的那部分 |
| **PPO** | Proximal Policy Optimization | 下層用的強化學習演算法 |
| **frozen actor** | — | 凍結的 Meta Motivo 本體 |

**下層的三個小網路（合稱 φ）**：

| 網路 | 做什麼 |
|---|---|
| `LatentAdapter` | 根據身材 β 調整意圖向量 `z`，等於告訴凍結 AI「你現在在一個不一樣的身體裡」 |
| `ActionHead` | 微調凍結 AI 吐出的 69 個關節指令 |
| `RootWrenchHead` | 額外對機器人腰部施加一個「隱形的手」扶住它。**這是訓練用的拐杖，最後會被退火到零** |

---

以下五張表是**查得到就好**的詞彙表，不需要按順序讀。後面章節出現看不懂的名詞就回來這裡查。

#### (a) 強化學習詞彙

| 名詞 | 白話解釋 |
|---|---|
| **policy（策略）** | 「看到什麼狀態就做什麼動作」的函數。這裡就是 φ 那三個網路加上凍結 AI |
| **rollout** | 讓策略實際去跑一段，把每一步的狀態、動作、獎勵記下來 |
| **on-policy** | 只能用「當前這版策略自己跑出來的資料」來更新。PPO 是 on-policy，所以每輪都要重新模擬 |
| **reward（獎勵）** | 每一步給的分數。這裡是 `0.65·追蹤 + 0.15·正則 + 0.20·存活`，範圍 [0,1] |
| **return（回報）** | 從某一步開始，未來所有獎勵的折扣總和。策略想最大化它 |
| **discount γ（折扣因子）** | 未來的獎勵打幾折。γ=0.97 表示 33 步之後的獎勵大約只值現在的三分之一 |
| **value function `V`** | 「從這個狀態開始，預期能拿到多少 return」的估計。用一個網路學 |
| **advantage `A`** | 「這個動作比平均好多少」＝ q(s,a) − v(s)。**PPO 用它決定要加強還是抑制某個動作** |
| **GAE** | Generalized Advantage Estimation。算 advantage 的一種折衷：完全用實際結果太吵，完全信 value 太偏，GAE 用 λ=0.95 在兩者之間插值 |
| **bootstrap** | 片段被切斷時，用 `V(最後一個狀態)` 補上「後面還沒發生的部分」，而不是當成 0 |
| **truncation vs termination** | **截斷**是「24 步到了，但故事還沒完」→ 要 bootstrap；**終止**是「跌倒了，故事真的結束」→ 補 0。混淆這兩個會教壞策略，讓它以為活到窗尾毫無價值 |
| **log-prob** | 策略選到某個動作的機率取對數。PPO 比較新舊策略的 log-prob 差來衡量「這一步走多遠」 |
| **ratio / clip / clipfrac** | `ratio = 新策略機率 / 舊策略機率`。PPO 把它夾在 [0.8, 1.2]，`clipfrac` 是被夾住的樣本比例。**超過 0.3 表示步伐太大** |
| **KL** | 新舊策略的分布差異。健康的 PPO 每輪只有 0.01–0.05；本專案曾一度到 94（見 §11） |
| **epoch / minibatch** | 同一批資料重複用 4 遍（epoch），每遍切成 4 小塊（minibatch）分別更新 |
| **entropy** | 策略的隨機程度。這裡設 0，因為凍結 AI 已經提供了行為先驗 |
| **REINFORCE** | 最原始的策略梯度法（舊系統 `model/train.py` 用的）。一整段只給一個純量訊號，訊噪比極差 |
| **score function** | `∇log π(a|s)`，策略梯度的核心項。它的變異隨動作維度成長，這是舊系統的病根 |
| **BC / behaviour cloning** | 行為複製。直接用監督式學習叫網路輸出「正確答案」。這裡的正確答案是 `a_ref` —— 能命令出參考關節角的那個 ctrl，有封閉解、零變異 |
| **MDP** | Markov Decision Process，強化學習的標準問題形式：狀態、動作、轉移、獎勵 |
| **PopArt** | 把 value 的目標值做running 正規化，避免尺度漂移讓 value 網路學不動 |

#### (b) 雙層最佳化詞彙

| 名詞 | 白話解釋 |
|---|---|
| **bilevel（雙層）** | 一個最佳化問題的答案，是另一個最佳化問題的輸入。這裡：上層決定參考動作，下層在那個參考底下練策略 |
| **TTSA** | Two-Timescale Stochastic Approximation。**兩層用不同速度更新**，讓下層看起來「已經收斂」，上層才有穩定的東西可以最佳化。這裡是 30:1 的學習率比 × 10:1 的更新頻率 = 300:1 |
| **hypergradient** | 「上層目標對上層參數的梯度」。因為下層的最佳解也會跟著上層變，所以它比一般梯度多出好幾項 |
| **T1 / T2 / T3** | hypergradient 拆成三項（§4.3）。**T1** = 參考直接影響目標（autograd 精確算得出，保留）；**T2** = 參考影響模擬結果（模擬器不可微，用 ES 估）；**T3** = 參考影響「策略最終會學成什麼樣」（需要二階導數，捨棄） |
| **Hessian-inverse-vector product** | T3 需要的東西：二階導數矩陣的反矩陣乘上一個向量。在有接觸不連續的物理模擬上算不出來 |
| **Gauss-Seidel 交替** | 捨棄 T3 之後實際在做的事：固定 φ 優化 p、固定 p 優化 φ，來回輪流。**它的解跟真正的 bilevel 解不同**，§4.3 有誠實說明代價 |
| **Borkar 條件** | TTSA 理論上收斂需要的學習率條件。用 Adam 時不嚴格成立，本文件只把它當「比例該取多少」的指引 |
| **ES / Evolution Strategy** | 演化策略。不用梯度：在參數上加隨機擾動，看目標變好還是變壞，反推該往哪走 |
| **antithetic（反對稱）** | ES 的標準技巧：`+ε` 和 `−ε` 成對測試。共同的噪音互相抵消，估計變異大幅下降 |
| **CRN / Common Random Numbers** | 成對的兩次模擬，**除了那個 ±ε 之外所有隨機數都逐位元相同**（同一段動作、同一個起點、同一串動作噪音）。ES 訊號只有約 10%，沒有 CRN 就會被模擬噪音完全淹沒。**這是必要條件不是優化** |
| **rank normalization** | 把 40 個擾動的結果換成名次再加權，讓估計不受目標函數尺度漂移影響 |
| **proximal 項** | `‖u − u_prev‖²`，懲罰「這一步走太遠」，讓上層平順 |
| **trust region（信賴域）** | 硬性上限：每次上層更新 `‖Δu‖_∞ ≤ 0.02`。跟 proximal 的差別是它**夾死**而不是懲罰 |
| **hard tanh box（硬 box）** | 把 p 的輸出過 `tanh` 再乘一個固定寬度，讓每個 retarget 參數**在結構上不可能**超出範圍。防退化的第一道防線，不靠調權重 |
| **saturation（飽和）** | `|tanh(u)| > 0.9` 的比例。表示 p 貼在 box 的牆上、想出去但出不去 —— **超過 0.2 就是退化警報** |
| **box-corner normalization** | 每個 S/C 分項的單位，取「p 走到 box 角落時該項的值」。沒有這一步，各項的原始尺度差到 10¹¹ 倍，權重完全沒有意義（§11 偏差 B） |
| **degeneracy（退化）** | 上層作弊的方式：不去改善機器人，而是把參考動作搬到機器人身上，差距一樣會變小。整份文件最大篇幅在防這件事 |

#### (c) MuJoCo 與物理

| 名詞 | 白話解釋 |
|---|---|
| **MuJoCo** | 物理模擬器。整個專案的模擬都在它上面跑 |
| **qpos / qvel / ctrl** | 廣義座標（76 維：根部 7 + 關節 69）／廣義速度（75 維）／致動器指令（69 維） |
| **nq / nv / nu** | 上面三者的維度。**`nv=75` 但 `nu=69`**，差的 6 個就是「根部沒有馬達」—— 這是很多問題的根源 |
| **free joint（自由關節）** | 骨盆與世界之間的連接，6 個自由度（3 平移 + 3 旋轉），用 7 個數字表示（位置 3 + 四元數 4）。**沒有致動器** |
| **hinge（鉸鏈關節）** | 一般的單軸關節，69 個，每個都有馬達 |
| **jnt_range** | 每個關節的角度上下限。MuJoCo 會**靜默夾住**超限的指令 |
| **actuator（致動器／馬達）** | 這裡是 affine 位置伺服：`力 = gainprm[0]·ctrl + biasprm[0] + biasprm[1]·角度 + biasprm[2]·角速度`。解出來 `ctrl ∈ [−1,1] ⟺ 角度 ∈ jnt_range` **精確成立**（誤差 4.4e-16），這就是 §1.3 那個關鍵發現 |
| **gainprm / biasprm / forcerange** | 致動器的增益、偏置、力上限。縮放身體時必須跟著改，否則小孩用成人的馬達（§8.1 R1） |
| **armature / damping / stiffness** | 關節的等效轉動慣量／阻尼／被動彈簧。**放大馬達增益時 armature 和 damping 必須同倍放大**，否則顯式積分器會發散（§11 偏差 A） |
| **geom / contact** | 碰撞幾何／實際接觸點。腳有沒有踩到地是讀 `data.contact`，不是用高度猜的 |
| **xfrc_applied** | 直接施加在某個 body 上的外力與力矩。`RootWrenchHead` 就寫這裡 |
| **mj_step / mj_step1 / mj_forward / mj_kinematics** | 走一個物理步／只走前半步（更新 sensor）／算完整前向動力學／只算位置學。`mj_step(nstep=15)` **之後必須再 `mj_step1`**，否則 observation 讀到的 sensor 是舊的 |
| **action_repeat / physics_dt** | 一個控制步 = 15 個物理步；物理步長 1/450 秒 → 控制頻率 **30 Hz** |
| **warning** | MuJoCo 的數值發散警告。worker 必須把它當成「這條軌跡死了」處理，**絕不能讓 worker 自己崩潰** |

#### (d) 幾何與運動學

| 名詞 | 白話解釋 |
|---|---|
| **FK / forward kinematics** | 正向運動學：從關節角度算出每個身體部位在世界中的位置。本專案用 torch 重寫了一份**可微分**的版本 |
| **quaternion（四元數）** | 用 4 個數字表示 3D 旋轉，避免萬向鎖。根部朝向用它 |
| **up vector / `up_z`** | 身體「向上軸」在世界 z 方向的分量。1 = 完全直立，−1 = 完全倒立。**這個資產的公式是 `2(q_y q_z + q_w q_x)`，不是教科書的那條**（骨盆帶 `euler="90 0 0"`）|
| **exp-map** | 用一個 3 維向量表示旋轉（方向 = 轉軸，長度 = 角度）。上層的根部姿態修正用這個參數化 |
| **root / Pelvis** | 骨盆，運動鏈的根。文中「root」一律指它 |
| **EE / end effector** | 末端：雙手、雙腳、頭 |
| **COM / center of mass** | 質心。用來判斷平衡 |
| **Froude number** | 無因次的步態速度指標 `v²/(g·L)`。用它比較不同體型的動作才公平 —— 小孩走得慢不是「做錯」 |
| **manifold / `project_z`** | 凍結 AI 只認得長度剛好 16 的 `z` 向量（球面）。`project_z` 把 `z` 投影回那個球面上，少了這一步等於餵它沒見過的輸入 |
| **SMPL** | 人體參數化模型的標準。這批骨架和動作都是從它來的 |

#### (e) 系統與外部元件

| 名詞 | 白話解釋 |
|---|---|
| **Meta Motivo / FB-CPR** | 那個凍結不動的預訓練 AI。輸入 358 維觀測 + 256 維意圖 `z`，輸出 69 維動作 |
| **humenv** | Meta Motivo 附帶的環境套件。本專案**只借用它的 observation 建構函數**，不用它的環境迴圈（它的 `is_terminated()` 永遠回傳 False）|
| **worker / SimPool** | 32 個獨立進程，每個跑 8 個模擬，共 256 個。繞過 gymnasium 自己寫的，為了吞吐量 |
| **shared memory（共享記憶體）** | 主進程與 worker 之間傳資料的方式，全程無 pickle 序列化 |
| **spin-then-yield** | worker 等待新指令的方式：先空轉約 2000 次，再讓出 CPU。實測比 `multiprocessing.Barrier` 快 40% |
| **EGL** | 無螢幕環境下的 OpenGL 後端。算圖（錄影）時要 `MUJOCO_GL=egl` |
| **W&B / Weights & Biases** | 訓練指標的線上儀表板。`--wandb` 開啟，`--wandb-group` 把三個 stage 綁成一個實驗 |

### 0.7 訓練時會看到的指標

跑訓練時終端會刷、`outputs/bilevel_logs/stage{N}_metrics.jsonl` 會寫、W&B 會畫的就是這些。
分成四個面板，前綴決定它畫在哪一張圖。

#### `lower/` — 下層在學得怎麼樣

| 指標 | 含義 | 健康值 |
|---|---|---|
| `reward` | 每步總獎勵 `0.65·追蹤 + 0.15·正則 + 0.20·存活`，有界 [0,1] | 越高越好 |
| `r_track` | 追蹤獎勵，五個通道（關節角／速度／末端／根部／質心）的加權和 | **Stage 1 的門檻是 > 0.6** |
| `r_pose` | `exp(−2 × 關節角均方誤差)`。1 = 完美貼合 | 越高越好 |
| `r_ee` | `exp(−40 × 末端誤差)`。五個末端（雙手、雙腳、頭）相對骨盆的位置 | 越高越好 |
| `pose_err` | 關節角均方誤差的**原始值**（rad²），沒過指數核 | **> 0.6 會觸發追蹤失敗終止** |
| `term_rate` | 這一輪 256 個片段中提前終止（跌倒／追丟）的比例 | **Stage 1 的門檻是 < 0.3**（見 §0.9 Stage 1）|
| `mean_steps` | 平均活了幾步（滿分 24） | 越接近 24 越好 |
| `e_ext` | 外力用了多大（已正規化到體重）。**這是拐杖的用量** | 應隨退火趨近 0 |
| `pg` | PPO 的 policy gradient loss | 震盪正常，不是進度指標 |
| `v` | value function 的 loss | 應下降並穩定；**上層更新後暴增 = 警訊** |
| `z` | `1 − cos(z_β, z0)`，adapter 把意圖向量拉離原點多遠 | 小而穩定 |
| `bc` | behaviour cloning 輔助項，前 2000 iter 退火掉 | 隨 `sched/lambda_bc` 歸零 |
| `kl` | 新舊策略的 KL 散度 | 暴增 = 步子邁太大 |
| `clipfrac` | 被 PPO clip 截掉的樣本比例 | 0.05–0.3 之間算正常 |

#### `upper/` — 上層在做什麼、有沒有作弊

| 指標 | 含義 | 健康值 |
|---|---|---|
| `F` | 上層總目標 = `λ_gap·G + λ_fid·S + λ_feas·C + λ_ext·E_ext + λ_phys·P + λ_prox·prox` | 下降 |
| `G` | **Gap** —— 機器人實際做出來的 vs 修正後參考 的差距 | 下降 |
| `S` | **語意保真** —— 修正後參考 偏離 原始動作 多少 | **Stage 2 門檻：< 2** |
| `C` | **可行性** —— 參考在這具身體上合不合法（關節超限／穿地／腳滑） | 下降 |
| `prox` | `‖u − u_prev‖²`，上層這一步走了多遠 | 小 |
| `ref_min_z` | **上層的頭條成績**。參考動作最低幾何點的高度（負值 = 陷進地板）| **≥ −0.005 m** |
| `frac_illegal` | 參考影格中有關節超出物理極限的比例。**只是觀察用**，見 §0.3 修正註記 | 下降即可 |
| `u_saturation` | **退化警報**。p 的輸出貼在硬 box 牆上的比例 | **> 0.2 就是危險訊號** |
| `u_absmax` | p 輸出的最大絕對值 | 觀察用 |
| `u_step_linf` | 上層更新後 `‖Δu‖_∞`（只在上層更新的那輪出現） | ≤ 0.02（trust region）|
| `u_step_clipped` | 這步有沒有被 trust region 夾住 | 偶爾 1 可接受 |
| `es_grad_norm` / `es_delta` | ES 估計的梯度大小／反對稱差分大小（**Stage 3 才有**） | 非零 = ES 有訊號 |

> **`u_saturation` 和 `ref_min_z` 是最該盯的兩個。**
> 前者 > 0.2 表示目標函數在把 p 往退化推（§3.5）；
> 後者是「上層到底有沒有解決它該解決的問題」的唯一直接答案。

#### `upper_terms/` — 上層把預算花在哪一項

S、C、G 的逐項拆解。全部已按「p 在硬 box 邊界能達到的值」正規化，所以都落在 [0,1]，
可以直接互相比較 —— 誰大就是誰在吃掉上層的注意力。

| 前綴 | 通道 |
|---|---|
| `s_*` | `s_pose` 關節角偏離 · `s_reach` 無因次末端伸展 · **`s_contact` 腳的接觸時序（防退化主力）** · `s_heading` 轉向速率 · `s_froude` 無因次步態速度 |
| `c_*` | **`c_limit` 關節超限（編碼 95.8% 那個發現）** · `c_penetrate` 穿地 · `c_float` 該著地卻飄浮 · `c_slide` 接觸中腳滑 · `c_smooth` 參考動作平滑度 |
| `g_*` | `g_pose` 關節角 · `g_ee` 末端位置 · `g_root_pos` 根部位置 · `g_root_rot` 根部姿態 |

> 實測 p=0 時 `c_smooth` 已經在 0.94、`s_froude` 在 0.47 —— 代表 p 對這兩項幾乎沒有施力空間。
> 反過來 `c_limit` 在 0.006、`s_contact` 在 0.069，那才是 p 真正能動的地方。

#### `sched/` — 退火排程（確認拐杖有被拿掉）

| 指標 | 含義 |
|---|---|
| `wrench_scale` | 外力強度倍率。iter 0–2000 維持 1.0，之後 cosine 退火，8000 歸零 |
| `lambda_bc` | behaviour cloning 權重。0–2000 cosine 退火到 0 |

這兩條必須**確實走到 0**。`wrench_scale` 沒歸零就代表評估時機器人還在被隱形的手扶著。

#### 頂層

`stage`（1/2/3）、`iter`、`elapsed`（秒）。

---

### 0.8 幾個支撐整份文件的數字

| 數字 | 意思 |
|---|---|
| **95.8%** | 參考影格中至少有一個關節超出物理極限的比例。上層存在的理由 |
| **36** | 上層能調的參數個數。刻意設這麼少，是防退化的第一道防線 |
| **25%** | 參考動作最多被允許往機器人漂移的比例（由 `λ_fid : λ_gap = 3 : 1` 決定） |
| **24 步** | 一個訓練片段的長度，0.8 秒 |
| **256 × 24** | 每輪同時跑 256 個片段 × 24 步 = 6144 筆資料 |
| **9 + 2 + 1** | 9 具身材訓練、2 具留著測試泛化、1 具（`adult`）是 source —— 所有 retarget 的基準，不參與訓練 |
| **8 → 12** | 出廠資產只有 8 具站得起來；校準 actuator 後 12 具都可用（§11 偏差 A）|

### 0.9 每個 Stage 在做什麼、為什麼要這樣切

訓練不是一次跑到底，而是分成三個階段接力（第四個是第三個的全量版）。
實作在 `train_bilevel.py:apply_stage`。

#### 貫穿全部的一條原則

> **每個 Stage 只新增「一個」新的難度來源。**

這樣一旦壞掉，你知道是什麼弄壞的。如果一次把身材多樣性、會動的參考、退火中的外力、
ES 雜訊全部打開，訓練崩了你查不出是哪一個造成的 —— 而每次重跑要好幾個小時。

#### 逐階段

**Stage 1 —— 只練下層，把其他變因全部釘死**

| 這階段 | 設定 |
|---|---|
| 身材 | **只有 2 具**（`child`、`teen`），刻意覆寫掉 9 具的設定 |
| 參考動作 p | **永久凍結在 0**（`upper_warmup_iters = 10⁹`），就是 naive retarget |
| 外力 | **常開不退火**（`wrench_hold_iters = 10⁹`）|
| ES | 關 |

**為什麼**：這一階段在回答一個最基本的問題 —— **下層到底追不追得動？**
所以參考動作必須是靜止不動的靶（p 凍結），外力這根拐杖必須一直在（不退火），
身材只留兩具體型相近的（`child` 0.585 m / `teen` 0.799 m）。
只用 2 具是為了**快**：迭代快，失敗也便宜。

**門檻**：2000 iter 內 `r_track > 0.6` 且 `term_rate < 0.3`。

> 原本是 `term_rate < 0.05`。2026-08-10 改成 0.3：實測用**精確 `a_ref` 開迴路**
> 驅動（策略能拿到的最好動作），150 個 window 裡仍有 26–31% 提前終止。也就是
> 0.05 這個數字在參考動作本身就達不到。同時把 `term_root_dist` 從 0.5 提到 0.75
> —— 根部沒有致動器，參考的根部軌跡來自成人落腳點按身高縮放，關節追得再準根部
> 還是會漂。
>
> 門檻定在 **0.3**：實跑 2000 iteration 收在 `term_rate` 0.31–0.32、`r_track` 0.69，
> 而閉迴路歸因顯示剩下的終止是移動中段的真實跌倒（`root_h` 30%、`up_z` 14%，
> `root_dist` 與 `pose_err` 皆 0%），不是判定條件的假陽性。0.3 是「下層確實學會
> 站著追動作」的證據線，把它壓更低是 Stage 3（外力退火後）該做的事。

**過不了就停在這裡。** 如果機器人連「不會動的靶 + 有人扶著 + 相近體型」都追不上，
後面把靶弄成會動的、把手放開、再加 8 具更難的身材，只會更糟。

---

**Stage 2 —— 放進全部身材，並讓上層開始動**

| 相對 Stage 1 的變化 | 說明 |
|---|---|
| 身材 2 → **9 具** | 從 `child`/`teen` 擴到含 `athletic`(101 kg)、`long_limbed`(1.07 m) 等 |
| p **解凍** | 前 500 iter 仍凍結，之後每 10 iter 更新一次 |
| 外力 | **仍然常開** |
| ES | **仍然關** |

**為什麼外力還不退**：因為這階段已經同時引入兩個新東西（身材多樣性、會動的參考），
再把拐杖抽掉就是三個。外力留著，等下一階段單獨處理。

**為什麼 p 還要再等 500 iter**：`F(p, φ)` 的意思是「在**這個** φ 之下，參考動作該怎麼修」。
如果 φ 還是隨機的，量到的差距反映的是「策略還沒學好」而不是「參考動作有問題」，
上層會朝著錯誤方向修。等 φ 在新身材上先站穩，`F` 才有意義 —— 這也是
TTSA 理論裡「φ 接近 φ*(p)」那個假設的實務替代品（§4.2）。

**門檻**：`ref_min_z ≥ −0.005 m`（參考不再陷進地板）、`u_saturation < 0.2`、`S < 2`，
**而且 §3.5 的誠實性檢驗要過** —— `D` 對 p=0 參考也要降，不能只有對 p 參考降。

> 原本第一條是「`frac_illegal` 從 0.958 降到 < 0.2」。2026-08-10 改掉：那個比例
> 正確但誤導 —— 實際越界只有 6.5% 的 (影格, 關節)、平均 0.6°，夾掉這種幅度對追蹤
> 幾乎沒影響，把它壓到 0.2 是在追一個不重要的數字。上層真正在修的是**穿地**
> （p=0 時 88% 影格陷入地板 12–16 mm），所以門檻換成直接量那件事。
> `frac_illegal` 仍然記錄、仍該下降，只是不再當成通過條件。

---

**Stage 3 —— 抽掉拐杖，補上看不見的那半邊梯度**

| 相對 Stage 2 的變化 | 說明 |
|---|---|
| 外力 | **開始退火**：iter 0–2000 維持，2000–8000 cosine 降到 **0** |
| ES | **開啟** |

**為什麼外力要抽掉**：它從來就不是交付物的一部分。真實系統沒有一隻隱形的手扶著腰。
留著它訓練是因為早期的策略站都站不穩、拿不到任何有用的獎勵訊號；
一旦能追了，就必須讓它學會自己站。
**所有評估數字一律在 `f_max = 0` 下量**（`eval_bilevel.py` 強制），
否則等於在灌水。

**為什麼這時才開 ES**：這牽涉到 §4.3 的核心 —— 上層梯度有三項，
第一項（T1）可以用 autograd 精確算出來，但它**看不見外力作弊**：
`E_ext` 與 `P` 只透過模擬器依賴 p，而模擬器不可微分，所以 T1 對它們恆等於零。
純 T1 的上層，會樂於把參考動作改得更難，反正下層可以用外力硬撐、差距照樣很小。
ES 是用來估這塊看不見的梯度的，所以它必須和外力退火**同時**開啟 —— 早開沒有意義
（外力還是常數，沒有作弊空間可言），晚開則會在退火期間放任作弊。

**注意**：ES 依賴 **CRN（common random numbers）** —— 同一組擾動的正負兩邊必須共用
完全相同的隨機數（clip、window 起點、RSI 噪音、整條 action noise）。
訊號只有約 10% 的相對變化，沒有 CRN 就會被 rollout 雜訊完全淹沒，比不加還糟。

---

#### 為什麼是接力而不是續跑

跨 stage 用 `--init-from` 而不是 `--resume`，**只帶網路權重**：

- optimizer 的 Adam 動量是對**舊的目標函數**累積的，換階段後已經失效
- learning rate schedule 必須從 0 重新算
- iteration 計數不能延續，否則會直接跳過新階段的 warmup

同一階段內中斷才用 `--resume`，那才需要完整還原（動量、RNG、normalizer）。

---

### 0.10 這份文件怎麼讀

| 你是誰 | 建議順序 |
|---|---|
| **想知道這在幹嘛** | §0（本節）→ §2 架構圖 → §9 需求對應表 |
| **要評估設計對不對** | §1 現況問題 → §3 防退化（**最關鍵**）→ §4 兩層怎麼耦合 → §8 風險 |
| **要接手寫程式** | §7 檔案結構 → §5 下層 → §6 模擬 → §11 實作結果與偏差 |
| **要跑訓練** | §0.9（上一節）→ §8.2 分階段落地 → §11 已知偏差 → §12 資料現況 |

> **§11 很重要**：前面 §1–§10 是**設計時**寫的，§11 是**實作完**回填的實測結果，
> 有四處與原設計不同。兩邊衝突時**以 §11 為準**。

> **看不懂的名詞一律回 §0.6 查。** 那裡有六張表：符號（p/φ/β/G/S/C…）、下層三個網路、
> 強化學習詞彙、雙層最佳化詞彙、MuJoCo 與物理、幾何與運動學、系統與外部元件。
> 後面章節不會再重複解釋這些詞。

---

## 目錄

0. [給第一次看的人](#0-給第一次看的人)
1. [現況與問題陳述](#1-現況與問題陳述)
2. [整體架構與資料流](#2-整體架構與資料流)
3. [Upper level：目標函數與防退化](#3-upper-level目標函數與防退化)
4. [Bilevel 形式與 TTSA](#4-bilevel-形式與-ttsa)
5. [Lower level：PPO + 解析側通道](#5-lower-levelppo--解析側通道)
6. [SimPool：多進程模擬設計](#6-simpool多進程模擬設計)
7. [檔案結構與去留](#7-檔案結構與去留)
8. [風險與分階段落地](#8-風險與分階段落地)
9. [與 `new.md` 的對應表](#9-與-newmd-的對應表)
10. [開放問題](#10-開放問題)

---

## 1. 現況與問題陳述

### 1.1 現有系統的四個結構性問題

現行 pipeline (`model/train.py`) 是單階段的：

| 問題 | 證據 |
|---|---|
| **Retargeting 是離線烘焙的** | `scripts/qpos_retarget.py` 是 CLI，寫出 `.npz`；`model/dataset.py:57` 訓練時只做 `np.load(...)["qpos"]`。retargeting 品質完全不參與最佳化。 |
| **只有一個純量 scale 可調** | `scripts/qpos_retarget.py:91 retarget_qpos(qpos, scale)` 全部內容就是 `out[:, 0:3] *= scale`。唯一的 scale 來自 rest-pose pelvis 高度比 (`:172-174`)，adult→child = 0.6110。 |
| **beta 是常數，adapter 的條件輸入零資訊** | `scripts/build_dataset.py:45 MORPHOLOGY_LABEL = "child"` 寫死，`manifest.jsonl` 全部 1530 列都是 `child`。`LatentAdapter(beta_dim=8, ...)` 的 beta 從頭到尾是同一個向量。 |
| **65% 的訓練資料對主要 loss 沒有貢獻** | 1530 列中 **990 列** `"retargeted_motion": null` → `model/dataset.py:56` 回傳 `qpos_ref=None` → `model/losses.py:126-127 functional_equivalence` 靜默回傳 `(0.0, {})`。`model/train.py:286` 均勻取樣，所以每個 batch 約 2/3 對 D 項是死重。 |

另外三個較小但會咬人的問題：

- **Episode-level REINFORCE 的訊噪比**：`model/train.py:158` 把 300×69 = 20700 個抽樣維度的 log-prob 壓成一個純量，再乘上一個 episode 級的 advantage。`model/diagnose_single_task.py` 的存在本身就是在懷疑 task-composition 噪音主導了學習訊號。
- **`z_beta` 脫離 FB manifold**：所有存檔的 `z` norm 剛好 `16.0 = √256`，`metamotivo/fb/model.py:126 project_z` 強制此約束，但 `model/networks.py:36` 回傳未投影的 `z0 + α·delta`。frozen actor 從未見過離開球面的 z。
- **既有實驗結果本身在警告**：`outputs/{baseline,eval}/report.json`（11 個 held-out task × 10 trials）顯示 adapter 把 `D` 從 40.51 降到 21.40，但 `L_phys` 從 **2.75 惡化到 3.83** — adapter 在拿物理可行性換參考動作的相似度。

### 1.2 一個至關重要的新發現：參考動作是「非法」的，不只是不精準

實測 540 clips × 69 hinge joints（131,700 frames）對 `assets/robots/child/robot.xml` 的 `jnt_range`：

```
frames with >=1 out-of-limit joint: 126166/131700 = 95.8%
(frame,joint) pairs out of limit:   579670/9087300 = 6.38%
joints with range < 0.3 rad:        12 / 69
```

MuJoCo 對 `ctrllimited=true` 的 actuator 會**靜默 clamp**，所以 tracking loss 有一個永遠消不掉的下限。這與既有稽核吻合：`outputs/retarget_actuator_feasibility.csv` 顯示 child 有 91% 的 frame 動力學不可行（`frac_frames_infeasible` 平均 0.910，80% 的 clip 是 1.0）。

**這是 upper level 最強的存在理由。** 同時它讓「防退化」問題比表面上容易得多：p 有一大塊**真實的、可量測的**修正空間可以先吃掉，還遠遠碰不到動作語意。

### 1.3 第二個發現：action space 就是正規化後的關節角空間

Actuator 是 `<general biastype="affine">`，力 = `gainprm[0]·ctrl + biasprm[0] + biasprm[1]·q + biasprm[2]·q̇`。令力為零解出平衡角，實測：

```
max|q(ctrl=-1) - jnt_lo| = 2.78e-16
max|q(ctrl=+1) - jnt_hi| = 4.44e-16
```

也就是 `ctrl ∈ [-1,1] ⟺ qpos_j ∈ jnt_range_j` **精確成立**。因此

```
a_ref,t = clip( 2·(q̂_{t+1} − lo)/(hi − lo) − 1 , −1, 1 )        # (69,)
```

是一個**精確、零變異、免費**的 ActionHead 監督目標。第 5 節會把它當 warm-start 用。

### 1.4 為什麼新的 reward 設計沒有踩到舊的坑

commit `9600539 "feat: remove reward loss"` 拿掉了 `−R_task`，理由（`model/train.py:12-26`）是 humenv 的 reward 是**針對 source body 的運動學寫的**，在目標身體上最佳化它等於逼 adapter 回頭模仿原本的身材。

**這個理由仍然成立，而且新設計不受影響**：`new.md` 的 tracking / regularization / survival reward 全部都在**目標身體自身的 FK 與動力學**上計算，沒有任何一項引用 source body 的比例。commit message 當時劃的那條界線，正好把新 reward 放在安全的一邊。

---

## 2. 整體架構與資料流

```
(clip m, body β)
  │
  ├─ src_qpos (T,76) ──► RetargetNet p=f(β) ──► apply_retarget ─┬─► ref_raw   (可微，餵 feasibility penalty C)
  │                          ↑ 36 維全域參數                     └─► ref = clamp(ref_raw, jnt_range)
  │                          │                                        │
  │                     [硬 tanh box]                                 ├──► RSI 初始狀態 (+ 關節高斯噪音)
  │                                                                   └──► per-step tracking reward 的目標
  │
  └─ z0 (256,) ──► LatentAdapter ──► project_z ──► z_beta ──► frozen FB actor ──┬─► ActionHead ──► μ_ctrl (69)
                        ↑ β                                                     └─► RootWrenchHead ──► μ_wrench (6)
                                                                                        │
                                             a ~ N([μ_ctrl, μ_wrench], σ)  ──────────────┘
                                                              │
                                        SimPool (32 proc × 8 sims = 256 envs, 30 Hz)
                                                              │
                                                       τ = (qpos, qvel, obs, rew_terms, done)
                                                              │
                          ┌───────────────────────────────────┴──────────────────────────────┐
                          ▼                                                                   ▼
              Lower: PPO + GAE (更新 φ 每 iter)                        Upper: F(p) (更新 p 每 K=10 iter)
              φ = {LatentAdapter, ActionHead, RootWrenchHead}          p = RetargetNet 權重
```

**每個 iteration 的形狀**：256 windows × 24 steps = 6144 transitions。10000 iterations = 61.4M env steps。

**資料規模**：540 clips × 10 morphologies = **5400 (motion, body) pairs**，餘 3 個身材 held-out。

---

## 3. Upper level：目標函數與防退化

> 本節是整份文件最關鍵的部分。

### 3.1 退化問題的精確形狀

`new.md` 說「Upper 就是要衡量機器人實際做出來的狀態跟參數化的參考動作 gt 之間的差距」。直接寫成 `min_p E‖s_robot − ref_p‖²` 會退化，原因很具體：

> 能到達 p 的可微梯度**唯一**提供的方向，就是 `∂/∂ref ‖τ − ref‖² · ∂ref/∂p`，字面上是「把 ref 搬到 τ 身上」。

這不是邊角案例 — 這是可解析部分的**全部內容**。本節其餘所有東西都在對抗這一項。

但退化受**容量**限制，而這件事很重要：

> **p = f(β) 是每個身材 36 個全域數字，橫跨 540 個 clip、所有 window 位置、所有 timestep 共用。** 整個 upper level 的有效自由度約 360，對上 5400 個 (motion, body) pair。

36 個全域數字沒辦法把 540 段風格迥異的動作變成站姿。真正會發生的失敗不是塌成平凡解，而是**系統性振幅偏移**：p 把所有 joint gain 往 box 下緣壓（`g → 0.8`）、把 `dz_root` 壓低，因為低振幅、貼地、被阻尼的動作一律比較好追。以下設計要防的是這個。

### 3.2 防禦一：結構防禦（最重要，不依賴權重調校）

`RetargetNet` 輸出 pre-activation `u ∈ R^36`，經**硬 tanh box** 轉成 retarget 參數：

```
log_s_root = log(h_tgt/h_src)·1₃ + 0.15·tanh(u[0:3])     # root xyz scale, ±16%
dz_root    =                       0.08·tanh(u[3])        # root z 偏移 (m)
w_root     =                       0.15·tanh(u[4:7])      # root 姿態修正, exp-map, body frame, ±8.6°
g_joint    = 1.0                 + 0.20·tanh(u[7:21])     # 14 groups 振幅 gain, ±20%
b_joint    =                       0.15·tanh(u[21:35])    # 14 groups 角度偏移, ±0.15 rad
log_tau    =                  log(1.25)·tanh(u[35])       # time-warp, Stage 3 前凍結在 0
```

`apply_retarget` 的作用：

```python
ref_root_pos  = src_root_pos * exp(log_s_root) + [0, 0, dz_root]
ref_root_quat = src_root_quat ⊗ exp_map(w_root)
ref_hinge     = g_joint[group_of_joint] * src_hinge + b_joint[group_of_joint]
ref_raw       = concat(ref_root_pos, ref_root_quat, ref_hinge)
ref           = clamp(ref_raw, jnt_lo, jnt_hi)      # 只 clamp hinge
```

**14 個 joint group，左右綁定**（不學手性）：
`hip, knee, ankle, toe, thorax, shoulder, elbow, wrist, hand, torso, spine, chest, neck, head`
尺寸 `6,6,6,6,6,6,6,6,6,3,3,3,3,3 = 69`。分組沿用 `scripts/scale_robot.py:58 BODY_GROUPS` 的身體分群邏輯。

三個關鍵性質：

1. **p = 0 精確重現現有的 `retarget_qpos()`**。`h_tgt/h_src` 直接沿用 `scripts/qpos_retarget.py:83 load_root_rest_height`。naive retarget 是**架構原點**，不是一個靠權重維持的軟目標。這是不會失效的那種保證。
2. **在 box 角落**：root scale 偏差 ≤16%、每個關節振幅 ≤20% 且偏移 ≤0.15 rad。塌成站姿在**結構上不可能**，不是「被懲罰」而是「做不到」。
3. **β-only 條件**：`RetargetNet(β) → u`，沒有 clip identity、沒有 timestep、沒有 state。逐 clip 或逐幀的 p 會立刻退化。

### 3.3 防禦二：clamp / penalty 分離

`apply_retarget` 回傳**兩個**張量：

- `ref_raw`（未 clamp、可微）→ 餵給 feasibility penalty `C`
- `ref = clamp(ref_raw, jnt_range)`（可微但飽和）→ 餵給 RSI 與 tracking reward

理由：`torch.clamp` 在範圍外的 subgradient 是零，會**恰好殺掉我們最需要的那個梯度**（把違規角度拉回合法範圍的梯度）。保留 raw 值給 penalty 用即可繞過。

### 3.4 Upper 目標函數

```
F(p) = λ_gap  · G(τ, ref_p)          # new.md 要的：robot 與參考的差距
     + λ_fid  · S(ref_p, src)        # 語意保真（對 SOURCE clip，身材不變量）
     + λ_feas · C(ref_raw, β)        # 這個參考在這具身體上合不合法
     + λ_ext  · E_ext(τ)             # 外力作弊懲罰
     + λ_phys · P(τ)                 # robot 物理合理性
     + λ_prox · ‖u − u_prev‖²        # proximal / trust region
```

| 權重 | 值 |
|---|---|
| `λ_gap` | 1.0 |
| `λ_fid` | **3.0** |
| `λ_feas` | 2.0 |
| `λ_ext` | 0.5 |
| `λ_phys` | 0.5 |
| `λ_prox` | 1.0 |

#### 為什麼 `λ_fid : λ_gap = 3 : 1`（這個比例是可解釋的，不是盲調）

`λ_gap·G` 保留下來的梯度是「把 ref 拉向 τ」的純位移，`λ_fid·S` 是「把 ref 拉向 source 語意」的純位移。兩者的穩定點是一個收縮估計：

```
ref* ≈ argmin_r  λ_gap‖r − τ‖² + λ_fid‖r − src‖²
     = (λ_gap·τ + λ_fid·src) / (λ_gap + λ_fid)
```

所以 **`λ_gap/(λ_gap+λ_fid) = 25%` 直接就是「參考動作最多被允許往 robot 漂移的比例」**。這是要拿來推理的旋鈕，不是要盲試的超參數。它與硬 box 相乘疊加（box 額外限制絕對漂移量）。

#### G — 差距項

與 lower level 的 tracking reward 用同一組通道，權重也對齊：

```
G = mean_t [ 1.0 · ‖q_t − q̂_t‖² / 69                                       # hinge 角度 (rad²)
           + 2.0 · (1/5) Σ_e ‖(p_e − p_root)_t − (p̂_e − p̂_root)_t‖²        # 5 個 EE，root-relative (m²)
           + 1.0 · ‖p_root,t − p̂_root,t‖²
           + 0.5 · ‖log(q̂_root,t⁻¹ ⊗ q_root,t)‖² ]
```

`q̂` 用 **clamped** 的 `ref`，不是 `ref_raw` — 不能因為機器人達不到一個非法角度就怪它。
EE 清單重用 `model/kinematics.py:8 EE_BODIES = ["L_Hand","R_Hand","L_Toe","R_Toe","Head"]`。

#### S — 語意保真（有牙齒的那一項）

對照對象是 **source clip 在 source body 上**的表現，且全部用無因次或尺度正規化的量，讓「合法的身材差異」不付代價、「語意改變」付大代價：

```
S = 1.0 · mean_t ‖θ_ref,t − θ_src,t‖² / 69                        # 關節角：拓樸相同 → 本來就是身材不變量
  + 1.0 · mean_t (1/5) Σ_e ‖ê_ref,e,t − ê_src,e,t‖²               # ê = (p_e − p_root)/L_limb(β)：無因次 reach
  + 2.0 · mean_t BCE(σ(−k·z_foot,ref,t), c_src,t)                 # 接觸樣態 (k=200) ← 主力
  + 1.0 · mean_t (Δψ_ref,t − Δψ_src,t)²                           # 逐步航向變化 (rad)
  + 1.0 · mean_t (Fr_ref,t − Fr_src,t)²                           # Froude number ‖v_root,xy‖/√(g·L_leg(β))
```

`L_limb(β)`、`L_leg(β)` 每具身體從 MjModel 的 `body_pos` 讀一次即可。

**接觸樣態項是主力**：如果 p 壓低振幅或降低 root，腳離地/著地的時序會立刻位移，這一項馬上點火。這正是退化最先破壞的東西。

`S` 只依賴 p，梯度**精確且零變異**。

#### C — 參考在目標身體上的可行性（純 p，梯度精確）

這一項把 §1.2 的 95.8% 發現直接編碼進目標函數：

```
C = 4.0 · mean_{t,j} relu(|a_raw,j,t| − 1)²        # actuator/關節限位違反，a_raw = 2(q̂_raw−lo)/(hi−lo)−1
  + 2.0 · mean_t relu(−(minGeomZ_t − z_tol))²      # 穿地，z_tol = −0.005（沿用 qpos_retarget.py:161 的預設）
  + 0.5 · mean_t relu(minGeomZ_t − z_float)²       # 飄浮（source clip 有腳著地時卻離地 >3 cm）
  + 1.0 · mean_t Σ_feet c_src,t · ‖v_foot,xy,t‖²   # source 宣告接觸期間的腳滑
  + 0.5 · mean_t ‖q̈_ref,t‖² / 69                   # 參考動作平滑度（二階差分）
```

`minGeomZ` 用向量化 torch 形式取代 `scripts/qpos_retarget.py:99 _min_geom_z` 的 Python geom 迴圈：box 的 8 個角是固定的 `(8,3)` 符號矩陣，整件事是一個 einsum over `(B,T,26,8,3)`。

**這裡也是 `ground_correct_qpos` 的歸宿**：`scripts/qpos_retarget.py:127` 那個逐幀啟發式抬升，被一個由可微 penalty 驅動的**可學全域 `dz_root`** 取代。概念上乾淨，且移除了 pipeline 中唯一不可微、不能 runtime 化的步驟。

#### E_ext 與 P — 反作弊項

```
E_ext = mean_t [ ‖f_ext,t‖²/(Mg)² + ‖m_ext,t‖²/(Mg·L_leg)² ]
P     = physics_penalty 的 per-step 版（limit / fall / com_support / foot_slide / penetrate / smooth，見 §7.3）
```

這兩項**必須放在 F 裡，不能只放在 lower reward**。否則會形成迴圈：p 把參考變難 → lower level 用 100 N 的外力硬撐 → 差距照樣小 → p 完全不用付代價。

兩項都只透過 τ 依賴 p，所以對**精確梯度貢獻為零** — 它們由 §4.4 的 ES 估計器涵蓋。Stage 1–2 若不開 ES，就只能靠硬 box 兜底。

### 3.5 防禦三：診斷（在賠掉一整次訓練前就發現退化）

每 10 iterations 記錄：

| 指標 | 健康值 | 意義 |
|---|---|---|
| `frac_saturated = mean(\|tanh(u)\| > 0.9)` | **< 0.2** | **警報**。>20% 的分量貼在 box 牆上，代表目標函數在往退化推，`λ_fid` 太小。 |
| `S(ref_p) / S(ref_0)` | < 2.0 | 語意漂移量 |
| `ref_min_z`（參考最低幾何點的高度） | ≥ **−0.005 m** | **upper level 的頭條成功指標**（2026-08-10 改；理由見下） |
| `frac_illegal_frames(ref_p)` | 下降即可，**不設門檻** | 見 §0.3 的修正註記：0.958 是「至少一個關節越界」，但平均只超 0.6°，壓到 0.2 既不必要也不代表品質 |
| `mean(g_joint)` per group | 不應單調趨向 0.8 | 14 個 group 一起往下漂 = 振幅萎縮失敗模式 |
| `G(τ, ref_0)` vs `G(τ, ref_p)` | **兩者都要降** | **決定性檢驗** |

最後一項最重要：**如果 `G(τ, ref_p)` 降而 `G(τ, ref_0)` 不降，upper level 就是在作弊** — 差距是靠移動參考關掉的，不是靠機器人變好。多算一次 torch FK 就能得到，成本可忽略。

---

## 4. Bilevel 形式與 TTSA

### 4.1 問題設定

```
Lower:  φ*(p) ∈ argmax_φ  J(φ; p) = E_{(m,β)~D, w~Window, τ~p_φ(·|ref_p)} [ Σ_{t=0}^{23} γᵗ r(s_t, a_t; ref_p) ]
Upper:  min_p  F(p, φ*(p))                                                    # §3.4
```

- `φ = {LatentAdapter, ActionHead, RootWrenchHead}`（value net `ξ` 是輔助的，不屬於 bilevel 變數）
- `p = RetargetNet 權重`
- `H = 24`

**耦合是雙通道的**，這點不尋常且重要：p 同時進入 lower level 的 **reward** `r(·; ref_p)` **和初始狀態分布**（RSI 從 `ref_p[window_start]` 起始）。後者是一個 episode 內的直接依賴 — 見 §4.3 的 T2。

### 4.2 TTSA 更新規則

```
for k = 1 .. 10000:
    以 (φ_k, p_k) 收集 256 windows × 24 steps                  # §6
    φ_{k+1} = Adam_φ(φ_k, ĝ_φ),  η_φ = 3e-4 → 1e-4 (cosine)     # PPO, §5
    把 ĝ_u^{ES} 累積進 upper-gradient buffer
    if k > 500 and k % K == 0:
        p_{k+1} = Adam_p(p_k, T1 + ĝ_u^{ES}/K + prox),  η_p = 1e-5
        硬夾 ‖u_p − u_p^{prev}‖_∞ ≤ 0.02
        u_p^{prev} ← u_p
```

| 旋鈕 | 值 | 理由 |
|---|---|---|
| `η_φ / η_p` | 3e-4 / 1e-5 = **30 : 1** | 時標分離 |
| Upper cadence `K` | **10** | 有效時標比 300 : 1 |
| p warmup | **前 500 iter 凍結在 p=0** | φ 得先值得被量測，`F(p,φ)` 才有意義。這是「φ 接近 φ*(p)」的實務替代品。 |
| `λ_prox` | 1.0，外加每步 `‖Δu‖_∞ ≤ 0.02` 硬夾 | 界定每個 upper step 造成的 reward 非平穩程度 |

**誠實註記（要寫進 `train_bilevel.py` 的 module docstring）**：Borkar 的 TTSA 條件（`Ση=∞`、`Ση²<∞`、`η_p/η_φ→0`）在 Adam 下並不嚴格成立。理論只是比例的指引；真正提供穩定性的是 proximal 項 + 硬夾 + cadence。不要在程式碼註解裡宣稱收斂保證。

### 4.3 Hypergradient 分解 — 保留什麼、捨棄什麼

令 `r = ref_p`，`τ` 為實際 rollout：

```
dF/dp  =  ∂F/∂r · ∂r/∂p                        [T1]  直接項，參考側
       +  ∂F/∂τ · ∂τ/∂r · ∂r/∂p                [T2]  模擬器對參考的 episode 內反應
       +  ∂F/∂τ · ∂τ/∂φ · dφ*/dp               [T3]  真正的 hypergradient（學習反應）
```

**T1：完整保留。** `∂r/∂p` 是 `RetargetNet` + `apply_retarget` 的 autograd。`∂F/∂r` 對 `S`、`C`、`λ_prox` 是精確的（它們不依賴其他東西），對 `G` 是給定 τ 下精確的（把實際 τ 當常數張量）。**零變異**。這是訊號主體，也正是「retargeting 用 torch 實作」的價值所在。

> ⚠️ 關鍵：`λ_ext·E_ext` 與 `λ_phys·P` 的 **T1 恆等於零** — 它們只透過模擬器依賴 p。所以純 T1 路徑**看不見外力作弊**。這正是 ES 要補的洞。

**T2：用反對稱 ES 估計。** T2 不小：`∂τ/∂r` 包含 RSI 通道，那裡初始狀態**字面上就是** `ref_p[t0]`。忽略 T2 的 upper level 不知道「移動參考也會移動機器人每個 window 的起點」。而 `∂τ/∂r` 恰恰是 MuJoCo 給不出來的東西。見 §4.4。

**T3：捨棄。** 理由依重要性排序：

1. `dφ*/dp = −(∇²_φφ J)⁻¹ ∇²_φp J` 需要在一個接觸不連續的 MDP 上做 RL 目標的 Hessian-inverse-vector product。連 `∇_φ J` 本身都只有 score-function 估計；二階估計的變異隨 `1/σ⁴` 爆炸，在每 iteration 6144 個樣本下不可用。沒有 MJX/jax 就沒有可微模擬器的替代路線。
2. 這是標準的一階／截斷 hypergradient 近似（first-order MAML、first-order DARTS、one-step-unrolled bilevel 都是同一招）。已知在「lower level 對 upper 變數的反應夠平滑」時可行 — 這裡由建構保證，因為 p 的硬 box 界定了 reward 目標能移動多遠。
3. 在時標分離下，該修正項通常是幅度／曲率修正，不是下降方向的符號翻轉。

**誠實的但書（同樣要寫進 docstring）**：捨棄 T3 之後我們解的**不是** bilevel 問題，而是 `min_p F(p, φ_k)` 搭配 `φ_k ← RL(p_k)` 的 Gauss-Seidel 交替。其駐點與 bilevel 的駐點不同。具體代價：**p 沒有動機去挑一個「現在難、但能養出更好 policy」的參考動作**。對這個應用而言這是特性不是缺陷 — 我們要的是忠實且可行的參考，不是課程規劃；而忠實與可行正是 T1 的 `S` 與 `C` 所編碼的。

### 4.4 T2 的 ES 估計器

擾動 p 的**輸出** `u ∈ R^36`，不是它的 ~10k 權重。36 維讓 ES 變得可行。

- 每 iteration：**10 bodies × 2 antithetic pairs = 40 個擾動組**，每組 6 windows = 240 windows；剩下 16 windows 是未擾動的 control，用於乾淨的 logging 與 T1 評估。
- `σ_p = 0.02`（pre-tanh 空間，約 0.4% gain 變化、0.003 rad 偏移變化）
- 只作用在依賴模擬器的部分 `F_sim = λ_gap·G + λ_ext·E_ext + λ_phys·P`：

  ```
  ĝ_u = (1/(2 σ_p G_b)) Σ_g [ F̂_sim(u + σ_p ε_g) − F̂_sim(u − σ_p ε_g) ] · ε_g
  ```

  然後 `u.backward(gradient=ĝ_u)` 傳到 `RetargetNet` 的權重。**不需要額外的模擬成本**。
- 累積 `K=10` iterations 再更新 → 每個 upper step 每具身體有 20 個 antithetic pair，對 36 維足夠。**ES 累積窗與 TTSA cadence 刻意設成同一個 K。**
- 形成 `ĝ_u` 前對 40 個 `ΔF_sim` 做 **rank normalization**（標準 ES 做法），讓估計器不受 `F_sim` 尺度漂移影響。

> **Common Random Numbers 是必要條件，不是優化。**
> 同一個 antithetic pair 內，`(clip, window_start, RSI 關節噪音, 整條 (24,75) action noise stream)` 必須**逐位元相同**，只有 `±σ_p ε` 不同。每 iteration 預先取樣 `(256,24,75)` 的 action noise，配對的列共用同一份。
> 沒有 CRN 的話，ES 訊號（在 `G` 上約 10% 相對變化）會被 rollout 噪音完全淹沒，該項會比不加還糟。
> **驗證方法**：令 `σ_p = 0` 的 pair 必須產生逐位元相同的 `qpos` 軌跡。

---

## 5. Lower level：PPO + 解析側通道

### 5.1 演算法選擇：PPO + GAE + 兩條解析側通道

**為什麼不沿用 episode-level REINFORCE**：它在 `300 × 69 = 20700` 個抽樣維度上只給一個純量學習訊號。縮短到 24 步會讓**每樣本 SNR 更差**，不是更好 — cost 的尺度縮小了，但 score function 的量級沒有。`model/diagnose_single_task.py` 的 docstring 已經懷疑 task-composition 噪音主導。per-step reward + GAE + β-conditioned value function 同時解決那個問題和 window 內的 credit assignment。

**為什麼不用 A2C**：在每 iteration 6144 個 fresh on-policy 樣本下可行，但這個架構是「強 frozen prior 上的 residual」，一次壞更新就毀掉 prior。PPO 的 clip 基本上是免費保險。

**為什麼不 backprop through time**：沒有可微模擬器（無 MJX、無 jax）。進入 `z_beta` 的解析路徑是真的，但它止於第一次 `env.step`。

### 5.2 Policy

```
z_beta = project(z0 + α · MLP_adapter([β, z0]))                   # α = 0.05（從 0.1 降），project = 16·normalize
raw    = frozen._actor(frozen._normalize(obs), z_beta, 0.2).mean  # (B,69)，經 z_beta 可微
μ_ctrl = ActionHead(raw, β)                                       # (B,69)
μ_wr   = RootWrenchHead([obs_root_feats, β, raw])                 # (B,6)  ← 新增
a ~ N([μ_ctrl, μ_wr], diag(σ)),   σ_ctrl = 0.05, σ_wrench = 0.05
```

- **加上 `project_z`**（§1.1 的既有缺陷）。`16·F.normalize()` 可微，一行。
- **單一 75 維 Gaussian，在 pre-squash 空間取樣**。worker 端套 `clip(a[:69], −1, 1)`（MuJoCo 因 `ctrllimited=true` 本來就會夾）與 `f = f_max·tanh(a[69:72])`、`m = m_max·tanh(a[72:75])`。PPO 的 log-prob 用**未 squash** 的值，所以它就是一個純 Gaussian，不需要 tanh Jacobian 修正。
- `RootWrenchHead`：`MLP(6 + 8 + 69 → 128,128 → 6)`，輸出在 **root local frame**；主進程用上一步的 root quat 旋到 world 再寫 `xfrc_applied`。旋轉等變，比 world frame 好學太多。
- 沿用 `model/train.py:141-143` 的技巧：呼叫 `model._normalize` / `model._actor` 而非 `@torch.no_grad()` 包住的 `model.act()`，讓梯度流進 `z_beta`。

**外力是訓練拐杖，不是交付物**：
`f_max = 0.5·M·g`（iter 0–2000），cosine 退火到 0（iter 8000）；`m_max = 0.5·f_max·L_leg`。
**所有回報的 eval 數字一律在 `f_max = 0` 下量測，從 iter 1 開始就是。**

### 5.3 Value function

```
V_ξ(obs(358), z_beta.detach()(256), β(8), phase(2)) → scalar        MLP 512, 512
phase = (t/H, (H−t)/H)
```

- **必要**。純量 EMA baseline 做不到 window 內的 credit assignment，也吸收不了 per-(clip, body) 的巨大 return 異質性（`diagnose_single_task.py` 標記過的混淆）。條件在 `(z_beta, β)` 上讓 V 直接學到 per-pair baseline。
- **`phase` 不可省**：`H=24` 時 value 必須知道 window 即將被截斷，否則 `t=24` 的 bootstrap 系統性錯尺度。
- 另加 **per-(clip, body) advantage 正規化**：對每個 pair 維護 return 均值的 EMA（momentum 0.99），在 pair 內標準化 advantage。
- Value target 用 running mean/std 正規化（PopArt-lite）。`H=24` 時 `t=0` 的 return 約有一半來自 bootstrap，V 早期尺度錯掉會讓 GAE 變垃圾。**加 100 iteration 的 value-only warmup**（policy 凍結）再開 policy loss。

### 5.4 GAE、γ、截斷

- **γ = 0.97**（有效視野 33 步 ≈ 1.1 s @ 30 Hz — 比 0.8 s 的 window 長，讓平衡恢復被賦值；又短到 bootstrap 不會主宰）。Ramp：iter 0–1000 用 γ=0.95，之後 0.97。
- **GAE λ = 0.95**
- **`t=24` 是 TRUNCATION 不是 termination** → 用 `V(s_24)` bootstrap。
- **跌倒 / 追蹤失敗是 TERMINATION** → `V=0`，該 window 中其後所有 step 從 loss 中 mask 掉。**不做 window 中途重置** — 重置需要即時送新的參考進去，且會破壞 GAE，為了邊際吞吐不值得。

### 5.5 PPO 更新

每 iteration 6144 transitions；**4 epochs × 4 minibatches**（minibatch 1536）；clip ε = 0.2；value clip 0.2；grad-norm clip **1.0**（從 `model/config.py:45` 的 5.0 降下來 — residual 架構脆弱）。

**`z_beta` 在每個 minibatch forward 內用當前 adapter 從 `(β, z0)` 重新計算**，讓 PPO ratio 正確反映 adapter 的更新。**adapter 是 policy 的一部分，不能當成凍結的 context 向量。** 它每 iteration 重算，不是每 window；每個 clip 一個 `z0`（來自 `data/z/<task>/<task>_<trial>.npy`），每具身體一個 `β`，都是確定性的。探索完全來自 action Gaussian。

總 loss：

```
L = L_clip + 0.5·L_value + λ_z·mean(1 − cos(z_beta, z0)) + λ_bc(k)·mean‖μ_ctrl − a_ref‖²
```

**`lambda_z` 如何存活**：仍然是一個直接、可微、不碰模擬器的項（就是 `model/train.py:305` 的做法，`λ_z = 0.1`）。兩點升級：

1. 改用 **cosine 形式** `1 − ⟨ẑ_β, ẑ_0⟩`。z 已投影到球面，歐氏範數會去懲罰一個現在由建構固定的半徑 — cosine 才是該流形上正確的距離。
2. 對 minibatch 內**唯一的 (clip, body) pair** 計算，而非 per-transition，避免被 window 數量隱性加權。

**BC 輔助項（利用 §1.3）**：

```
a_ref,t = clip( 2·(q̂_{t+1} − lo)/(hi − lo) − 1 , −1, 1 )
λ_bc : 1.0 → 0，cosine 退火於 iter 0–2000
```

這是一條**稠密、精確、零變異**的梯度直接進 ActionHead，條件數遠優於 PPO 能產出的任何東西。
**必須退火到零**：長期開著會與 frozen prior 對抗，而且它忽略動力學（讓伺服**停在** `q̂` 的位置目標，不等於在負載下**驅動你到達** `q̂` 的目標）。

### 5.6 Per-step reward

全部在 worker 內從 `data.qpos / qvel / xpos / actuator_force / contact` 加上參考幀算得。參考動作的 FK 在 window reset 時做 25 次 `mj_kinematics`（每次 0.0033 ms，共 0.08 ms，可忽略）。

```
r_t = 0.65·r_track + 0.15·r_reg + 0.20·r_surv                  # 有界於 [0,1]
```

**Tracking**（DeepMimic 式指數核 — 有界、對 value net 尺度友善）：

```
r_pose = exp(−2.0   · (1/69) Σ_j (q_j − q̂_j)² )                                  # rad²
r_vel  = exp(−0.005 · (1/69) Σ_j (q̇_j − q̂̇_j)² )                                 # (rad/s)²
r_ee   = exp(−40    · (1/5)  Σ_e ‖(p_e − p_root) − (p̂_e − p̂_root)‖² )            # m²
r_root = exp(−10    · ( ‖p_root − p̂_root‖² + 0.5·‖log(q̂_root⁻¹ ⊗ q_root)‖² ) )
r_com  = exp(−10    · ‖com_xy − ĉom_xy‖² )                                       # subtree_com[0]，同 kinematics.py:32

r_track = 0.35 r_pose + 0.15 r_vel + 0.25 r_ee + 0.15 r_root + 0.10 r_com
```

> **`r_pose` 用原始弧度，不要除以 joint range。** 那 12 個 range 只有 0.16–0.30 rad 的近乎鎖死 DOF 一旦被正規化就會主宰 `r_pose`，但它們對視覺幾乎沒有貢獻。用原始弧度時它們自然自限，因為物理本來就會夾住。

**Regularization**（單一指數包住加權懲罰和，維持在 `(0,1]`）：

```
e_act    = (1/69)‖a_ctrl‖²
e_smooth = (1/69)‖a_ctrl,t − a_ctrl,t−1‖²
e_res    = (1/69)‖a_ctrl − raw_prior‖²                       # 讓 head 維持是 frozen actor 的 residual
e_tau    = (1/69) Σ_j (data.actuator_force_j / forcerange_j)²
e_ext    = ‖f_ext‖²/(Mg)² + ‖m_ext‖²/(Mg·L_leg)²
e_slip   = Σ_{接觸中的腳} ‖v_foot,xy‖²

r_reg = exp( −( 0.1·e_act + 1.0·e_smooth + 0.5·e_res + 0.5·e_tau + 8.0·e_ext + 2.0·e_slip ) )
```

`e_ext` 的係數 8.0 是全場最大的：**外力是滿足其他所有項最便宜的路徑**，必須讓它貴。

**Survival**（`r_surv = 1` 存活否則 `0`，且終止結束該 window）：

```
alive =  root_z > 0.5 · ẑ_root,t
      ∧  up_z  > 0.2                                    # up_z = 2·(q_y·q_z + q_w·q_x)
      ∧  ‖p_root − p̂_root‖ < 0.5 · (L_leg(β)/L_leg(adult))
      ∧  (1/69) Σ_j (q_j − q̂_j)² < 0.6                  # tracking-failure termination
```

> ⚠️ **`up_z` 公式逐字抄自 `model/losses.py:203`，連註解一起。**
> 這個 asset 的 Pelvis 帶 `euler="90 0 0"`，rest quaternion 是 `(0.7071, 0.7071, 0, 0)`，local +Y 而非 +Z 對應 world up。教科書的 `1 − 2(qx²+qy²)` 在這裡是錯的。`model/losses.py:188-195` 那段註解是有人付過代價換來的。

**tracking-failure termination 是 motion-tracking RL 中槓桿最大的一招** — 它讓 `r_surv` 成為真訊號而不是常數。

> **worker 回傳未加權的 14 維 term 向量，不是純量 reward。**
> 理由有二：(1) ES 估計器必須能在不重跑模擬的情況下重新加權 `F_sim`；(2) reward 權重掃描變成免費。每個 (env, step) 約 14 個具名浮點數，權重由主進程套用。

---

## 6. SimPool：多進程模擬設計

### 6.1 拓樸

**32 worker process × 8 sims = 256 envs**，用 `os.sched_setaffinity(0, {i})` 把 worker `i ∈ [0,32)` 釘在第 `i` 個**實體核**，不碰 HT siblings（32–63）。主進程浮在 HT sibling 上。實測釘核值約 14%。

實測替代方案（24-step iteration）：16×16 = 707 ms；30×9 = **632 ms**；48×6 = 1550 ms；64×4 = 1671 ms。超過 32 之後 worker 互搶實體核，加上 spin-wait 會急劇惡化。30×9 與 32×8 在雜訊範圍內，取 32×8 剛好湊到 256。

> **不要用 `multiprocessing.Barrier`。** 相同工作量實測 1030 ms vs 632 ms — 33+ 個 waiter 時每次 barrier release 都是 thundering herd，一個 iteration 48 次 barrier 直接主宰執行時間。
> 用共享記憶體 token + **spin-then-yield**（自旋約 2000 次後 `time.sleep(0)`）；實測 632 ms vs 純自旋 759 ms。

**關於沒有選的那條路**（記在 docstring 供日後參考）：thread pool 在 `mj_step` 上幾乎完美擴展（它會釋放 GIL — 實測 8 threads 跑 8 個並行 env 是 1.30 ms wall，單一 env 是 1.23 ms），但 `compute_humanoid_self_obs_v2` 是純 numpy、受 GIL 綁住（8 threads 只有 3.3×）。「physics 用 thread pool + obs 在主進程批次化」是一條實在的路，價值約 2×，但需要把 `compute_humanoid_self_obs_v2` 改寫成批次形式並確保逐位元一致。v1 用 process 是對的選擇；吞吐若成為瓶頸再去拿。

### 6.2 跨進程邊界的資料

全部走 `multiprocessing.shared_memory` + numpy view，逐步無 pickle。佈局的單一真相來源在 `bilevel/sim/protocol.py`。

**Main → workers**

| Block | Shape | dtype | 備註 |
|---|---|---|---|
| `ctrl` | (256, 69) | f32 | 已夾到 [−1,1] |
| `xfrc` | (256, 6) | f32 | root wrench，**world frame**（主進程從 root-local 旋過來） |
| `reset_mask` | (256,) | u8 | |
| `ref_qpos` | (256, 25, 76) | f64 | p 調整後、已 clamp 的參考 window；只在 window reset 時寫。3.9 MB |
| `ref_qvel0` | (256, 75) | f64 | 由 `mj_differentiatePos` 產生 |
| `rsi_noise` | (256, 69) | f32 | **在主進程取樣**，才能 seed 且供 ES pair 共用 CRN |
| `cmd` | (32,) | i64 | 每 worker 一個 token；負值 = shutdown |

**Workers → main**

| Block | Shape | dtype | 備註 |
|---|---|---|---|
| `obs` | (256, 358) | f32 | |
| `rew_terms` | (256, 14) | f32 | **未加權**的分項 |
| `done` | (256,) | u8 | |
| `qpos` | (256, 76) | f64 | 供 `G` 與診斷用 |
| `qvel` | (256, 75) | f64 | |
| `ack` | (32,) | i64 | |

總計 < 6 MB。

### 6.3 每步協定

```
main:     寫 ctrl、xfrc（若要 reset 則一併寫 reset_*）；token += 1；cmd[:] = token
          spin-then-yield 直到 (ack == token).all()
          讀 obs、rew_terms、done、qpos、qvel

worker w: spin-then-yield 直到 cmd[w] != seen；seen = cmd[w]；若 seen < 0 則 exit
          for 其負責的 8 個 env slot e:
              if frozen[e]:        continue              # 已終止 → 零成本
              if reset_mask[e]:    do_rsi(e)
              else:
                  d.ctrl[:] = ctrl[e]
                  d.xfrc_applied[PELVIS_ID, :] = xfrc[e]
                  mj_step(m, d, nstep=15);  mj_step1(m, d)
              obs[e]       = compute_humanoid_self_obs_v2(...)
              rew_terms[e] = step_terms(...)
              done[e]      = not alive(...)
          ack[w] = seen
```

> **`mj_step(nstep=15)` 之後必須再呼叫 `mj_step1`。**
> `compute_humanoid_self_obs_v2` 讀 `data.sensordata[:144]`（48 個 `framelinvel`/`frameangvel` sensor），只有 sensor stage 更新後才有效。`humenv/env.py:121-126 HumEnv.step` 就是這樣做的，照抄。

### 6.4 RSI / window reset

用 `reset_mask` 以獨立指令帶內下達（該步不跑物理，只出 obs — 所以一個 window 是 25 個指令跑 24 個物理步，約 4% 額外開銷）。

```python
mujoco.mj_resetData(m, d)                        # 注意：這也會清掉 xfrc_applied
d.qpos[:]   = ref_qpos[e, 0]                     # p 調整後的參考幀
d.qpos[7:] += rsi_noise[e]                       # 只對 69 個 hinge 加高斯；root pos/quat 精確保留
np.clip(d.qpos[7:], jnt_lo, jnt_hi, out=d.qpos[7:])
d.qvel[:]   = ref_qvel0[e]
mujoco.mj_forward(m, d)                          # 填 sensordata → obs 立即有效

if d.ncon and d.contact.dist.min() < -0.01:      # 噪音把身體推進自己或地板
    rsi_noise[e] 減半重試（最多 3 次），再不行退回零噪音
```

> **`ref_qvel0` 必須用 `mujoco.mj_differentiatePos(model, v, 1/30, qpos[t], qpos[t+1])`**，不能用 qpos 的有限差分。
> free joint 的 `qvel[3:6]` 是 **body-local 角速度**，4 維四元數的差分產不出 3 維角速度。`scripts/check_retarget_actuator_feasibility.py` 的 docstring 已記錄過這個陷阱。

**σ_q 排程**：每個 window 重抽 `σ_q ~ U(0, 0.08)` rad，而不是用固定值。value function 因此看到初始追蹤誤差的一個分布，條件化效果遠優於單一操作點，同時充當隱性的韌性課程。固定 `σ_q = 0.05` 是備案。

### 6.5 外力施加

`d.xfrc_applied[PELVIS_ID, 0:3] = f_world`、`[3:6] = m_world`。MuJoCo 在 body CoM 以 world 座標施加，且會**跨 `mj_step` 持續**直到被覆寫（並由 `mj_resetData` 清零 — 所以 §6.4 的順序很重要）。head 輸出 root-local，主進程用上一步的 root quaternion 旋轉。大小由 `f_max·tanh(·)` 界定，退火見 §5.2。

### 6.6 沒有 `is_terminated()` 的終止處理

完全繞過 `humenv`（`humenv/env.py:138-139 is_terminated()` 永遠回傳 `False`）— worker 自己持有 `MjModel`/`MjData` 並自行評估 §5.6 的 `alive` 判準。

- `done` 時 worker 設 `frozen[e] = 1`，該 window 剩餘時間跳過所有物理（終止的 env 變免費）。
- 主進程把終止後的所有 step 從 GAE 與 loss 中 mask 掉，終端 bootstrap `V = 0`。
- **同時捕捉發散**：`humenv/env.py:122-125` 在 `data.warning.number.any()` 時拋例外。worker 內改為視同 `done=1` + `mj_resetData` + freeze，該步 reward 歸零。
- **worker 絕對不能因為一個壞 clip 而死** — 死掉會讓主進程的 spin loop 永久 deadlock。需要 watchdog：`ack[w]` 5 秒未推進就記錄該 worker 的 env slot 並乾淨中止。

### 6.7 吞吐量

實測（當前機器 load avg ~180）：

| 元件 | ms / iteration |
|---|---|
| Sim pool (32×8，釘核，spin+yield，25 指令) | ~660 |
| Policy forward，24 次循序 @ B=256 | 71 |
| Host↔device 傳輸 | 75（用 pinned buffer 可降） |
| PPO 更新（4 epochs × 4 minibatches，6144 transitions） | ~100–150（估） |
| Upper level torch FK + `S`/`C`/ES（每 K=10 次） | ~30（攤提） |
| **合計** | **~0.85–1.0 s** |

**10000 iterations ≈ 2.4–2.8 小時**（當前負載）；閒置機器預期 **~1.2–1.5 小時**。61.4M env steps @ 約 7–15k env-steps/s。

> 承諾排程前請在閒置機器上重測 — 目前的 load average 是一個真實的干擾因子。

---

## 7. 檔案結構與去留

### 7.1 新增：`model/bilevel/`

```
model/bilevel/
├── config.py           272   所有超參數
├── quat.py             124   批次四元數運算（torch）
├── torch_kin.py        334   可微 FK ── 上層的地基
├── retarget.py         256   p = RetargetNet + 硬 tanh box
├── semantics.py        268   S（語意保真）與 C（可行性）
├── data.py             306   clips × bodies × windows 取樣
├── policy.py           165   φ = 三個 head 掛在 frozen actor 上
├── rewards.py          303   per-step 14 項（純 numpy）
├── sim/
│   ├── protocol.py     129   共享記憶體佈局：單一真相來源
│   ├── worker.py       228   單一 worker 進程（不得 import torch）
│   └── pool.py         219   32 × 8 = 256 env 的 lockstep 池
├── rollout.py          243   收集一批 on-policy 資料
├── ppo.py              268   下層更新
├── upper.py            497   上層更新（截斷 hypergradient）
├── train_bilevel.py    318   TTSA 主迴圈 ── entry point
├── eval_bilevel.py       —   held-out 四象限評估
└── tests/
    ├── test_torch_kin.py   201   R3 硬門檻
    ├── test_retarget.py    168   p=0 錨點
    └── test_simpool.py     157   RSI / 外力 / 終止 / 吞吐
```

**基礎**

| 檔案 | 主要 API | 說明 |
|---|---|---|
| `config.py` | `BilevelConfig` | 一個 dataclass 裝下全部超參數，附排程輔助（`wrench_scale(it)`、`bc_scale(it)`、`gamma_at(it)`、`lr_lower_at(it)`）。任何偏離 `model/config.py` 舊值的地方都註明了舊值與理由 |
| `data.py` | `WindowDataset.sample()` → `(clip, body, t0)`；`.build_batch()`；`.body_assignment()`；`BodySpec`；`ref_qvel_from_qpos()` | 430 clips（train split，磁碟上共 540）一次全載入 RAM（63 MB），不需 manifest、不需 build step。`body_assignment` 是**固定的** env slot → body 連續指派，避免 worker 切換 `MjModel`。`ref_qvel0` 必須走 `mj_differentiatePos`，free joint 的 `qvel[3:6]` 是 body-local 角速度 |

**上層（p）**

| 檔案 | 主要 API | 說明 |
|---|---|---|
| `quat.py` | `quat_mul` `quat_rot` `quat_log_map` `exp_map_to_quat` `quat_up_z` | 形狀泛用 `(...,4)`／`(...,3)`，全部可微。`quat_up_z` 是 `2(q_y q_z + q_w q_x)`，逐字抄自 `model/losses.py:203` —— 這個 asset 的 Pelvis 帶 `euler="90 0 0"`，教科書公式在這裡是錯的 |
| `torch_kin.py` | `TorchKinematics.forward(qpos (B,T,76))` → `xpos (B,T,25,3)`, `xquat (B,T,25,4)`；`.com()` `.min_geom_z()` `.normalized_ctrl()` `.clamp_hinges()` | 建構時從 `MjModel` 讀死 `body_pos`/`body_quat`/`jnt_axis`/`geom_*`。`min_geom_z` 取代 `scripts/qpos_retarget.py:99` 的 Python geom 迴圈（box 8 角是固定 `(8,3)` 符號矩陣 → 一個 broadcast over `(B,T,26,8,3)`）。**索引全部預先轉成 Python int**，否則每個 body 一次 GPU 同步（實測 508ms → 25.8ms） |
| `retarget.py` | `RetargetNet(β(8)) → u(36)`；`RetargetParams(u)` → 硬 box；`apply_retarget(src, params)` → `(ref_raw, ref)` | `ref_raw` 未 clamp、保留範圍外梯度給 `C`；`ref` 已 clamp、給 RSI 與 tracking reward。**p=0 逐位元重現 `qpos_retarget.py:91`**。69 個 hinge 綁成 14 組、左右共用 |
| `semantics.py` | `UpperGeometry`（FK 快取）、`semantic_fidelity` → S、`reference_feasibility` → C、`gap` → G、`weighted_sum`、`frac_illegal_frames` | 三個函式都回傳**未加權的原始分項 dict**；加權與正規化統一在 `weighted_sum` 做 |
| `upper.py` | `UpperLevel.calibrate()` `.evaluate()` `.step()` `.accumulate_es()`；`EsPlan` | `calibrate` 以「p 在 box 邊界的值」為每項單位；`evaluate` 組 F(p) 並掛上 T1 精確圖；`step` 反傳 + L∞ trust region 回溯投影；`EsPlan` 管 CRN 配對與 per-body 擾動組 |

**下層（φ）**

| 檔案 | 主要 API | 說明 |
|---|---|---|
| `policy.py` | `LowerPolicy.latent(β,z0)` → `z_beta`；`.act_mean(obs,β,z_beta,root_feats)` → `(mean(75), raw_prior(69))`；`.split_action()` | 走 `frozen._normalize`/`_actor`（繞開 `@torch.no_grad()`）讓梯度進 `z_beta`。動作 = 69 ctrl + 6 root wrench，單一 75 維高斯、pre-squash 取樣 |
| `rewards.py` | `RewardContext.build()`、`reference_cache()`、`step_terms(...)` → `(14,)`、`combine(terms, cfg)` | **純 numpy，不得 import torch**（32 個 worker 進程 import）。worker 回傳**未加權**分項，加權在主進程 —— ES 才能不重跑模擬就重新加權 |
| `ppo.py` | `compute_gae()`、`update_lower()`、`PairAdvantageNormalizer`、`RunningScalar`、`reference_ctrl()` | `t=24` 是 truncation（bootstrap `V`），跌倒是 termination（`V=0`）。`reference_ctrl` 就是 BC 目標，由 `ctrl↔qpos` 恆等式免費得到 |

> `ValueNet` 不在此目錄 —— 它與 `LatentAdapter`/`ActionHead`/`RootWrenchHead` 一起放在
> `model/networks.py`，讓四個網路定義集中一處。`policy.build_value_net()` 是建構捷徑。

**模擬（`sim/`）**

| 檔案 | 說明 |
|---|---|
| `protocol.py` | `block_specs()` 列出每個共享記憶體 block 的 name/shape/dtype，主進程與 worker 都由它建 numpy view —— 形狀只能在一處改。`SharedBuffers` 負責 create/attach/close，`WorkerInit` 是唯一一次 pickle |
| `worker.py` | 釘核 → spin-then-yield → RSI／`mj_step(nstep=15)` + `mj_step1` → obs → reward → done。**不得 import torch**。發散不能讓 worker 死掉（主進程在 spin，會 deadlock），改成該 slot 凍結 |
| `pool.py` | `SimPool.reset_windows()` `.step()` `.close()` + watchdog。用 token + spin-yield，**不是 `mp.Barrier`**（實測 1030ms vs 632ms） |

**訓練與驗證**

| 檔案 | 說明 |
|---|---|
| `rollout.py` | `Collector.collect()` 驅動 SimPool 走 H=24 步，主進程負責：由 p 建參考、RSI 噪音取樣、CRN action noise、root-local wrench 旋到 world、終止後 mask。模擬用的參考是 detached；`upper.py` 需要梯度時自己重算 |
| `train_bilevel.py` | `build()` 接線、`apply_stage()` 套 Stage 1–4 預設、`train()` 跑 TTSA 主迴圈 |
| `tests/` | `test_torch_kin.py`（13 具 × 隨機姿態 vs `mj_kinematics`）、`test_retarget.py`（p=0 錨點 + box + 梯度）、`test_simpool.py`（RSI/外力/終止/吞吐）。三者都是**擋路的門檻**，不是回歸測試 |
| `eval_bilevel.py` | **尚未實作。** Held-out 評估：2 具未見身材（`giant`、`short_stocky`）+ held-out task split，`f_max=0`，**整段 clip**（非 24 步）rollout，沿用 `losses.functional_equivalence` 與 `losses.physics_penalty`，讓數字與 `outputs/{baseline,eval}/report.json` 直接可比 |

### 7.2 保留並重用

| 檔案 | 重用內容 |
|---|---|
| `model/networks.py` | `LatentAdapter`、`ActionHead` 原樣重用（`LatentAdapter.forward` 加 `project_z`；新的 `RootWrenchHead` 也放這裡） |
| `model/dataset.py` | `BETA_AXES:14`、`load_beta:18` 逐字重用 |
| `model/kinematics.py` | `EE_BODIES:8`、`FOOT_BODIES:9`、`ROOT_BODY:10`、`quat_to_yaw:48` 重用；`batch_forward_pose:13` 作為 torch FK 的對照標準 |

### 7.3 保留為 baseline，不改造

`model/train.py`、`model/train_explore.py`、`model/config.py`、`model/config_explore.py`、`model/run_train.py`、`model/diagnose_single_task.py`、`model/evaluate.py`、`model/baseline.py`。

它們的 docstring 記錄了付過代價的發現：為何 `exploration_std=0.05` 而非 0.2（`config.py:35-38`）、為何拿掉 `R_task`（`train.py:12-26`）、為何 `total_loss` 不是進度訊號（`train.py:190-196`）、task-composition 混淆（`diagnose_single_task.py`）。

想要乾淨的樹可以移到 `model/legacy/`，但**必須保持可執行** — 新系統得贏過它們，而且你會需要那個對照。

> 移動時注意：checkpoint 內 pickle 的 `cfg` 指向已不存在的路徑（`assets/robots/robot_child.xml`），`model/evaluate.py:89,111` 會因此爆掉。若要跑舊 baseline，需先修這條路徑。

### 7.4 保留供評估、不擴充

**`model/losses.py`。** `functional_equivalence:122` 與 `physics_penalty:272` 原樣不動，由 `eval_bilevel.py` 呼叫。

它們對新系統是錯的**形狀**（whole-trajectory numpy、post-hoc、只吃 qpos），**不要硬凹成 per-step 形式**。移植到 `bilevel/rewards.py` 的是：

| 來源 | 移植方式 |
|---|---|
| `_fall_penalty` 的 `up_z = 2(q_y q_z + q_w q_x)`（`:203`） | **連註解逐字抄**進 survival 判準 |
| `_joint_limit_penalty`（`:167-179`） | 改寫成單一向量化 `np.clip` over `jnt_range[1:]`；目前對 70 個 joint 的 Python 迴圈在 per-step 路徑上是純開銷 |
| `FOOT_CONTACT_HEIGHT = 0.05`（`:153`）的高度代理 | **換成 `data.contact` / `mj_contactForce` 的真實接觸偵測** — worker 有這個資訊而 post-hoc 版本沒有。這對 `_com_support_penalty`、`_foot_slide_penalty` 是實質的準確度升級 |
| `_com_support_penalty:209`、`_penetration_penalty:253`、`_smoothness_penalty:260` | 去掉外層 `np.mean` 即為 per-step 版 |
| `d_root` 的曲率機制（`:51-96`） | **per-step 不需要**，只留在 eval 路徑 |

> 附帶修正：`physics_penalty(..., dt=1.0)` 及其所有呼叫端都用預設值，導致 `foot_slide` 與 `smooth` 是 per-frame 而非 per-second 單位（@30 Hz 真值分別是 900× 與 810000×）。新的 per-step 版本一律用 `dt = 1/30`。eval 路徑維持現狀以保持可比性，但要在 docstring 註明。

### 7.5 被取代

| 舊 | 新 |
|---|---|
| `scripts/qpos_retarget.py:91 retarget_qpos` | `bilevel/retarget.py` 的 p=0 特例 |
| `scripts/qpos_retarget.py:127 ground_correct_qpos` | **移除**。不可微、逐幀、離線。改由 `C` 的穿地項驅動的可學 `dz_root`（§3.4）。腳本本身保留供產生離線 baseline |
| `scripts/qpos_retarget.py:99 _min_geom_z` | 向量化進 `torch_kin.py` |
| `scripts/build_dataset.py` | **不需要**。它寫死 `MORPHOLOGY_LABEL="child"`（`:45`）且 `MORPHOLOGY_SRC`（`:44`）指向已不存在的 `assets/robots/robot_child_parameter.json`；而它的整個前提（預先烘焙 `retargeted_motion`）正是本計畫要消除的。`bilevel/data.py` 直接讀 `data/origin_motion/`、`data/z/`、`assets/robots/*/parameter.json` |

---

## 8. 風險與分階段落地

### 8.1 風險排序

**R1 — `heavy` / `giant` / `short_stocky` 可能物理上不可控。**
實測 13 具身體的 actuator：

```
body            mass   gain[0]  frcrange  gain==adult
adult           71.8     287.3     360.0        True
athletic       101.2     287.3     360.0        True
child           38.2     287.3     360.0        True
elderly         66.1     383.0     479.8       False   ← 唯一有縮放的
giant          110.7     287.3     360.0        True
heavy          210.0     287.3     360.0        True   ← 2.9× adult 質量，adult actuator
long_limbed     47.4     287.3     360.0        True
pear_shaped     76.3     287.3     360.0        True
petite          39.7     287.3     360.0        True
short_limbed    64.3     287.3     360.0        True
short_stocky   130.5     287.3     360.0        True
tall_slim       49.6     287.3     360.0        True
teen            54.4     287.3     360.0        True
```

12/13 用 adult 的 `gainprm` **和** `forcerange`，質量橫跨 38–210 kg。既有稽核顯示 child（最輕的之一）已經 91% 的 frame 動力學不可行。若把 `heavy` 放進訓練集，它會永遠貢獻純梯度噪音。

**緩解（在做任何事之前）**：用 `scripts/scale_robot.py` 重新產生這些身體並**開啟** actuator scaling（拿掉 `--no-actuator-scale`；`compute_joint_loads:103` 就是為此寫的，只是 commit `9495329` 產生那 11 具身體時把它關掉了，`parameter.json` 裡留下 `"scale_actuators": false`）。再用 `scripts/torque_capability_check.py --mode static` 篩出 10 具可行身材。

**R2 — 截斷 hypergradient 唯一的精確訊號就是「把 ref 移向 robot」。**
結構性問題，§3.1 與 §4.3 已詳述。三重緩解：硬 box（無條件生效）+ `λ_fid`（可調）+ ES（Stage 3）。
**警報指標**：`frac_saturated > 0.2`。**決定性檢驗**：`G(τ, ref_0)` 必須跟著 `G(τ, ref_p)` 一起降。

**R3 — torch FK 與 MuJoCo 靜默不一致。**
Pelvis 帶 `euler="90 0 0"`、rest quaternion 是 `(0.7071, 0.7071, 0, 0)`，這個 asset 已經產生過一次「up 軸」bug（`model/losses.py:188-195`）。若 `torch_kin.py` 偏了，upper level 會在一個模擬器根本沒有的幾何上最佳化，而**所有 loss 曲線看起來會完全正常**。
**緩解：先寫斷言測試** — 13 具身體 × 1000 個隨機 qpos，`max|torch_fk(q) − mj_kinematics(q)| < 1e-5`，含 `min_geom_z` 對 `scripts/qpos_retarget.py:99` 的對照。這件事要在 upper level 的任何其他東西之前完成。

**R4 — 非平穩 reward 破壞 PPO。** 每次 p 更新都會位移 value target。緩解：`K=10` + `η_p=1e-5` + proximal + L∞ 硬夾。**監測**：每次 upper step 之後的 value loss；若尖峰 >2× 就加大 K。

**R5 — 外力變成永久拐杖。** policy 一定會學會用 0.5·Mg 撐住自己。緩解：退火到零 + `e_ext` 權重 8.0 + **從 iter 1 起所有 eval 都在 `f_max=0`**。

**R6 — 24 步視野學不到需要 >0.8 s 上下文的東西。** 步態相位、轉身完成、跳躍頂點。梯度層面接受這個限制，但每 100 iterations 追蹤一次 300 步 rollout 指標，才會發現短視野 policy 是否在把局部良好、全局不連貫的動作拼在一起。

**R7 — 對目標身體根本不可能的參考幀。** headstand 重定向到 `short_stocky` 可能無論 p 怎麼調都起不來。24 步 window 下 policy 永遠學不到從那裡恢復，它會變成永久噪音源。緩解：per-(clip, body) advantage 正規化（§5.3），並可選擇在前 1000 iterations 課程式過濾掉最難的 10% pair。`model/diagnose_single_task.py` 已經懷疑過舊系統有這個混淆，別讓它靜默重演。

**R8 — RSI 噪音造成穿透的初始狀態**，在寬體身材上最嚴重。緩解：§6.4 的 clip + `ncon`/`contact.dist` 檢查 + 重試。

**R9 — ES 訊號被 rollout 噪音淹沒。** 有限差分在 `G` 上約 10% 相對變化。**CRN 不是可選項。** 若 ES 指標看起來像白噪音，用「`σ_p=0` 的 pair 必須產生逐位元相同的 qpos」來驗證 CRN。

**R10 — 共用機器上的吞吐。** 目前 load avg ~180，32 個實體核。釘核，並準備好 2–3× 的 wall time 變異。

**R11 — `z_beta` 脫離 FB manifold。** 現有程式碼的已驗證缺陷。跑任何東西之前先加 `project_z`（一行，可微）。

### 8.2 分階段落地

**Stage 0 — 驗證基礎（不可跳過，約 1 天）**

1. `torch_kin.py` 對 `mj_kinematics` 的斷言測試，13 具身體全過（R3）
2. 逐身體 actuator 可行性稽核；選定 10 具訓練身材（R1）
3. RSI 煙霧測試：13 bodies × 540 clips，從 5 個隨機幀以 `σ_q=0.08` 做 RSI 並用零 ctrl 走 24 步；統計發散數與穿透數
4. 在**閒置機器**上重測 SimPool 吞吐；定下 `N_w` 與 `M`
5. 驗證 `apply_retarget(src, u=0)` 逐位元重現 `scripts/qpos_retarget.py:91 retarget_qpos`
6. 記錄 baseline：`frac_illegal_frames = 0.958` 與每具身體的 `S(ref_0)` — 這是 Stage 2 的目標線

**Stage 1 — 只跑 lower level，p 凍結在 0（承重階段）**

完整 PPO 堆疊，256 windows × 24 steps，BC 輔助開，`f_max = 0.5·Mg` 固定。**只用 2 具身材**（`child`、`teen`）以便快速迭代。

**通過門檻**：2000 iterations 內達到 `r_track > 0.6` 且 `term_rate < 0.3`（原為 < 5%，改動理由見 §0.9 Stage 1）。
**若 lower level 連 naive reference 都追不上，後面所有東西都沒有意義** — 停下來修 lower level。

**Stage 2 — 精確梯度的 upper level**

開啟 p：`λ_gap + λ_fid + λ_feas + λ_prox` 加硬 box，`K=10`，`η_p=1e-5`，500 iteration warmup。ES 關閉。

**通過門檻**：
- `ref_min_z ≥ −0.005 m`（參考不再穿地；取代原本的 `frac_illegal < 0.2`，見 §0.9 Stage 2）
- `S(ref_p) < 2·S(ref_0)`
- `frac_saturated < 0.2`
- **且 `G(τ, ref_0)` 有降，不只是 `G(τ, ref_p)`**

**Stage 3 — 外力退火 + ES（也就是全量跑）**

`f_max → 0`（iter 2000–8000）。開啟 `λ_ext·E_ext + λ_phys·P + λ_gap·G` 的反對稱 ES。驗證 CRN。可選擇解凍 `log_tau`。給滿 `--iters 10000` 就是正式跑 —— 9 具身體、430 個訓練 clip。

跑完用 `model/bilevel/eval_bilevel.py` 做 held-out：2 具未見身材（`giant`、`short_stocky`）× held-out task split 的四象限，`f_max = 0`，長 rollout，用**原本的** `losses.functional_equivalence` / `losses.physics_penalty` 評分，讓結果與 `model/train.py` 的 baseline（`outputs/{baseline,eval}/report.json`）直接可比。

> 原本這裡還有一個「Stage 4 — 全量」。已移除：規模是 `--iters` 而非階段，Stage 2/3 早就用滿全部身材與 clip，那個 preset 與 Stage 3 逐欄位比對零差異。詳見 §0.9。

**Stage 4（選配）— 約束型 upper level**

把固定的 `λ_fid` 換成對 `E[S] ≤ ε` 的對偶上升：

```
λ_fid ← max(0, λ_fid + η_λ·(Ŝ − ε)),   η_λ = 0.01
ε 設為 p=0 時量到的 S 的 2 倍再加一點餘裕
```

這讓「可接受多少語意漂移」變成 `S` 單位下的可解釋量，而不是一個權重比；而且當 gap 項的尺度在訓練中漂移時它會自我修正。

---

## 9. 與 `new.md` 的對應表

| `new.md` 的要求 | 本設計的落點 | 狀態 |
|---|---|---|
| 訓練流程分兩階段：Upper / Lower | §2、§4.1 | ✅ |
| Upper 負責修正 retargeting motion，retargeting 要加入參數微調 | §3.2 `RetargetNet` + 36 維硬 box | ✅ |
| Lower 做 RL：最大化 tracking + regularization + survival reward | §5.6 | ✅ |
| Retargeting motion 是 runtime 取得的 | §2 資料流；`apply_retarget` 在每個 window reset 時跑 | ✅ |
| 原本 motion 共 540 × 10 筆 | 540 clips × 10 morphologies + 2 具 held-out；見 §11 偏差 A | ✅ |
| 訓練時取其中 **1024** 筆的任意 24 steps | **256** windows × 24 steps | ⚠️ 使用者已確認 |
| 整體訓練執行 10000 次 | §4.2 主迴圈 `for k = 1..10000` | ✅ |
| Upper 衡量機器人實際狀態與參數化參考動作 gt 的差距 | §3.4 的 `G` 項（`λ_gap = 1.0`） | ✅ |
| Rollout 時除 root 外關節要加高斯擾動 | §6.4 RSI：`qpos[7:] += rsi_noise`，root pos/quat 精確保留（實測誤差 0.00e+00） | ✅ |
| 用 TTSA 與 Approximate Gradient 避免等待 lower level 收斂 | §4.2、§4.3（T1 保留 / T2 用 ES / T3 捨棄） | ✅ |
| 額外的 external Force，30 Hz，對 root 施加，讓它不跌倒 | §5.2 `RootWrenchHead`；§6.5 `xfrc_applied`；控制頻率就是 30 Hz | ✅ |
| 外力要算入 regularization | §5.6 `e_ext`，權重 **8.0**（全場最大）；另在 §3.4 的 `E_ext` 再算一次 | ✅ |
| 用 latent adapter + action head，再多一個 retargeting 參數設計 | §5.2 三個 head + §3.2 `RetargetNet` | ✅ |

---

## 11. 實作結果與對本文件的修正

以下全部來自實作後的實測，與本文件先前的設計推估有出入之處以此節為準。

### 通過的硬門檻

| 門檻 | 結果 |
|---|---|
| **R3**：torch FK vs `mj_kinematics`，13 具身體 × 300 隨機姿態 | **PASS**，xpos/xquat/com/min_geom_z 全部 ≤ **2.2e-15** |
| **p=0 錨點**：`apply_retarget(u=0)` vs `qpos_retarget.py:91` | **PASS @ 2.2e-16**（float64 eps） |
| clamp/penalty 分離確實承重 | 經 `ref_raw` 的梯度 3.9e+03，經 clamped `ref` **恰好 0** |
| RSI 不碰 root | root qpos 誤差 **0.00e+00** |
| 外力真的進到模擬器 | +3000 N 下 root z 上升 +0.43 m |
| `frac_illegal_frames` 交叉驗證 | 訓練迴圈實測 **0.958**，與開場獨立量測的 95.8% 一致 |

### 偏差 A：actuator 必須按「實測力矩需求」校準，才湊得出 10+2 具身體

`scripts/audit_bodies.py` 量測每具身體撐住自己靜止站姿的力矩餘裕
（`forcerange / |static torque at qpos0|`，< 1 表示站不起來）。**出廠資產只有 8 具可用**：

```
adult 10.35   child 7.54   elderly 1.76   long_limbed 1.72
teen   1.60   petite 1.41  tall_slim 1.23  athletic 1.17
------------------------------ 以上可站立 / 以下不可 ------------------------------
pear_shaped 0.96   giant 0.48   short_limbed 0.09   short_stocky 0.04   heavy 0.03
```

13 具中有 12 具帶著 **adult 的** `gainprm`/`biasprm`/`forcerange`，質量卻橫跨 38–210 kg
（只有 `elderly` 被縮放過；9495329 產生的 11 具都記錄 `"scale_actuators": false`）。

**R1 建議的解法（`scale_robot.py` 載荷縮放）實測無效**（`scripts/regen_bodies.py` + 重稽核）：
它用**預測的**負載模型（rest pose 的 mass × lever arm）把 actuator 調成正比於負載，等於把餘裕
**收斂到 adult 自己的比值**而非給出餘裕 —— `heavy` 只到 0.11（仍差 10 倍），
而 `child` 7.54 → 3.00、`petite` 1.41 → **0.65（反而站不住）**。淨效果是少兩具、救回零具。
問題在於力矩需求根本不在 rest pose。

**有效解法：`scripts/calibrate_actuators.py`** —— 不用預測，直接**量測**。對真實 clip 的 p=0
參考姿態取靜態力矩的 p90，讓每具身體在同一段動作上取得與 adult 相同的餘裕，再加上
「rest pose 至少 3× 餘裕」的地板。結果全部 13 具都升到 ≥ 0.98，reference-pose 中位餘裕
0.24–0.46（adult 自己是 0.745）—— 每具身體變得與 frozen policy 原生的身體同一量級可控。

三個實作上必須做對、否則會靜默壞掉的點：

1. **`gainprm[0]`、`biasprm[0]`、`biasprm[1]` 必須同乘 k**。伺服平衡角是
   `q* = -(g·ctrl + b0)/b1`，三者同乘才保持 `ctrl∈[-1,1] ⟺ qpos∈jnt_range`（校準後實測
   1e-9~1e-10）。BC target 與可行性懲罰都靠這個恆等式。
2. **`armature` 與 `damping` 也必須同乘 k**。第一版漏了，結果 Kp 放大但轉子慣量沒有，
   顯式積分器的穩定條件 `dt·√(Kp/I) < 2` 被衝破：
   `athletic` 2.32、`child` **4.83**。整段訓練隨之崩壞 ——
   `pose_err` 0.05 → **8126**、終止率 0.34 → 0.86。補上後 `dt·ω` 與原始資產**逐具完全相同**，
   `pose_err` 回到 0.106。
3. **被動 `stiffness` 不可跟著放大**。它是與 actuator 獨立的建模選擇，放大等於在 actuator
   面前擺一個 k 倍強的彈簧，會出現在 `qfrc_inverse` 裡直接抵銷約 20% 的增益。

**最終配置**：`robots_dir = assets/robots_calib`（原始 `assets/robots/` 未動，legacy baseline
仍可重現）。

- **train (10)**：adult, child, teen, petite, tall_slim, long_limbed, short_limbed, athletic, elderly, pear_shaped
- **test (2)**：giant（110.7 kg / 1.17 m，比訓練集最高者更高）、short_stocky（130.5 kg，比最重者更重）
  —— 刻意選在訓練分布**之外**，held-out 數字才量測得到泛化而非內插
- **完全排除 (1)**：`heavy`（210 kg）。校準後 rest 餘裕仍只有 0.975，要達標需要約 250× 的
  actuator，那已經不是合理的人體。

這是一個**建模決策**（把 actuator 強度提高到超出原縮放模型所隱含的值），理由是：真實的重型
人體確實有等比更強的肌肉，而把每具身體正規化到 adult 的餘裕，正是跨形態研究該有的前提 ——
形態差異應該來自**幾何**，不是來自某些身體被削弱到動不了。

### 偏差 B：S / C 的權重必須按「box 邊界值」正規化

本文件 §3.4 給的權重（`c_limit=4.0` 等）在原始尺度下**完全不成立**。實測 p=0 時：

```
c_limit  9.7e-04     <- 本該主導、編碼 95.8% 違規發現的那一項
c_slide  5.6e-01
c_smooth 1.2e+03     <- 除以 dt² 再平方 = ×810,000
```

於是名目 4:1 的優先序實際是 **1:140**，C 的 99.9% 是平滑項，upper level 會把全部預算
花在磨平參考動作上。

第一版修法（按 p=0 值正規化）也錯：`s_heading` 在 p=0 是 **2.3e-32**（root 四元數原封複製，
轉向速率恆等）、`c_float` 是 **0**，除下去權重變 1e11。

**正解**：以 **p 在硬 box 邊界能達到的值**作為每項的單位（`UpperLevel.calibrate()`）。
每一項因此在 p 的整個可達集合上落在 [0,1]，config 的權重才真的是相對優先序。
副產品是一張有用的診斷表 —— p 對各項的槓桿：

```
c_limit    p=0 0.006 → corner 1.0   (30× 空間，符合設計意圖)
s_contact  p=0 0.069 → corner 1.0   (22× 槓桿，主力項有效)
c_slide    p=0 0.780               (p 幾乎無法改善腳滑)
c_smooth   p=0 0.940               (p 對平滑度基本無影響力)
s_froude   p=0 0.434
```

同時 §8.2 Stage 2 的門檻「`S(ref_p) < 2·S(ref_0)`」現在可直接讀成 `S < 2 × 0.56`。

### 偏差 C：效能 —— 模擬比預估快，但整體迭代慢

| 項目 | 本文件估計 | 實測 |
|---|---|---|
| SimPool（256 env × 24 步） | ~660 ms | **325 ms**（18,890 env-steps/s）✅ 快一倍 |
| 每 iteration 總計 | 0.85–1.0 s | **2.4 s** |
| 10000 iterations | 2.4–2.8 h | **~6.7 h** |

差距來自 PPO 更新（~1.2 s，16 次 forward/backward 穿過 frozen 12×2048 actor）與 collect 的
policy forward。過程中修掉兩個實測出來的效能地雷：

1. **torch FK 在 GPU 上 508 ms，CPU 只要 10 ms** —— forward 迴圈裡 `int(self.body_parent[b])`
   每個 body 強制一次 GPU→CPU 同步，24 body × 4 次 ≈ 96 次同步，98% 的時間在等同步。
   索引改成建構期就轉好的 Python int → **25.8 ms（20×）**，且 R3 門檻仍 PASS。
2. **`torch.distributions.Normal` 的 `validate_args` 檢查佔 17.8 ms/step = 426 ms/rollout**，
   比它後面的 frozen actor forward（6.2 ms）還貴。關掉並快取 std → 省 ~0.5 s/iteration。

### 偏差 D：env slot → body 改為固定指派

本文件未提。MuJoCo 的 sim 綁定單一 `MjModel`，若讓 env slot 每次 reset 換身材，每個 worker
得為 10 具身材各持一份 `MjModel`/`MjData` 並在 reset 時切換。改成
`body_assignment[i] = (i * n_bodies) // n_envs`（連續指派）後，實測 32 個 worker 中只有
**8 個**需要碰超過一具身體。副產品：每具身體擁有連續的 slot 區間，ES 的 per-body 擾動組
直接就是一個 slice。

### 已實作的檔案

`model/bilevel/` 全部落地並跑通：`config.py`、`quat.py`、`torch_kin.py`、`retarget.py`、
`semantics.py`、`data.py`、`policy.py`、`rewards.py`、`rollout.py`、`ppo.py`、`upper.py`、
`train_bilevel.py`、`sim/{protocol,worker,pool}.py`、
`tests/{test_torch_kin,test_retarget,test_simpool}.py`。
新增 `scripts/audit_bodies.py`、`scripts/regen_bodies.py`。
`model/networks.py` 加了 `project_z` 旗標（預設 False，legacy 路徑逐位元不變）、
`RootWrenchHead`、`ValueNet`。`model/train.py` 等 legacy baseline 未動。

`eval_bilevel.py`（held-out 評估，`f_max=0`、整段 clip、沿用 `losses.*`）尚未實作 —— 它是
Stage 3 跑完才用得到。

---

## 10. 開放問題 — 結論

| # | 問題 | 結論 |
|---|---|---|
| 1 | 訓練身材名單；`"split": "val"` 這個沒人讀的欄位怎麼辦 | **已定案。** 名單由 `scripts/audit_bodies.py` 決定（10 train / 2 test / 1 unused）。`"split"` 由 `scripts/write_body_splits.py` 寫成權威值，`data.load_body` 啟動時比對 config，不一致就報錯 —— 死欄位變活，且無法再靜默漂移 |
| 2 | 是否重生資產並開 actuator scaling | **已定案，做法不同。** `scale_robot.py` 的預測模型實測無效；改用 `scripts/calibrate_actuators.py` 的實測校準，輸出 `assets/robots_calib/`，原始 `assets/robots/` 未動，legacy baseline 仍可重現。細節見 §11 偏差 A |
| 3 | Held-out 維度 | **已定案：兩者交叉。** `eval_bilevel.py` 回報**四個象限**（見下） |
| 4 | `move-ego--90-2` 多出的 990 個 z | **自動解決。** `WindowDataset` 以 `origin_motion` 為主鍵，沒有對應 motion 的 z 直接忽略。無須補跑也無須刪除 |
| 5 | `skeleton.json` 只有 2 具有 | **已定案：不再需要。** 改讀 `MjModel.qpos0[2]`（`test_torch_kin.py` 驗證與 skeleton.json 完全相同），少掉一個中介檔案格式 |

### Q3 的落地：四象限評估

`model/bilevel/eval_bilevel.py`。身材軸走 `BilevelConfig.train_bodies`／`heldout_bodies`，
task 軸沿用既有的 `splits/{train,test}_tasks.txt`（與 `model/evaluate.py`／`baseline.py` 同一組，
數字才與 `outputs/{baseline,eval}/report.json` 可比）：

| 象限 | 量測的東西 |
|---|---|
| seen body × seen task | 訓練分布 —— 只是地板，不是成績 |
| seen body × unseen task | 跨**動作**的泛化 |
| unseen body × seen task | 跨**形態**的泛化 |
| **unseen body × unseen task** | 兩者同時 —— **這才是頭條數字** |

兩個與訓練刻意不同的地方：`f_max = 0`（外力是訓練拐杖，不能拿它來計分）、
rollout 拉長到數百步（24 步視野看不出 policy 是不是在把局部好、全局不連貫的動作拼起來，R6）。

評分用**未改動的** `losses.functional_equivalence` 與 `losses.physics_penalty`，
並且 **D 對兩個參考各報一次**：

- `D_p` —— 對 upper level 產生的 p 參考
- `D_0` —— 對 naive p=0 參考

這就是 §3.5 的決定性反退化檢驗。`D_p` 降而 `D_0` 不降 = upper level 靠移動參考關掉差距，
不是機器人變好。腳本會直接印警告。

---

## 12. 移除 pre-baked retargeted motion

`new.md` 要求「retargeting motion 是 runtime 取得的」，所以預先烘焙的參考動作不只是冗餘，
而是這個設計要取代的東西本身。已移除：

| 項目 | 內容 |
|---|---|
| `data/retargeted_motion/` | 540 個 `.npz`，78 MB（gitignore，純衍生物） |
| `datasets/.../retargeted_motion/` | 540 個 `.npz`，78 MB（**git 追蹤中**，可由 git 還原） |
| `manifest.jsonl` 的 `retargeted_motion` 欄位 | 1530 列全部移除，剩 `id / reward_name / trial / origin_z / morphology / morphology_label` |

`model/bilevel/` 完全不受影響 —— 它從來沒有讀過這些檔案。

**對 legacy 路徑的實際後果（已明確標註，不讓它靜默塌掉）**：`model/dataset.py` 的
`qpos_ref` 現在恆為 `None`，因此 `functional_equivalence` 回傳 0.0，
`model/train.py` 目標函數中的 **D 項恆等於零**，實際被最佳化的只剩
`λ_z‖z_β−z0‖² + λ_phys·L_phys`。

這一點特別容易漏看 —— 原本 1530 列就有 990 列 `retargeted_motion: null`，D=0 早就是常態。
因此 `CrossEmbodimentDataset` 建構時會**主動印出警告**，`model/train.py` 與
`scripts/build_dataset.py` 的 docstring 也都標了。要跑回原本的 baseline，兩行指令可重生：

```bash
uv run scripts/qpos_retarget.py --input_dir data/origin_motion \
    --output_dir data/retargeted_motion --target_xml assets/robots/child/robot.xml
uv run scripts/build_dataset.py
```

`scripts/qpos_retarget.py` 刻意保留 —— 它是 p=0 的權威定義，
`tests/test_retarget.py` 的錨點斷言以它為準。

---

## 附錄：本文件引用的實測命令

```bash
# §1.2 關節限位違反率
./.venv/bin/python -c "
import mujoco, numpy as np, glob
m = mujoco.MjModel.from_xml_path('assets/robots/child/robot.xml')
lo, hi = m.jnt_range[1:,0], m.jnt_range[1:,1]
tot_f=bad_f=tot_e=bad_e=0
for f in sorted(glob.glob('data/origin_motion/*/*.npz')):
    q = np.load(f)['qpos'][:,7:]; v = (q<lo)|(q>hi)
    tot_f += q.shape[0]; bad_f += int(v.any(1).sum())
    tot_e += v.size;     bad_e += int(v.sum())
print(bad_f/tot_f, bad_e/tot_e)"

# §1.3 ctrl ⟺ qpos 仿射恆等式
./.venv/bin/python -c "
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path('assets/robots/child/robot.xml')
g, b0, b1 = m.actuator_gainprm[:,0], m.actuator_biasprm[:,0], m.actuator_biasprm[:,1]
jid = np.array([m.actuator_trnid[i,0] for i in range(m.nu)])
print(np.abs(-(g*m.actuator_ctrlrange[:,0]+b0)/b1 - m.jnt_range[jid,0]).max())"

# §6.7 單核 30 Hz control step 成本
./.venv/bin/python -c "
import mujoco, numpy as np, time
m = mujoco.MjModel.from_xml_path('assets/robots/child/robot.xml'); m.opt.timestep = 1/450
d = mujoco.MjData(m); mujoco.mj_forward(m,d)
for _ in range(50): mujoco.mj_step(m,d,nstep=15)
t0 = time.perf_counter()
for _ in range(300):
    d.ctrl[:] = np.random.uniform(-.2,.2,m.nu); mujoco.mj_step(m,d,nstep=15)
print((time.perf_counter()-t0)/300*1000, 'ms')"

# §8.1 逐身體 actuator 稽核
./.venv/bin/python -c "
import mujoco, json
from pathlib import Path
for p in sorted(Path('assets/robots').iterdir()):
    if not (p/'robot.xml').exists(): continue
    m = mujoco.MjModel.from_xml_path(str(p/'robot.xml'))
    print(p.name, m.body_mass.sum(), m.actuator_gainprm[0,0], m.actuator_forcerange[0,1])"
```
