# Cross-Embodiment Bilevel Training

讓一個**在成人身體上訓練好、且完全凍結**的動作 AI，去驅動 11 具體型完全不同的人形機器人
（小孩 38 kg、巨人 110 kg、矮胖 130 kg …），做出同一批動作。

核心想法是**兩層同時學**：

- **下層**學「怎麼在這個身體裡動」
- **上層**學「這個動作在這個身體上，正確的樣子應該長怎樣」

完整設計與所有取捨的理由在 [`docs/proposal.md`](docs/proposal.md)（約 1500 行）。

---

## 1. 問題

我們有一個預訓練好的 **Meta Motivo (FB-CPR)** 策略。它很強，但它只認得**成人**的身體：

```
觀測 obs (358) ─┐
                ├─► frozen actor ─► 69 個關節指令
意圖 z   (256) ─┘
```

如果直接把它接到小孩身上，動作會壞掉 —— 腿短了、質量分布變了、馬達力量也不一樣。

**而且參考動作本身就有問題。** 540 段動作是在成人身上錄的，要當小孩的目標得先「換算」
（retargeting）。原本的換算只有一個數字：根部座標乘身高比、關節角直接照抄。實測結果是
**p=0 的參考動作在 88% 的影格陷進地板 12–16 mm** —— 機器人一開始就被塞在地面裡。

所以問題不只是「策略不會動」，而是**連目標本身都是錯的**。

---

## 2. 三個角色

| 角色 | 是誰 | 動不動 |
|---|---|---|
| **source（來源）** | `adult` —— 動作原本錄在誰身上 | 固定 |
| **target（目標）** | `child`, `teen`, `giant`, … 共 11 具 | 固定 |
| **frozen actor** | Meta Motivo 本體 | **完全凍結，一個參數都不更新** |
| **上層 p** | `RetargetNet`：把 β 映射成 36 個 retarget 參數 | 學 |
| **下層 φ** | `LatentAdapter` + `ActionHead` + `RootWrenchHead` | 學 |

`β` 是 8 維的**身材描述向量**（腿/手/軀幹/頭 的長度倍率與粗細倍率）。

---

## 3. 架構與資料流

```
                             p = RetargetNet(β)  ──► u ∈ R³⁶ ──► hard tanh box
                                                                      │
 (clip m, body β)                                                     ▼
      │                                                    apply_retarget
      ├── src_qpos (成人動作) ──────────────────────────────────┬──► ref_raw  (未夾, 可微) ──► 可行性懲罰 C
      │                                                        └──► ref = clamp(ref_raw)  ──► RSI + 追蹤獎勵
      │                                                                    │
      │                                                       ┌────────────┘
      │                                                       ▼
      └── z0 ──► LatentAdapter(β) ──► z_β ──► frozen actor ──► ActionHead ──► ctrl (69)
                                                          └──► RootWrenchHead ──► wrench (6)
                                                                    │
                            RSI 從 ref[0] 起始 ──► SimPool (32 進程 × 8 模擬 = 256) ──► 軌跡 τ
```

一次迭代 = **256 個 window × 24 步 @ 30 Hz**（每個 window 0.8 秒），約 0.8–1.0 秒。

---

## 4. 數學

### 4.1 上層目標 `F(p)`

上層要找一個「忠實、且這具身體做得到」的參考動作：

```
F(p) = λ_gap  · G(τ, ref_p)      機器人實際做出來的 vs 修正後參考   的差距
     + λ_fid  · S(ref_p, src)    修正後參考       vs 原始成人動作   的偏離   ← 防退化主力
     + λ_feas · C(ref_raw, β)    參考在這具身體上合不合法（超限/穿地/腳滑）
     + λ_ext  · E_ext(τ)         外力作弊懲罰
     + λ_phys · P(τ)             機器人物理合理性
     + λ_prox · ‖u − u_prev‖²    proximal（別一步走太遠）

λ = (1.0, 3.0, 2.0, 0.5, 0.5, 1.0)
```

**退化問題**：只看 `G` 的話，上層有一個作弊解 —— 不改善機器人，直接**把參考動作搬到機器人身上**，
差距一樣歸零。整份設計最大的篇幅在防這件事，三道防線：

1. **硬 tanh box（結構防禦，不靠調權重）**

   ```
   log_s_root = log(h_tgt/h_src)·1₃ + 0.15·tanh(u[0:3])    根部縮放,  ±16%
   dz_root    =                       0.08·tanh(u[3])       根部升降,  ±8 cm
   w_root     =                       0.15·tanh(u[4:7])     根部姿態,  ±8.6°
   g_joint    = 1.0                 + 0.20·tanh(u[7:21])    14 組關節振幅, ±20%
   b_joint    =                       0.15·tanh(u[21:35])   14 組關節偏移, ±0.15 rad
   log_tau    =                  log(1.25)·tanh(u[35])      時間伸縮（Stage 3 前凍結）
   ```

   36 個**全域**數字要覆蓋 540 段動作 × 9 具身體，容量上不可能塌成站姿。
   而且 **p = 0 精確重現原本的 naive retarget** —— 它是架構原點，不是靠權重維持的軟目標。

2. **`λ_fid : λ_gap = 3 : 1`（比例可解釋，不是盲調）**

   `G` 與 `S` 的穩定點是收縮估計 `ref* = (λ_gap·τ + λ_fid·src)/(λ_gap+λ_fid)`，
   代入即 **參考最多只能往機器人漂移 25%**。

3. **診斷**：`u_saturation = mean(|tanh(u)| > 0.9)`，超過 0.2 就是退化警報。
   決定性檢驗：**`G(τ, ref_0)` 必須跟著 `G(τ, ref_p)` 一起降** —— 只有後者降就是在作弊。

### 4.2 雙層形式與 hypergradient

```
下層:  φ*(p) ∈ argmax_φ  J(φ; p) = E[ Σ_{t<24} γᵗ r(s_t, a_t; ref_p) ]
上層:  min_p  F(p, φ*(p))
```

耦合是**雙通道**的：p 同時進入 reward **和初始狀態分布**（RSI 從 `ref_p[t0]` 起始）。

```
dF/dp = ∂F/∂ref · ∂ref/∂p              [T1]  autograd 精確，零變異 —— 訊號主體
      + ∂F/∂τ · ∂τ/∂ref · ∂ref/∂p      [T2]  模擬器不可微 → 反對稱 ES 估計
      + ∂F/∂τ · ∂τ/∂φ · dφ*/dp         [T3]  捨棄
```

- **T1 保留**：這就是「retargeting 用 torch 重寫成可微分」的價值所在。
  但 `E_ext` 與 `P` 的 T1 **恆為零**（它們只透過模擬器依賴 p）→ 純 T1 看不見外力作弊，這是 ES 要補的洞。
- **T3 捨棄**：需要在接觸不連續的 MDP 上做 Hessian-inverse-vector product，算不出來。
  **誠實的但書**：捨棄 T3 後解的**不是** bilevel 問題，而是 `min_p F(p, φ_k)` 的 Gauss-Seidel 交替。
  代價是 p 沒有動機去挑「現在難、但能養出更好策略」的參考 —— 對這個應用是特性不是缺陷。

**TTSA（兩個時標）**：`η_φ = 3e-4`、`η_p = 1e-5`（30:1），上層每 `K = 10` 輪才更新一次（→ 有效 300:1），
外加 proximal 項與每步 `‖Δu‖_∞ ≤ 0.02` 的硬信賴域。

### 4.3 T2 怎麼估：ES（Evolution Strategy，演化策略）

**ES 是一種不用梯度的梯度估計法**：在參數上加隨機擾動，看目標函數變好還是變壞，反推該往哪走。

用它是因為 T2 裡的 `∂τ/∂ref`（「參考動一點點，模擬出來的軌跡會怎麼變」）需要對 MuJoCo 微分，
而 MuJoCo 不可微。而 T2 又不能忽略，有兩個理由：

- **RSI 通道**：每個 window 的初始狀態**字面上就是** `ref_p[t0]`。沒有 T2 的上層不知道
  「移動參考也會移動機器人每次的起點」。
- **外力作弊看不見**：`E_ext` 與 `P` 只透過模擬器依賴 p，它們的 T1 恆為零。

作法 —— **擾動 p 的「輸出」`u ∈ R³⁶`，不是它的 ~10k 個權重**（36 維才讓 ES 可行）：

```
每 iteration：10 具身體 × 2 對反對稱 = 40 組擾動 × 6 windows = 240 個
              剩下 16 個是未擾動的對照組（給乾淨的 logging 與 T1）
σ_p = 0.02（pre-tanh 空間，約 0.4% gain 變化、0.003 rad 偏移變化）

              1
  ĝ_u  =  ───────── Σ_g [ F̂_sim(u + σ_p ε_g) − F̂_sim(u − σ_p ε_g) ] · ε_g
           2 σ_p G_b

  F_sim = λ_gap·G + λ_ext·E_ext + λ_phys·P        只含依賴模擬器的部分
```

然後 `u.backward(gradient=ĝ_u)` 接回 `RetargetNet` 的權重 —— 對 autograd 而言 `ĝ_u`
就只是一個從外面塞進來的上游梯度。

| 細節 | 說明 |
|---|---|
| **antithetic（反對稱）** | `+ε` 與 `−ε` 成對測試，共同噪音互相抵消，估計變異大幅下降 |
| **成本** | **零額外模擬**。那 240 個 window 本來就要跑，只是把其中一些的 `u` 換掉 |
| **rank normalization** | 40 個 `ΔF_sim` 換成名次再加權，讓估計不受 `F_sim` 尺度漂移影響 |
| **累積 K=10 輪再更新** | 與 TTSA 的 cadence 刻意設成同一個 K |

> ⚠️ **CRN（Common Random Numbers）是必要條件，不是優化。**
> 同一對反對稱的兩次模擬，除了 `±σ_p ε` 之外，`(哪段動作、window 起始幀、RSI 關節噪音、
> 整條 (24,75) 動作噪音)` 必須**逐位元相同**。ES 的訊號在 `G` 上只有約 10% 的相對變化，
> 沒有 CRN 就會被 rollout 噪音完全淹沒 —— **加了比不加還糟**。
> 驗證方法：令 `σ_p = 0` 的一對必須產生逐位元相同的 `qpos` 軌跡。

**為什麼只在 Stage 3 開**：ES 貴在變異不在算力。Stage 1/2 外力拐杖常開，`E_ext` 那個洞還不會被利用；
Stage 3 一旦開始把外力退火到零，上層就有動機作弊了 —— 所以**外力退火與開 ES 必須同時發生**。

### 4.4 下層 per-step reward

全部在 worker 內從 `data.qpos / qvel / xpos / actuator_force / contact` 加參考幀算出。

```
r_t = 0.65 · r_track + 0.15 · r_reg + 0.20 · r_surv          有界於 [0,1]
```

**追蹤**（DeepMimic 式指數核，有界、對 value net 尺度友善）：

```
r_pose = exp(−2.0   · (1/69) Σ_j (q_j − q̂_j)² )
r_vel  = exp(−0.005 · (1/69) Σ_j (q̇_j − q̂̇_j)² )
r_ee   = exp(−40    · (1/5)  Σ_e ‖(p_e − p_root) − (p̂_e − p̂_root)‖² )
r_root = exp(−10    · ( ‖p_root − p̂_root‖² + 0.5·‖log(q̂_root⁻¹ ⊗ q_root)‖² ) )
r_com  = exp(−10    · ‖com_xy − ĉom_xy‖² )

r_track = 0.35 r_pose + 0.15 r_vel + 0.25 r_ee + 0.15 r_root + 0.10 r_com
```

> `r_pose` 用**原始弧度**，不除以 joint range —— 那 12 個 range 只有 0.16–0.30 rad 的
> 近乎鎖死關節一旦正規化就會主宰誤差，但它們對視覺幾乎沒貢獻。

**正則**（單一指數包住加權懲罰和）：

```
e_act    = (1/69)‖a‖²                                   動作大小
e_smooth = (1/69)‖a_t − a_{t−1}‖²                       抖動
e_res    = (1/69)‖a − raw_prior‖²                       別離凍結 AI 太遠
e_tau    = (1/69) Σ_j (力矩_j / forcerange_j)²           馬達吃力程度
e_ext    = ‖f‖²/(Mg)² + ‖m‖²/(Mg·L_leg)²                外力大小
e_slip   = Σ_{接觸中的腳} ‖v_foot,xy‖²                    腳滑

r_reg = exp( −( 0.1 e_act + 1.0 e_smooth + 0.5 e_res + 0.5 e_tau + 8.0 e_ext + 2.0 e_slip ) )
```

`e_ext` 的 **8.0 是全場最大** —— 外力是滿足其他所有項最便宜的路徑，必須讓它貴。

**存活**（`r_surv ∈ {0,1}`，判 0 就終止該 window）：

```
alive =  root_z    > 0.5 · ẑ_root,t                              沒垮下去
      ∧  up_z      > ûp_z − 0.8                                  沒比參考翻得更倒
      ∧  ‖p_root − p̂_root‖ < 0.75 · (L_leg(β)/L_leg(adult))      沒飄走
      ∧  (1/69) Σ_j (q_j − q̂_j)² < 0.6                           沒追丟
```

> 兩個容易踩的坑：
> **(1)** `up_z = 2(q_y q_z + q_w q_x)`，**不是**教科書的 `1 − 2(qx²+qy²)` ——
> 這個資產的 Pelvis 帶 `euler="90 0 0"`。
> **(2)** 直立判定是**相對參考姿態**的。資料集裡 21% 的影格本來就不直立（倒立、爬行、躺著），
> 絕對門檻會把「正確做出倒立」判成跌倒。

### 4.5 下層 PPO 總 loss

```
L = L_clip  +  0.5 · L_value  +  λ_z · mean(1 − cos(z_β, z0))  +  λ_bc(k) · mean‖μ_ctrl − a_ref‖²
```

| 項 | 作用 |
|---|---|
| `L_clip` | 標準 PPO clipped surrogate，`ε = 0.2`，advantage 用 GAE(γ=0.97, λ=0.95) |
| `L_value` | value 迴歸，目標做 running 正規化（PopArt-lite）|
| `λ_z = 0.1` | 把 `z_β` 錨在 `z0` 附近。用 **cosine** 而非歐氏 —— `z` 已投影到半徑 16 的球面上 |
| `λ_bc` | **行為複製**，`100 → 0` cosine 退火於 iter 0–2000 |

**BC 的目標是封閉解**，這是本專案第二個關鍵發現：

```
a_ref,t = clip( 2·(q̂_{t+1} − lo)/(hi − lo) − 1 , −1, 1 )
```

因為致動器是 affine 位置伺服，`ctrl ∈ [−1,1] ⟺ qpos ∈ jnt_range` **精確成立**（實測誤差 4.4e-16）。
所以「能命令出參考關節角的那個 ctrl」是免費、稠密、零變異的監督訊號，條件數遠優於 PPO 產出的任何東西。
**但必須退火到零** —— 長期開著會與凍結 AI 對抗，而且它忽略動力學（讓伺服**停在** `q̂`，
不等於在負載下**驅動你到達** `q̂`）。

---

## 5. 訓練流程

每個 stage 只引入**一個**新的困難來源，這樣壞掉時知道是誰弄壞的。每個 stage 用前一個的通過作為前提。

| Stage | 身材 | p | 外力拐杖 | ES | 通過門檻 |
|---|---|---|---|---|---|
| **1** 下層承重 | 2 具（child, teen）| 只解凍 `dz_root` | 常開 | 關 | `r_track > 0.6` 且 `term_rate < 0.3` |
| **2** 上層開動 | 9 具 | 全部 36 維 | 常開 | 關 | `ref_min_z ≥ −5 mm`、`u_saturation < 0.2`、`S < 2`，**且 `G(τ,ref_0)` 也要降** |
| **3** 抽拐杖 | 9 具 | 全部 36 維 | **退火到 0** | **開** | 這就是正式全量跑 |

**Stage 1 為什麼要解凍 `dz_root`**：p 全凍時這階段贏不了。`p=0` 的參考陷在地板裡 12–16 mm
（因為原本逐幀的 `ground_correct_qpos` 被換成了可學的 `dz_root`），RSI 每次都把機器人塞進地面，
物理再把它彈出來，然後它就倒了。所以 Stage 1 拿那 1 個修復破碎前提的維度，Stage 2 拿其餘 35 個。

**Stage 3 為什麼外力退火和 ES 必須同時做**：`E_ext` 的 T1 恆為零，所以精確梯度**看不見**外力作弊。
只退火不開 ES，上層就可以放心把參考弄難、讓下層用外力硬撐。

### 指令

```bash
# 清掉舊產物
rm -f model/bilevel/checkpoints/stage*.pt outputs/bilevel_logs/stage*_metrics.jsonl

# Stage 1
python -m model.bilevel.train_bilevel --stage 1 --iters 2000 --device cuda:1 \
    --wandb --wandb-group bilevel-run1 --wandb-name stage1

# Stage 2（接力，用 --init-from 不是 --resume）
python -m model.bilevel.train_bilevel --stage 2 --iters 3000 --device cuda:1 \
    --init-from model/bilevel/checkpoints/stage1_002000.pt \
    --wandb --wandb-group bilevel-run1 --wandb-name stage2

# Stage 3（正式全量）
python -m model.bilevel.train_bilevel --stage 3 --iters 10000 --device cuda:1 \
    --init-from model/bilevel/checkpoints/stage2_003000.pt \
    --wandb --wandb-group bilevel-run1 --wandb-name stage3
```

- `--init-from`：跨階段接力，**只載網路權重**，optimizer 與排程重來
- `--resume <明確路徑>`：同階段續跑，連 Adam moment、RNG、advantage EMA 全載
- **`--resume` 不要用裸的（auto）** —— 它按檔名取最大號，可能挑到別次跑的 checkpoint

時間：Stage 1 約 25 分鐘、Stage 2 約 1 小時、Stage 3 約 3–4 小時（32 實體核 + 1 GPU）。

---

## 6. 怎麼看訓練有沒有在進步

W&B 上分五個面板（前綴決定畫在哪張圖）：

| 面板 | 最該盯的 | 健康值 |
|---|---|---|
| `lower/` | `r_track`、`term_rate` | 上升 / 下降 |
| `lower/` | `kl`、`clipfrac` | 0.01–0.05 / 0.05–0.3。**爆掉代表 PPO 步伐失控** |
| `upper/` | `ref_min_z` | ≥ −0.005 m（參考不再穿地）|
| `upper/` | `u_saturation` | **< 0.2，否則是退化警報** |
| `long/` | `survive` | 299 步連續 rollout 的存活比例 |
| `sched/` | `wrench_scale`、`lambda_bc` | 確認拐杖真的有被拿掉 |

**`long/` 是最誠實的那個面板。** 訓練每個 24 步 window 都會從參考重新 RSI，誤差不累積；
真正做一整段動作時沒有這個重置。所以每 100 iteration 會跑一次 **299 步（10 秒）、`f_max = 0`** 的
連續 rollout（[`model/bilevel/longeval.py`](model/bilevel/longeval.py)）。成本 < 0.03 秒/iter。

---

## 7. 怎麼驗收

### 看影片（最直接）

```bash
python scripts/rollout_video.py \
    --ckpt model/bilevel/checkpoints/stage3_010000.pt \
    --task move-ego-0-2 --body child
```

輸出左右對照的 mp4（左：參考動作，右：物理機器人）到 `outputs/rollout_video/`。
`--list-tasks` 列出可用動作、`--wrench 1` 開回外力看它有多依賴拐杖、`--steps N` 限制長度。

用的是與訓練 `long/` 指標**完全同一個** `rollout_one()`，所以影片下的數字和曲線不會對不上。

### 跑完整評估

```bash
python model/bilevel/eval_bilevel.py --ckpt model/bilevel/checkpoints/stage3_010000.pt
```

四象限、299 步、`f_max = 0`：

| | 見過的 task | 沒見過的 task |
|---|---|---|
| **見過的身材** | 訓練分布，理智下限 | 跨動作泛化 |
| **沒見過的身材** | 跨體型泛化 | **兩個一起 —— 誠實的頭條數字** |

評分用**未改動的** `model/losses.py`，所以數字與舊 baseline（`outputs/{baseline,eval}/report.json`）直接可比。
且 `D` 對兩個參考各報一次（`D_p` 與 `D_0`），這就是 §4.1 那個防作弊的決定性檢驗。

---

## 8. 檔案地圖

```
model/bilevel/
  config.py         所有超參數，一處。每個非顯然的數字都附實測理由
  data.py           WindowDataset：540 段動作全載 RAM，抽樣 (clip, body, t0)
  torch_kin.py      可微分批次 FK。硬性要求：對 mj_kinematics 誤差 < 1e-5
  retarget.py       RetargetNet + 硬 tanh box + apply_retarget
  semantics.py      S / C / G 三個上層目標分項
  policy.py         凍結 FB-CPR + LatentAdapter + ActionHead + RootWrenchHead
  rewards.py        純 numpy（絕不 import torch）。worker 用，回傳未加權的 14 維
  sim/              SimPool：32 進程 × 8 模擬，共享記憶體，spin-then-yield
  rollout.py        驅動 SimPool 24 步 + GPU 前向，持有 ES 需要的 CRN 噪音
  ppo.py            GAE、clipped surrogate、per-(clip,body) advantage 正規化、λ_z、BC
  upper.py          F(p) 組裝、T1 精確路徑、T2 的 ES 路徑、信賴域
  longeval.py       週期性長 rollout 診斷（proposal R6）
  train_bilevel.py  TTSA 主迴圈 ← 入口
  eval_bilevel.py   四象限 held-out 評估

scripts/
  rollout_video.py       單段動作的左右對照影片 ← 驗收用這個
  calibrate_actuators.py 按實測力矩需求校準馬達（沒這步只有 8/13 具站得住）
  scale_robot.py         由 β 生成身體 XML

docs/proposal.md         完整設計。§0.6 有 87 條名詞解釋，§11 是實作後的實測修正
```

**保留為 baseline、不改造**：`model/train.py`、`train_explore.py`、`evaluate.py`、`baseline.py`。
它們的 docstring 記錄了付過代價的發現（為何 `exploration_std=0.05` 而非 0.2、為何拿掉 `R_task`、
為何 `total_loss` 不是進度訊號）。新系統必須贏過它們。
