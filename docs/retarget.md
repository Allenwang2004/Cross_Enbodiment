# `apply_retarget` 是怎麼做的

程式碼：[`model/bilevel/retarget.py`](../model/bilevel/retarget.py)

---

## 一句話

**拿成人的動作，用 36 個數字改一改，變成小孩的動作。**

那 36 個數字由一個小網路根據身材算出來，而且**整段動作從頭到尾都用同一組**。

---

## 1. 輸入和輸出

```
輸入：src_qpos   (B, T, 76)    成人的動作
      36 個參數                 由身材 β 決定

輸出：ref_raw    (B, T, 76)    小孩的參考動作（未夾）
      ref        (B, T, 76)    小孩的參考動作（夾進關節限位）
```

`B` = 幾段動作同時處理，`T` = 幾幀。

**76 是什麼**：MuJoCo 描述這個人形的完整姿勢需要 76 個數字。

```
qpos = [ 骨盆位置 (3) | 骨盆朝向 (4) | 關節角度 (69) ]
         x, y, z        四元數         69 個關節
```

---

## 2. 三個步驟

整個函式就是對這三塊各做一件事。

### 步驟 A — 骨盆位置：縮小，然後抬高

```python
out_pos = root_pos * scale + [0, 0, dz]
```

| 參數 | 幾個數字 | 做什麼 |
|---|---|---|
| `scale` | 3 | 把整條軌跡按身高比縮小 |
| `dz` | 1 | 整體垂直抬高（把穿地補起來）|

小孩骨盆靜止高度 0.5846 m、成人 0.9567 m，所以 `scale = 0.611`。

實測第 0 幀：

```
成人  [-0.0159,  0.0222,  0.9395]
  ↓  × 0.611
小孩  [-0.0097,  0.0136,  0.5741]
```

腿短的人走同一段路，步幅本來就該按比例縮小 —— 這就是那件事。

### 步驟 B — 骨盆朝向：轉一個小角度

```python
out_quat = normalize(quat_mul(root_quat, exp_map(root_rot)))
```

| 參數 | 幾個數字 | 做什麼 |
|---|---|---|
| `root_rot` | 3 | 整段動作的軀幹前傾／側傾修正，最多 ±8.6° |

用**右乘**是有意的：右乘等於「在骨盆自己的座標系裡轉」。這樣同一個修正在朝北走和朝南走的 clip 上意思一樣。左乘會變成世界座標系的旋轉，那就跟 clip 剛好面向哪邊綁在一起了。

### 步驟 C — 69 個關節角：放大／縮小，然後平移

```python
out_hinge = hinge * gain + bias
```

| 參數 | 幾個數字 | 做什麼 |
|---|---|---|
| `gain` | 14 | 動作幅度放大或縮小，最多 ±20% |
| `bias` | 14 | 整體偏移一個固定角度，最多 ±0.15 rad（8.6°）|

**為什麼是 14 而不是 69**：69 個關節被歸成 14 組，每組共用一個數字。

```
Hip  Knee  Ankle  Toe  Thorax  Shoulder  Elbow  Wrist  Hand   ← 9 組，左右各 3 軸 = 6
Torso  Spine  Chest  Neck  Head                              ← 5 組，單一 3 軸 = 3

9 × 6 + 5 × 3 = 69
```

分組是從關節名字自動推出來的，不是手寫清單：

```
"L_Shoulder_x"  →  去掉軸  "L_Shoulder"  →  去掉 L_/R_  →  "Shoulder"
```

**左右共用同一組**，所以「左手學一套、右手學另一套」這種事在架構上做不出來 —— 13 具身體全都左右對稱，一個左右不對稱的全域修正永遠不可能是合法的身材適應。

---

## 3. 最重要的性質：**沒有時間軸**

看形狀怎麼對上：

```
源動作   (B,  T,  69)      ← 有 T 幀
參數     (B,  1,  69)      ← 只有 1，在時間軸上廣播
```

也就是：

> **同一個修正被一模一樣地套到每一幀上。**

`gain[Knee] = 1.15` 的意思是「這具身體的膝蓋，**整段動作**振幅放大 15%」，不是「第 87 幀的膝蓋放大 15%」。

這是整個防退化設計的地基。上層唯一的可微梯度方向是「把參考搬到機器人身上」—— 如果它能逐幀微調，一步就能作弊成功。但 36 個全域數字要同時覆蓋 430 段動作 × 9 具身體 × 每一幀，**容量上不可能**把它們一起塌成站姿。

---

## 4. 那 36 個數字從哪來

```
身材 β (8)  →  RetargetNet  →  u (36)  →  硬 tanh box  →  6 個具名參數
               7076 個參數      pre-activation
```

`RetargetNet` 是個三層 MLP（`8 → 64 → 64 → 36`）。

**它刻意只吃 β。** 沒有 clip 編號、沒有時間、沒有機器人當前狀態。加任何一個都會讓上面那個容量論證失效。

### 硬 tanh box

```python
t = tanh(u)                                # 永遠落在 (-1, 1)

scale = exp(log(h_tgt/h_src) + 0.15 * t[0:3])     # ±16%
dz    =                        0.08 * t[3]        # ±8 cm
rot   =                        0.15 * t[4:7]      # ±8.6°
gain  = 1.0                  + 0.20 * t[7:21]     # ±20%
bias  =                        0.15 * t[21:35]    # ±0.15 rad
```

`t` 被 tanh 鎖在 (−1, 1)，所以每個參數**在結構上就出不了那個範圍**。不是「超出去會被罰」，是「做不到」。

**為什麼用 `tanh` 不用 `clamp`**：`clamp` 在界外的梯度是 0，一旦某維被推出界就再也回不來。`tanh` 永遠有梯度，只是靠近邊界時愈來愈小 —— 推不出去，但不會斷線。

### `u = 0` 的特殊地位

`RetargetNet` 的輸出層**權重和 bias 都零初始化**，所以訓練第 0 步 `u ≡ 0`。代入上面的式子：

```
scale = exp(log(h_tgt/h_src) + 0) = h_tgt/h_src
dz = 0,  rot = 0,  gain = 1,  bias = 0
```

整個函式化簡成：

```
ref[0:3] = src[0:3] * (h_tgt / h_src)
ref[3:]  = src[3:]                        ← 朝向和關節角原封不動
```

**這正好是舊的 `scripts/qpos_retarget.py:91 retarget_qpos`。**

所以「最樸素的換算」是這個架構的**原點**，不是一個靠調權重維持的軟目標。訓練從那裡出發，而且不管權重怎麼變，`u=0` 永遠會退回它（`tests/test_retarget.py` 明確斷言）。

---

## 5. 為什麼回傳兩個張量

```python
ref_raw = concat(out_pos, out_quat, out_hinge)   # 未夾
ref     = clamp_hinges(ref_raw)                  # 夾進 jnt_range，只夾關節
```

| | 給誰用 | 為什麼 |
|---|---|---|
| `ref_raw` | 可行性懲罰 `C` | 界外**仍有活的梯度** |
| `ref` | RSI、追蹤獎勵 | 物理上合法的目標 |

如果只回傳夾過的版本，`torch.clamp` 在界外的梯度是 0，會**正好殺掉**「把超限角度拉回合法範圍」這個唯一需要的梯度。

實測：某段 clip 前 5 幀、`u=0` 時，clamp 動了 **10 / 345** 個 (幀, 關節) —— 約 2.9%。

---

## 6. 第 36 個數字：時間伸縮（目前關閉）

前 35 個都是「改數值」，只有最後一個 `tau` 是**改取樣位置**：

```python
輸出第 k 幀  ←  取自源動作的第 k·tau 幀（線性內插）
```

`tau = 1.25` 表示參考動作播慢 25%。小孩腿短、步頻本來就跟成人不同，強迫他用成人的節奏本身就是錯的。

實作上刻意讓 `tau == 1` 時插值權重**恰好是 0**，所以 p=0 的路徑和純切片逐位元相同。

目前 `enable_time_warp = False`，`tau` 永遠是 1，要到 Stage 3 才考慮解凍。

---

## 7. 完整程式碼（去掉錯誤處理）

```python
def apply_retarget(src_qpos, params, kin, group_idx, n_out=None):
    # 0. 時間伸縮（tau=1 時等同純切片）
    if any(params["tau"] != 1.0):
        src_qpos = _resample_time(src_qpos, params["tau"], n_out)
    else:
        src_qpos = src_qpos[:, :n_out]

    root_pos  = src_qpos[..., 0:3]
    root_quat = src_qpos[..., 3:7]
    hinge     = src_qpos[..., 7:]

    # A. 骨盆位置：縮放 + 抬高
    scale = params["root_scale"].unsqueeze(1)          # (B,3) → (B,1,3)
    dz    = params["root_dz"].reshape(-1, 1, 1)
    out_pos = root_pos * scale + cat([0, 0, dz])

    # B. 骨盆朝向：在自己的座標系裡轉一個小角度
    drot = exp_map_to_quat(params["root_rot"]).unsqueeze(1)
    out_quat = normalize(quat_mul(root_quat, drot))

    # C. 69 個關節：14 組的 gain/bias 展開後逐元素套用
    gain = params["joint_gain"].index_select(-1, group_idx).unsqueeze(1)   # (B,14)→(B,1,69)
    bias = params["joint_bias"].index_select(-1, group_idx).unsqueeze(1)
    out_hinge = hinge * gain + bias

    ref_raw = cat([out_pos, out_quat, out_hinge], dim=-1)
    return ref_raw, kin.clamp_hinges(ref_raw)
```

**全部是逐元素乘加，沒有迴圈、沒有 if、沒有查表。** 所以它整條都可微，autograd 可以從追蹤誤差一路傳回 `RetargetNet` 的權重 —— 這就是「上層 T1 梯度是精確的」那句話的實際內容。

---

## 8. 36 維總表

| u 索引 | 參數 | 作用在 | 運算 | 上限 | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|---|---|---|---|
| 0–2 | `root_scale` | 骨盆位置 | 乘（log 空間）| ±16% | 凍結 | ✅ | ✅ |
| 3 | `root_dz` | 骨盆高度 | 加 | ±0.08 m | **✅** | ✅ | ✅ |
| 4–6 | `root_rot` | 骨盆朝向 | 右乘四元數 | ±0.15 rad | 凍結 | ✅ | ✅ |
| 7–20 | `joint_gain` | 69 個關節 | 乘（14→69）| ±20% | 凍結 | ✅ | ✅ |
| 21–34 | `joint_bias` | 69 個關節 | 加（14→69）| ±0.15 rad | 凍結 | ✅ | ✅ |
| 35 | `log_tau` | **時間軸** | 重取樣 | ×[0.8, 1.25] | 凍結 | 凍結 | 選配 |

凍結的作法是把那一維的 `u` 乘 0（`free_mask`）：`tanh(0) = 0`，該參數**精確取到 p=0 的值**且收不到梯度。不需要第二套程式碼路徑。

---

## 9. 常見疑問

**Q：為什麼 Stage 1 只解凍 `dz_root` 一維？**

因為 p 全凍時 Stage 1 贏不了。舊 pipeline 有一個逐幀的 `ground_correct_qpos` 把參考抬離地面，改成可學的 `dz_root` 之後，凍結 p 等於**同時凍掉唯一能抬它的東西**。實測 p=0 的參考在 88% 的影格陷進地板 12–16 mm，RSI 每次都把機器人塞進地面，物理再把它彈出來，然後它就倒了。

所以 Stage 1 拿那 1 個「修復破碎前提」的維度，Stage 2 拿其餘 35 個。

**Q：`dz_root` 為什麼要 warm start？**

那個偏移是**解析可算的**（就是參考自己的穿地中位數，約 +11 mm），而 `lr_upper = 1e-5` 從零學它要約 16000 個上層步 —— 實測 50 個 iteration 只走了 0.00005 m。所以直接算出來裝進輸出層的 bias，梯度只負責微調。

這不削弱 `u=0` 那個保證 —— 那講的是函式的代數性質，不是訓練從哪裡起步。

**Q：`RetargetNet` 是每具身體一個嗎？**

**不是，全部共用一份。** 正是這個共用逼得 p 必須是 β 的真實函數，而不是 9 組各自為政的修正。`Retargeter` 只是把「共用的網路 + box + 某一具身體的 FK」綁在一起的包裝。

**Q：一個 iteration 呼叫幾次？**

**一具身體一次**（9 組，每組約 28 個 env），不是 256 個 env 各叫一次 —— 同一具身體的 env 本來就共用同一個 `u`。

唯一的例外是 ES：它用 `u_override` 讓同一具身體的某些 env 帶不同的 `u`，這就是反對稱擾動「不需要額外模擬成本」的原因。
