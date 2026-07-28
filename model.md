## Task description

單一身材訓練流程

1. 標準身材產生 motion / rollout
    
    這裡是用原本的 robot.xml 根據 Metamotivo reward base generation 產出來的結果，包含 trjectory 跟 z0
    
    $$
    \tau_0, z_0
    $$
    
2. 對目標身材 (β) 做 motion retargeting
    
    這裡是對新的架構做 retargeting
    

$$
τ_β^{ref}=Retarget(τ_0,β)
$$

1.  初始 Adapter 產生
    
    $$
    z_\beta=G_\theta(\beta,z_0)
    $$
    
2. 在目標身材 simulator rollout：
    
    $$
    \tau_\beta=\mathrm{Action_{Head}(Action_{network}}(\pi_z,z_\beta,\beta))
    $$
    
3. 訓練目標改成：
    
    $$
    L=-R_{\text{task}}(\tau_\beta)+\lambda_{\text{rtg}}D(\tau_\beta,\tau^{\text{ref}}_\beta)+\lambda_z|z_\beta-z_0|^2+\lambda_{\text{phys}}L_{\text{phys}}
    $$
    

其中 (D) 是指 functional equivalence 可以包含：

$$
D =D_{\text{root}}+D_{\text{ee}}+D_{\text{contact}}+D_{\text{pose}}+D_{\text{velocity}}
$$

1. 根節點軌跡與姿態 ($D_{\text{root}}$)
    
    判斷動作（如「向右轉」）不能只看單幀姿勢，必須完整保留 **Root Heading**、**Yaw Rate** 與 **軌跡曲率（Trajectory Curvature）**。
    
2. 末端執行器軌跡 ($D_{\text{ee}}$)
    
    手、腳、頭等末端軌跡應該保持語意一致。例如揮手時手相對肩膀的路徑要像，走路時腳的擺動與落點要合理。
    
3. 接觸約束 ($D_{\text{contact}}$)
    
    腳底接觸地面、手碰物體、自接觸等都要保留。Contact-aware retargeting 特別強調保留 ground contact/self-contact 並降低 interpenetration，因為 foot sliding、穿模會嚴重破壞動作品質。(CVF 開放存取)
    
4. 物理可行性 ($D_{\text{pose}} / D_{\text{velocity}}$)
    
    採 **Physics-Aware Retargeting**，確保重映射後的動作能在目標身材上**實際執行**而不跌倒。
    

針對模型架構設計上

因為 在實際應用中，根據任務目標的不同，產出 *z* 的頻率有顯著差異：

- **目標達成 (Goal Reaching) 與 獎勵優化 (Reward Optimization)：**
    - **產出方式：一次產一個。**
    - 對於目標達成任務，*z* 是透過將目標狀態 *g* 丟入 Backward 網路 *B*(*g*) 得到的單一向量。在執行任務的過程中，這個 *z* 通常保持不變，作為機器人的長期意圖。
- **動作追蹤 (Motion Tracking)：**
    - **產出方式：產出一個序列。**
    - 為了精確跟隨一段動態的專家軌跡，系統會為軌跡中的**每一個時間步** *t* **推導出一個對應的** *zt*。
    - 具體做法是取當前時間步之後一段長度為 *L* 的窗口內所有狀態的 Backward 嵌入平均值。這意味著機器人每走一步，都會更新一次 *z*，形成一個引導動作的潛在向量序列。

所以先針對單一 frame 做，所以就是

$$
z_β=z_0+α⋅MLP_θ([β,z_0])
$$

layer：4 ～ 8 層

神經元數量：採用 Bottleneck 結構

• 輸入層：原始 z 向量 256 維
• 隱藏層：256 → 512 → 512 → 256
• 輸出層：新的 z 向量 256 維

神經網路設計等實際訓練後會再調整

注意在最後轉到 Actor network 的時候，也需要對 Actor network 之間的轉換做處裡，所以要在外層再加上一個 Action head，在不同身形之間的最後輸出做調整，所以總共會需要兩個神經網路 一個是針對 latent space 的轉換，一個是最後的 Action head，同時對兩層做梯度下降．