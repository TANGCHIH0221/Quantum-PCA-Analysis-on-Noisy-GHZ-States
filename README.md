# Quantum-PCA-Analysis-on-Noisy-GHZ-States
使用主成分分析（PCA）處理高維量子系統中的密度矩陣，協助簡化量子態重建與雜訊分析，並應用於 GHZ state 的去噪與主特徵提取。

隨著量子位元數量的增加，量子系統的希爾伯特空間維度呈指數增長，導致量子態的描述、分析及層析變得極其複雜，透過對N量子位元系統密度矩陣的本徵分解（類比於經典PCA），展示了如何分析、視覺化並簡化高維量子態。以一個10量子位元GHZ態在經歷退極化噪聲後的演化為例，成功提取了系統的主要量子成分，並通過保真度比較證明了此方法在狀態理解和噪聲魯棒性方面的優勢。

在 Quantum Tomography 中，10-qubit 系統的密度矩陣維度達 1024 × 1024，傳統方法難以直觀分析。透過 PCA（對密度矩陣進行本徵分解）來：
- 分析原始 GHZ 純態與雜訊混合態的結構
- 找出主成分與其權重（eigenvalue spectrum）
- 提取主純態並與理想 GHZ 態進行 fidelity 比較
- 示範如何以一個主成分近似表示整個 noisy state

技術架構

- 使用 QuTiP 模擬 10-qubit GHZ 态
- 對密度矩陣進行本徵分解
- 視覺化：
  - Eigenvalue spectrum（線性與對數）
  - Cumulative eigenvalues（純度 vs. 主成分個數）
  - Fidelity vs. 原 GHZ state
- 應用 PCA 過濾雜訊，提取主要純態
- 簡化成一階主特徵表示（filtered pure state）

| 比較對象                         | Fidelity | 說明                               |
| ---------------------------- | -------- | -------------------------------- |
| F(ρ\_true, ρ\_noisy)         | 0.707    | 原 GHZ 與雜訊混合態相似度（已明顯偏離）           |
| F(ρ\_true, ρ\_PCA-filtered)  | 1.000    | PCA 過濾後的純態與原 GHZ 幾乎完全一致         |
| F(ρ\_noisy, ρ\_PCA-filtered) | 0.707    | 主純態仍保留原 noisy state 中 50% 的訊號主成分 |
