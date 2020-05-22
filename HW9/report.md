# ML HW9 report
### b07902040 資工二 吳承軒
### 1. (3%) 請至少使用兩種方法 (autoencoder 架構、optimizer、data preprocessing、後續降維方法、clustering 算法等等) 來改進 baseline code 的 accuracy。
#### a.分別記錄改進前、後的 test accuracy 為多少。
改進前在test上的正確率為74.800%

改進後在test上的正確率為81.458%

#### b.分別使用改進前、後的方法，將 val data 的降維結果 (embedding) 與他們對應的 label 畫出來。
改進前:

The clustering accuracy is: 0.592

![](https://i.imgur.com/lYjfpwF.png)

改進後:

The clustering accuracy is: 0.822

![](https://i.imgur.com/AHPlDXy.png)

可以發現改進後，分群的效果明顯更好

#### c.盡量詳細說明你做了哪些改進。
1.batch_size由64改為16、learning_rate由1e-5改為1e-4、num_epoch由100改為120，適當地調整這三個參數，使梯度下降的過程更為準確、時間更為合理。

2.optimizer由Adam改為AdamW，AdamW改進了Adam中的L2 regularization受到梯度變化速度的影響。

3.clustering由MiniBatchKMeans改為Kmeans，MiniBatch是Kmeans節省時間，而此題的data不算太多，所以節省時間不多，改回Kmeans能使結果更精準。

### 2. (1%) 使用你 test accuracy 最高的 autoencoder，從 trainX 中，取出 index 1, 2, 3, 6, 7, 9 這 6 張圖片
#### a.畫出他們的原圖以及 reconstruct 之後的圖片。
上排為原圖，下排為reconstruct 之後的圖片。
![](https://i.imgur.com/P8aLijm.png)
### 3. (2%) 在 autoencoder 的訓練過程中，至少挑選 10 個 checkpoints 
#### a.請用 model 的 train reconstruction error (用所有的 trainX 計算 MSE) 和 val accuracy 對那些 checkpoints 作圖。
![](https://i.imgur.com/WUfwGuF.png)
#### b.簡單說明你觀察到的現象。
總共做了120次迭代，每10次存一個資料點。

可以發現training loss整體而言是在下降的，而validation上的正確率，和training loss並沒有明顯的關係，推測可能的原因有:

1.train和validation的分布本身有差異

2.這裡的loss function計算的是原圖和auto encoder後的圖之間的差異，與分群後的正確與否並無直接相關。
