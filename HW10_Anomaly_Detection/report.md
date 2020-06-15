# ML HW10 report
### 學號：b07902040 系級：資工二	 姓名：吳承軒
1. (2%) 任取一個baseline model (sample code裡定義的 fcn，cnn，vae) 與你在kaggle leaderboard上表現最好的model（如果表現最好的model就是sample code裡定義的model的話就再任選一個，e.g.  如果cnn最好那就再選fcn），對各自重建的testing data的image中選出與原圖mse最大的兩張加上最小的兩張並畫出來。（假設有五張圖，每張圖經由autoencoder A重建的圖片與原圖的MSE分別為 [25.4, 33.6, 15, 39, 54.8]，則MSE最大的兩張是圖4、5而最小的是圖1、3）。須同時附上原圖與經autoencoder重建的圖片。（圖片總數：(原圖+重建)*(兩顆model)*(mse最大兩張+mse最小兩張) = 16張

上排為原圖，下排為重建，左側為mse最小，右側為mse最大
#### base line model(from cnn):
![](https://i.imgur.com/9lFW0Pe.png)
#### best model(from fcn):
![](https://i.imgur.com/IzXCsQP.png)

2. (1%) 嘗試把 sample code中的K-means 與 PCA 分別做在 autoencoder 的 encoder output 上，並回報兩者的auc score以及本來model的auc。autoencoder不限。不論分數與本來的model相比有上升還是下降，請同學簡述原因。

原來的分數:0.59219

使用K-means, n=4:0.58378

使用PCA:0.51586

使用fcn作為autoencoder，做了K-means和PCA後，跟原本的model相比下降了，推測可能是latent vector本身已具有相當重要性及代表性，再降維會導致資訊流失，導致失真。

3. (1%) 如hw9，使用PCA或T-sne將testing data投影在2維平面上，並將testing data經第1題的兩顆model的encoder降維後的output投影在2維平面上，觀察經encoder降維後是否分成兩群的情況更明顯。（因未給定testing label，所以點不須著色）

使用Tsne

1.直接將testing data投影到二維平面:
![](https://i.imgur.com/pOQiL6Y.png)
2.使用第一題的baseline model:

![](https://i.imgur.com/3KfqPiL.png)

3.使用第一題的best model:
![](https://i.imgur.com/wWjSZC7.png)
經降維後，分群的現象顯然更明顯了。(這裡分為四群，估計有3群為inlier，1群為outlier，經實驗後分為四群效果較好。)

4. (2%) 說明為何使用auc score來衡量而非binary classification常用的f1 score。如果使用f1 score會有什麼不便之處？

  F1-score ： $2*precision*recall$ / (precision + recall)

  AUC-score : ROC曲線下的面積，ROC曲線的橫軸為假正率，縱軸為真正率。

  precision和recall，都需要訂出一個閾值才能取得，而在此題或是現實中，我們不知道這個閾值為多少，或是設定多少較好，而AUC score中的ROC曲線相當於遍歷了所有閾值，且不需要知道實際上有多少為1多少為0，就能有效判斷好壞，避免樣本不平衡造成的影響，如果使用f1 score會有設定閾值的困擾。