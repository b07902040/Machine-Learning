# ML HW5 Report
學號：b07902040 系級：資⼯⼆ 姓名：吳承軒
#### 1.(2%) 從作業三可以發現，使用 CNN 的確有些好處，試繪出其 saliency maps，觀察模型在做 classification 時，是 focus 在圖片的哪些部份？
![](https://i.imgur.com/3b84CJK.png)
以57, 3207, 4707, 3784四張照片為例。

可以發現除了第一張圖之外model有大致上focus在正確的位置。

第一張圖由於有多種食物和反光的影響，沒有focus在traget食物上。

第四張圖focus在蛋黃上，由於是屬於經典蛋狀的圖片，較好辨認。

#### 2.(3%) 承(1) 利用上課所提到的 gradient ascent 方法，觀察特定層的 filter 最容易被哪種圖片 activate 與觀察 filter 的 output。
觀察第十五層的第7, 16, 25, 36, 49個filter，這5個fliter_visualization的色調都相當紊亂，且色調上的pattern相近，可以推估這層filter是在做材質的分類。
#### 7.
![](https://i.imgur.com/C6WXizL.png)
![](https://i.imgur.com/9cHN0lw.png)
#### 49.
![](https://i.imgur.com/YF0WeNJ.png)
![](https://i.imgur.com/QM7meV8.png)
fliter7和filter49較為接近，所以其activate出來的圖片也較為接近，都是在物品邊界會有較大的gradient

#### 16.
![](https://i.imgur.com/nka4V5Z.png)
![](https://i.imgur.com/W4UMYxQ.png)
filter16主要是找彎曲狀色塊和偏藍色調，所以圖一左下角的陰影、圖二的藍色塊和圖四的手指周遭被activate較多。
#### 25.
![](https://i.imgur.com/DeUrorG.png)
![](https://i.imgur.com/5WnEomb.png)
fliter25主要找偏紅和物品的邊緣(亮度差異大)，被activate較多的區域為圖一下方的亮暗交錯、圖二的紅色處、圖三的盤緣及圖四的吐司邊。
#### 36.
![](https://i.imgur.com/BS8tV78.png)
![](https://i.imgur.com/SGakmgc.png)
filter36主要找周遭亮度、顏色變化較為過渡、連續的區塊，所以圖一和圖三中間較為複雜的區塊gradient最小，而構圖單純的圖四幾乎整張的gradient都相當大，和找周遭落差大的fliter25明顯呈現相反的特徵。

#### 3.(2%) 請使用 Lime 套件分析你的模型對於各種食物的判斷方式，並解釋為何你的模型在某些 label 表現得特別好 (可以搭配作業三的 Confusion Matrix)。
![](https://i.imgur.com/VsUdt2d.png)
![](https://i.imgur.com/LzyRVAu.png)
由於我在訓練時為了增加資料量，把val併入了train一起訓練，所以這些val都是model看過的樣本，所以整體正確率才這麼高。

即便如此，仍可以發現Noodles/Pasta, Rice, Soup等類別被分為其他類別的cases為0，說明我的model在這幾個類別中表現比較好。

透過圖二也可以發現其在rice上的表現相當好，估計應是透過色調為白色判斷的。

Confusion Matrix中也能看出Bread經常被誤認為Egg。

圖一就是這樣的情況，由於有太多種食物，無法精確標記。

圖四的Egg就相當完整，lime上也相當準確，甚至把手指完全獨立出來。

#### 4.(3%)  [自由發揮] 請同學自行搜尋或參考上課曾提及的內容，實作任一種方式來觀察 CNN 模型的訓練，並說明你的實作方法及呈現 visualization 的結果。
使用課堂上提到的deep dream觀察。

deep dream也就是用已經train好的model，固定model後，在原始的圖片做gradient accent，以此推估機器學到了什麼。

source code:https://github.com/utkuozbulak/pytorch-cnn-visualizations/tree/master/src
![](https://i.imgur.com/PaniyNO.jpg)
![](https://i.imgur.com/pZVw0F1.jpg)
使用網路上的source code，設定

```C
cnn_layer = 34
filter_pos = 94
lr=12
weight_decay=0.0001
```
來train training data中的其中一張肉類5_9.png，每10個epoch匯出一張照片，依次排序，得到上圖(左上為原圖)
可以發現在肉上，相較於其他區域，出現了較鮮艷的顏色，在第五張前後出現一隻上身似馬，下身似狗的生物的感覺，後面其背景開始有較大變化，使原先肉的位置變得較不明顯。