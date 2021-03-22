# AI_homework1

我所使用的是pytorch來完成的，並且使用的類神經網路為LSTM，並且輸入資料只有用備轉容量而已
並且我是用前七天預測來預測下一天的方式訓練的
以下我針對我的code做一些講解

每張圖在進行訓練的時候，我都有先做normalized的處理，並且maen跟std是之前就先用numpy算好並寫在那邊的

![image](https://github.com/qw61116111/AI_homework1/blob/main/image/mean.jpg)

接下來是我的dataset的處裡
遞一部分我把圖都讀進來，並且用for迴圈對每張圖進行減平均除標準差的normalized

![image](https://github.com/qw61116111/AI_homework1/blob/main/image/dataset.jpg)


接下來我網路的參數以及架構

![image](https://github.com/qw61116111/AI_homework1/blob/main/image/para.jpg)

![image](https://github.com/qw61116111/AI_homework1/blob/main/image/lstm.jpg)
