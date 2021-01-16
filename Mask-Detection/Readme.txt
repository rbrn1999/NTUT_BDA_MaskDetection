執行前需安裝
pip install tensorflow==2.3.1
pip install tensorflow-cpu==2.4.0
pip install opencv-python==4.4.0.46
pip install -U numpy==1.19.3   (更改numpy版本，最新版本1.19.4會有錯誤)
pip install playsound==1.2.2
pip install Pillow==8.0.1

以下套件會自動安裝若有錯誤可以參照以下版本
wheel==0.36.1
absl-py==0.11.0
tdqm == 4.54.1

訓練模型
請先下載資料集:https://drive.google.com/file/d/1avQ5z4FSdBa8lvxd9PsUeEC-eCxEDObA/view?usp=sharing 
創建一個Maskdata資料夾並將壓縮檔內三個資料夾解壓縮到裡面
執行python voc_to_tfrecord.py將資料集轉換成tfrecord檔使tensorflow能夠讀取，轉換完成的tfrecord檔會儲存在Maskdata資料夾
執行python check_dataset.py檢視tfrecord檔是否正確可以讀取，按下q能關閉視窗
執行python train.py開始訓練模型，預設跑100個epoch，每5個epoch會將當前的模型儲存在checkpoints資料夾
epoch及相關參數可以到components/config.py內修改口罩辨識

執行python main.py開啟口罩辨識程式
按下select model選擇checkpoints/model.h5或是自己訓練的模型 (checkpoints/model.h5是已經跑過100個epoch的模型)
按下open camera開啟相機可以看到當前畫面的辨識結果
按下open alert可以開啟警鈴聲達到提醒效果
