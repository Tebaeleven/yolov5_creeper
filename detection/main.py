import torch #TensorFlowなどと同じ機械学習ライブラリ
import cv2 #OpenCV 画像加工、機械学習
import numpy as np #演算ライブラリ
from mss import mss #スクリーンショットライブラリ
import time #時間計測ライブラリ

#YOLOv5のモデル読み込み&自作データセット読み込み
model = torch.hub.load('ultralytics/yolov5', 'custom', path='detection/best.pt',force_reload = True)

#テキスト色の設定
red=0,0,255
green=0,255,0
blue=255,0,0

with mss() as sct:
    #スクリーンショットの範囲
    monitor = {"top": 220, "left": 940, "width": 600, "height":600}
    
    #以下無限ループ
    while True:
        
        #スタート時間計測
        start_time = time.perf_counter()

        #モニタ範囲でスクショ
        screenshot = np.array(sct.grab(monitor))
        #YOLOv5に入力&結果取得
        results = model(screenshot, size=600)
        
        objects = results.pandas().xyxy[0]  # 検出結果を取得
        
        #bodyカウント
        b_count=0
        #headカウント
        h_count=0

        #物体の数だけカウント
        for i in range(len(objects)):
            #bodyクラスだった場合
            if objects["class"][i]== 0:
                
                name = objects["class"][i]
                #xmin = objects.xmin[0,1]
                #ymin = objects.ymin[0,1]
                b_count+=1
            
            #headクラスだった場合
            if objects["class"][i]== 1:
                
                name = objects["class"][i]
                #xmin = objects.xmin[0,1]
                #ymin = objects.ymin[0,1]
                h_count+=1

        #fps表示
        cv2.putText(results.imgs[0], f"FPS: {int(1/(time.perf_counter() - start_time))}", (5, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (green), 2)

        #headカウント表示
        cv2.putText(results.imgs[0], f"name: head count: {h_count}", (5, 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (green), 2)
        #bodyカウント表示
        cv2.putText(results.imgs[0], f"name: body count: {b_count}", (5, 90), cv2.FONT_HERSHEY_TRIPLEX, 1, (green), 2)

        
        #結果表示
        results.print()
        #バウンディングボックス表示
        results.render()
        #opencvでウィンドウ作成
        cv2.imshow("frame", results.imgs[0])
        #qキーで全て終了
        if(cv2.waitKey(1) == ord('q')):
            cv2.destroyAllWindows()
            break
