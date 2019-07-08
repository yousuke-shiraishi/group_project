#import
import numpy as np
import cv2
from time import sleep
import detection
#main.pyの中で必要な関数を定義

#各クラスをインスタンス化
detecter = detection.detection()

#if __name__ == __'main'__でここから処理開始
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    sleep(1)
    while True:
        ret, flame = cap.read()
        cv2.imshow('flame',flame)
        if ret:
            ext_img = detecter.object_detection(flame)
        else:
            break
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv2.imshow('ext_img',ext_img)
    cap.release()
    cv2.destroyWindow('flame')
cv2.imshow('ext_img2', ext_img)
    
        
    
#検出クラスを使って物体検出
#検出クラスの中での処理
    #cv2.videocaptureを使って画像を撮影
    #背景差分を使って、物体を検出
    #いくつかの条件を元に検出したものを確定させて画像として排出
    #条件例、１,検出された物体がある閾値以上のサイズとなる。2,前回の検出された物体と、状態が変わらなければ(落ち着いたら)等

#検出クラスを使って画像を排出

#(仮)ペットボトルかどうかを分類するモデルを使って分類

#排出された画像を使って分類モデルへ渡す。

    #分類が出来たら、dictに追加。
    #print(これで買い物終了ですか？)的なメッセージをだして、客からの反応を待つ。
    #終わりじゃなかったら検出タスクに戻る。

    #分類が出来なかったら、(確度が低い or 別のものとしての検出)
    #print('正しく検出出来ません')的なメッセージを出して、検出タスクへ戻る。

#買い物終了だった場合は合計金額を改めて出力する。
#リセットボタン(本当は決済会社からのなんかのレスポンス)をおすと検出待ちとなる。