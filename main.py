#import
import detection
<<<<<<< HEAD
import classifier
#main.pyの中で必要な関数を定義

#各クラスをインスタンス化
detecter = detection.detection()
classifier =classify.classify()
# 正解ラベル
label = ['cocacola-peach', 'ilohas', 'kuchidoke-momo', 'o-iocha', 'pocari-sweat', 'other_label']
# 商品価格
money = {'cocacola-peach':110, 'ilohas':120, 'kuchidoke-momo':130, 'o-iocha':140, 'pocari-sweat':150}


#if __name__ == __'main'__でここから処理開始
if __name__ == '__main__':
    #検出クラスを使って物体検出
    ext_img = detecter.object_detection(flame)
    if ext_img is None: #検出されていなかったらやり直し
        print('商品が正しく検出されませんでした。/n再度商品をレジボックスへセットしてください。')
        continue

    #分類モデルに画像を入力y
    pred = classifier.predict(ext_img)
    pred_name = label[np.argmax(pred)]
    if pred_label is (None or 'other_label': #分類出来ない、もしくは別の商品だったらやり直し
        print('正しく分類出来ませんでした。')
        continue
    
    

    
        
    
=======
from classify import predict_class
#main.pyの中で必要な関数を定義

#if __name__ == __'main'__でここから処理開始
if __name__ == '__main__':
    img = detection.main()
    label = predict_class(img)
    print(label)
        
#検出クラスを使って物体検出
>>>>>>> f7096fdb25ecd8bf7c43a4cb3f5d2528dfd2a9a5
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
