#import
import detection
import classify
import cv2
from time import sleep

#if __name__ == __'main'__でここから処理開始
if __name__ == '__main__':
    # クラスがインスタンス化
    detecter = detection.Detection()
    claster = classify.predict_class()
    
    #初期値
    cart = []
    amount = 0
    cart_loop = True
    pet_dict = {'アクエリ':140, 'ソーダフロート':150, 'cc レモン':160, 'ファンタ':170}
    pet_lict = ['アクエリ', 'ソーダフロート', 'cc レモン', 'ファンタ']

    #1回の買い物
    while cart_loop == True:
        cap = cv2.VideoCapture(0)
        print('検出開始します。')

        #検出タスク
        while True:
            sleep(0.2)
            ret, flame = cap.read()
            cv2.imshow('scan_Running',flame)

            #成功
            if ret:
                detected_image = detecter.object_detection(flame)

                #検出完了したらbreak
                if detected_image is not None:
                    print('scan_Successed')
                    cv2.destroyWindow('scan_Running')
                    cap.release()
                    
                    #検出結果の出力
                    cv2.imshow('Result', detected_image)

                    label = claster.predict(detected_image)

                    #今はラベルが帰って来てるけど、最終的には各クラスの確率を返す関数として、main.pyにてlabel付する。
                    #label = pet_list[np.argmax(claster.predict_multi_class(detected_image))]

                    cart.append(label)
                    print('商品は{}:150円です。'.format(label))
                    print('これでお買い物終了の場合はqを押してください。\nまだ商品がある場合は再度検出ボックスに商品を入れてください。')

                    break

            #失敗
            else:
                print('scan_Failured')
                continue

            #'q'が押されるとbreak
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print('Pressed finish button')
                cart_loop = False
                break

    # cart内商品の合計金額を出す。
    for pet in cart:
        amount += pet_dict[pet]
    print('合計金額は{}円です。しっかり払えや。'.format(amount))

