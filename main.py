#import
import detection
import classify
import cv2
from time import sleep
import numpy as np

#if __name__ == __'main'__でここから処理開始
if __name__ == '__main__':
    # クラスがインスタンス化
    detecter1 = detection.Detection()
    claster = classify.predict_class(model='bottle_model_weight.hdf5')

    #初期値
    cart = []
    amount = []
    cart_loop = True
    pet_dict = {'namacha':140, 'soda_float':150, 'cclemon':160, 'fanta_litchi':170, 'cocacola':180}
    cart_num = {'namacha':0, 'soda_float':0, 'cclemon':0, 'fanta_litchi':0, 'cocacola':0}
    pet_lict = ['namacha', 'soda_float', 'cclemon', 'fanta_litchi', 'cocacola']

    #レジの立ち上げ
    while True:
        #買い物開始
        while cart_loop == True:
            print('cart')
            cap = cv2.VideoCapture(0)
            print('検出を開始します。')

            #検出タスク
            while True:
                sleep(0.2)
                ret, flame = cap.read()
                cv2.imshow('scan_Running',flame)

                #成功
                if ret:
                    detected_image = detecter1.object_detection(flame)

                    #検出完了したら出力へ
                    if detected_image is not None:
                        print('スキャンに成功しました')
                        cv2.destroyWindow('scan_Running')
                        cap.release()

                        #商品を予測
                        label = claster.predict(detected_image)

                        #今はラベルが帰って来てるけど、最終的には各クラスの確率を返す関数として、main.pyにてlabel付する。
                        #label = pet_list[np.argmax(claster.predict_multi_class(detected_image))]

                        #商品と値段をリストへ追加
                        cart.append(label)
                        amount.append(pet_dict[label])

                        #カートの個数を更新
                        cart_num[label] += 1

                        #商品の値段を表示
                        print('商品は{}:{}円です。\n'.format(label, pet_dict[label]))

                        #これまでの商品と個数を表示
                        print('読み込み済み商品')
                        for goods, num in cart_num.items():
                            if num != 0:
                                print('{} : {}個'.format(goods, num))
                            else:
                                pass

                        #カート内商品の合計金額を出す。
                        print('合計金額 : {}\n'.format(sum(amount)))
                        print('商品を取り出してください\n'\
                              '直前の商品を取り消す場合は「r」を押してください\n'\
                              'これでお買い物終了の場合は「q」を押してください。\n'\
                              'まだ商品がある場合は再度検出ボックスに商品を入れてください。\n')

                        #商品が取り出されるまでループ
                        cap2 = cv2.VideoCapture(0)
                        detecter2 = detection.Detection()
                        while True:
                            ret, flame = cap2.read()
                            non_detected_image = detecter2.object_detection(flame)
                            if non_detected_image is not None:
                                cap2.release()
                                break #直前のループ
                            else:
                                continue
                        break #検出のbreak

                    #検出できなかったら無視
                    else:
                        pass

                #失敗
                else:
                    print('スキャンに失敗しました')
                    continue

                #'q'が押されると会計
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    #買い物ループを止める
                    cart_loop = False
                    # cart内商品の合計金額を出す。
                    cap.release()
                    cv2.destroyWindow('scan_Running')
                    print('お会計')
                    print('合計金額は{}円です。しっかり払えや。\n'.format(sum(amount)))
                    print('「s」を押すと会計開始\n'\
                          '「e」を押すとシステム終了')
                    break #kensyutu

                #'r'が押されると一つ前の商品を抜く
                elif key == ord('r'):
                    print('{}が取り消されました\n'.format(label))
                    cart.pop(-1)
                    amount.pop(-1)
                    cart_num[label] -= 1
                    print('検出を開始します')
                    continue
        
        key2 = cv2.waitKey(1) & 0xFF
        #'e'が押されるとレジの終了
        if key2 == ord('e'):
            print('システムが終了しました')
            break

        #'s'が押されると検出開始
        elif key2 == ord('s'):
            cart_loop = True
            continue
