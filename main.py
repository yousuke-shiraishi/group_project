#import
import detection
import classify
import cv2
from time import sleep
import numpy as np
import pygame.mixer


def reset():
    '''
    商品をリセット
    '''
    cart = []
    amount = []
    cart_num = {'namacha':0, 'soda_float':0, 'cclemon':0, 'fanta_litchi':0, 'cocacola':0}

    return cart, amount, cart_num


def syoukei(cart_num, amount):
    '''
    これまでの商品と個数、合計金額を出力
    '''
    print('読み込み済み商品')
    for goods, num in cart_num.items():
        if num != 0:
            print('{} : {}個'.format(goods, num))
        else:
            pass

    #カート内商品の合計金額を出す。
    print('合計金額 : {}\n'.format(sum(amount)))


#if __name__ == __'main'__でここから処理開始
if __name__ == '__main__':
    # クラスがインスタンス化
    detecter1 = detection.Detection()
    claster = classify.predict_class(model='bottle_model_weight.hdf5')

    #初期値
    cart, amount, cart_num = reset()
    cart_loop = True
    pet_dict = {'namacha':140, 'soda_float':150, 'cclemon':160, 'fanta_litchi':170, 'cocacola':180}

    #音楽の初期設定
    pygame.mixer.quit()
    pygame.mixer.init()

    #レジの立ち上げ
    while True:

        #買い物開始
        while cart_loop == True:

            #ビデオ立ち上げ
            cap = cv2.VideoCapture(0)
            print('検出を開始します。')

            #検出タスク
            while True:
                sleep(0.1)
                ret, flame = cap.read()
                cv2.imshow('scan_Running',flame)

                #成功
                if ret:
                    detected_image = detecter1.object_detection(flame)

                    #検出完了したら出力へ
                    if detected_image is not None:

                        cv2.destroyWindow('scan_Running')
                        cap.release()

                        #商品を予測
                        label = claster.predict(detected_image)
                        array, std, var = claster.output_logit(image=detected_image, layer_name='activation_49')

                        #今はラベルが帰って来てるけど、最終的には各クラスの確率を返す関数として、main.pyにてlabel付する。
                        #label = pet_list[np.argmax(claster.predict_multi_class(detected_image))]

                        #しきい値より大きい(該当しない商品)
                        if std > 1.0:
                            print('スキャンに失敗しました\n'
                                  '商品を取り出してください')

                            #sound
                            pygame.mixer.music.load("sound/zannen.mp3") #読み込み
                            pygame.mixer.music.play(1) #再生
                            sleep(1.5)
                            pygame.mixer.music.stop() #終了

                            #商品が取り出されるまでループ
                            cap2 = cv2.VideoCapture(0)
                            detecter2 = detection.Detection()
                            while True:
                                ret, flame = cap2.read()
                                non_detected_image = detecter2.object_detection(flame)
                                if non_detected_image is not None:
                                    cap2.release()
                                    break
                                else:
                                    continue

                        #しきい値以下(該当する商品)
                        else:
                            print('スキャンに成功しました')

                            #sound
                            pygame.mixer.music.load("sound/scan.mp3") #読み込み
                            pygame.mixer.music.play(1) #再生
                            sleep(1.5)
                            pygame.mixer.music.stop() #終了

                            #商品と値段をリストへ追加
                            cart.append(label)
                            amount.append(pet_dict[label])

                            #カートの個数を更新
                            cart_num[label] += 1

                            #商品の値段を表示
                            print('商品は{}:{}円です。\n'.format(label, pet_dict[label]))

                            #これまでの商品と個数を表示
                            syoukei(cart_num, amount)

                            print('商品を取り出してください\n')

                            #商品が取り出されるまでループ
                            cap2 = cv2.VideoCapture(0)
                            detecter2 = detection.Detection()
                            while True:
                                ret, flame = cap2.read()
                                non_detected_image = detecter2.object_detection(flame)
                                if non_detected_image is not None:
                                    cap2.release()
                                    break
                                else:
                                    continue

                        print('r : 直前の商品を取り消す\n'\
                              'a : 最初からやり直す\n'\
                              'q : お買い物終了\n\n'\
                              'まだ商品がある場合は再度検出ボックスに商品を入れてください\n')
                        
                        break

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

                    #カメラを止める
                    cap.release()
                    cv2.destroyWindow('scan_Running')

                    # cart内商品の合計金額を出す。
                    print('お会計')
                    print('合計金額は{}円です。しっかり払えや。\n'.format(sum(amount)))
                    print('s : 会計開始\n'\
                          'e : システム終了')

                    #sound
                    pygame.mixer.music.load("sound/okini.mp3") #読み込み
                    pygame.mixer.music.play(1) #再生
                    sleep(1.5)
                    pygame.mixer.music.stop() #終了

                    break


                #'r'が押されると一つ前の商品を抜く
                elif key == ord('r'):
                    print('{}が取り消されました\n'.format(label))

                    #取り消し処理
                    cart.pop(-1)
                    amount.pop(-1)
                    cart_num[label] -= 1

                    #sound
                    pygame.mixer.music.load("sound/ee.mp3") #読み込み
                    pygame.mixer.music.play(1) #再生
                    sleep(1.5)
                    pygame.mixer.music.stop() #終了

                    #小計の表示
                    syoukei(cart_num, amount)
                    print('検出を開始します')

                    continue


                #'a'が押されると初期画面へ
                elif key == ord('a'):
                    print('最初に戻ります\n\n'\
                          's : 会計開始\n'\
                          'e : システム終了')
                    
                    #買い物ループを止める
                    cart_loop = False

                    #カメラを止める
                    cap.release()
                    cv2.destroyWindow('scan_Running')

                    #sound
                    pygame.mixer.music.load("sound/syuuryou.mp3") #読み込み
                    pygame.mixer.music.play(1) #再生
                    sleep(1.5)
                    pygame.mixer.music.stop() #終了

                    break


        #初期画像の表示
        img = cv2.imread("thanks.jpg")
        cv2.imshow("Thanks!", img)
        key2 = cv2.waitKey(0) & 0xFF

        #'e'が押されるとレジの終了
        if key2 == ord('e'):
            cv2.destroyWindow('Thanks!')

            #reset
            cart, amount, cart_num = reset()
            print('システムが終了しました')
            break

        #'s'が押されると検出開始
        elif key2 == ord('s'):
            cv2.destroyWindow('Thanks!')

            #sound
            pygame.mixer.music.load("sound/irassyai.mp3") #読み込み
            pygame.mixer.music.play(1) #再生
            sleep(2)
            pygame.mixer.music.stop() #終了

            #初期化
            cart, amount, cart_num = reset()
            cart_loop = True
            continue
