import cv2
import detection
from time import sleep

if __name__ == '__main__':
    detecter1 = detection.Detection()

    cap = cv2.VideoCapture(0)
    print('撮影を開始します。')

    #検出タスク
    count = 1
    while True:
        sleep(0.001)
        ret, flame = cap.read()
        cv2.imshow('scan_Running',flame)

        #成功
        if ret:
            detected_image = detecter1.object_detection(flame)

            #検出完了したら出力へ
            if detected_image is not None:
                print('撮影{}回目'.format(count))
                cv2.imwrite('./image3/fanta_litchi/fanta_litchi__' + str(count) + '.jpg', detected_image)
                count += 1

                if count == 100:
                    break
