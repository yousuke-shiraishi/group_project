import cv2
import numpy as np
from time import sleep


class detection:
    '''
    検出を行うクラス
    Parameters
    --------------
    cvsize : 切り取るサイズ
    margin : マージン
    
    Attributes
    ------------
    default : デフォルト画像
    preimg : 1つ前の画像
    '''
    def __init__(self, cvsize=(224, 224), margin=10):
        
        #ハイパーパラメータ
        #self.img = img #入力画像
        self.cvsize = cvsize #切り取るサイズ
        self.margin=margin #マージン
        
        #インスタンス変数
        self.default = None #デフォルト画像
        self.preimg = None #1つ前の画像
        

    def _get_background_subtraction(self, img1, img2):
        '''
        背景差分を行う関数

        Paremeters
        ---------------
        img1 : 差分される画像(デフォルト or 1つ前の画像)
        img2 : 差分する画像(入力画像)

        Returns
        ---------
        fgmask : 背景差分の結果
        '''
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        fgmask = fgbg.apply(img1)
        fgmask = fgbg.apply(img2)

        return fgmask


    def _get_frame(self, frame, frame_return=False):
        """
        画像を取得し、リサイズ、切り取り、平滑化を行って返す関数

        Paramteres
        --------------
        frame : 検出する物体を写した画像
        frame_return: Trueならframeも返す

        Returns
        ---------
        gbur : トリミングしたぼかし画像
        frame : トリミングした検出画像
        """
        #resize
        frame = cv2.resize(frame, self.cvsize)
        #平滑化(ぼかし)
        gbur = cv2.GaussianBlur(frame, (5, 5), 0)

        if frame_return == True:
            return gbur, frame
        else:
            return gbur


    def object_detection(self, img):
        """
        背景差分で検出する関数
        Returns
        ---------
        detected_image : 検出した結果のnumpy配列
        """
        #検出範囲
        rangethreshold = int(self.cvsize[0] / 10)
    
        #デフォルト画像
        if self.default is None:
            self.default = self._get_frame(img)

        #1つ前の画像の初期値
        if self.preimg is None:
            self.preimg = self._get_frame(img)

        #1つ前の画像との背景差分の取得
        gbur, frame = self._get_frame(img, frame_return=True)
        fgmask = self._get_background_subtraction(self.preimg, gbur)

        #preimgの更新
        self.preimg = gbur

        #前の画像と変化なし
        if np.max(fgmask) < 1:
            #デフォルト画像との背景差分の取得
            fgmask = self._get_background_subtraction(self.default, gbur)

            #デフォルトと変化あり
            if np.max(fgmask) > 200:
                #変化がある位置を抜き出す
                Y, X = np.where(fgmask > 200)

                #横幅(最小値)
                x = int(np.min(X) - self.margin)
                #もし0を下回ればマージンなし
                if x < 0:
                    x = x + self.margin

                #横幅
                w = int(np.max(X) - x + 1 + self.margin)
                #マージンで幅の最大値超えたらマイナスする
                if x + w > frame.shape[1]:
                    w = w - x + w - frame.shape[1]

                #縦幅(最小値)
                y = int(np.min(Y) - self.margin)
                #マイナスならマージンなし
                if y < 0:
                    y = y + self.margin

                #縦幅
                h = int(np.max(Y) - y + 1+ self.margin)
                #マージンで縦の最大値超えたらマイナスする
                if y + h > frame.shape[0]:
                    h = h - y + h - frame.shape[0]

                #物体が検出範囲以上であれば検出する
                if w > rangethreshold and h > rangethreshold:
                    detected_image = frame[y: y + h, x: x + w]
                    return detected_image

                else:
                    pass
            else:
                pass
        else:
            pass

            
def main():
    '''
    検出を実行する関数
    '''
    #各クラスをインスタンス化
    detecter = detection()
    
    #カメラ起動
    cap = cv2.VideoCapture(0)
    sleep(1)
    
    #検出    
    while True:
        ret, flame = cap.read()
        cv2.imshow('検出中', flame)
        if ret:
            detected_image = detecter.object_detection(flame)
                
            #検出完了したらbreak
            if detected_image:
                print('検出されました')
                break
            
        else:
            print('検出できませんでした')
            break
            
        #'Q'が押されるとbreak
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print('終了ボタンが押されました')
            break
                
    #検出結果の出力
    if detected_image:
        cv2.imshow('検出結果', detected_image)
    cap.release()
    cv2.destroyWindow('検出中')
    
    return detected_image