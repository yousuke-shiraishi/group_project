import cv2
import numpy as np

        
def get_background_subtraction(img1, img2):
    '''
    背景差分を行う関数
    
    Paremeters
    ---------------
    img1 : デフォルト画像
    img2 : 検出する物体を写した画像
    
    Returns
    ---------
    fgmask : 背景差分の結果
    '''
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    fgmask = fgbg.apply(img1)
    fgmask = fgbg.apply(img2)
        
    return fgmask
        
    
def get_frame(frame, cvsize, frame_return=False):
    """
    画像を取得し、リサイズ、切り取り、平滑化を行って返す関数
    
    Paramteres
    --------------
    frame : 検出する物体を写した画像
    cvsize : 切り取るサイズ
    frame_return: Trueならframeも返す
    
    Returns
    ---------
    gbur : トリミングしたぼかし画像
    frame : トリミングした検出画像
    """
    #resize
    frame = cv2.resize(frame, cvsize)
    #平滑化(ぼかし)
    gbur = cv2.GaussianBlur(frame, (5, 5), 0)

    if frame_return == True:
        return gbur, frame
    else:
        return gbur
    
    
def object_detection(img1, img2, cvsize=(224, 224), margin=10):
    """
    背景差分で検出する関数
    Parameters
    -------------
    img1 : デフォルト画像
    img2 : 検出する物体を写した画像
    cvsize : 切り取るサイズ
    margin : マージン
    
    Returns
    ---------
    detected_image : 検出した結果のnumpy配列
    """
    #デフォルト画像
    default = get_frame(img1, cvsize)
            
    #検出範囲
    rangethreshold = int(cvsize[0] / 10)

    #背景差分の取得
    gbur, frame = get_frame(img2, cvsize, frame_return=True)
    fgmask = get_background_subtraction(default, gbur)

    #変化がある位置を抜き出す
    Y, X = np.where(fgmask > 200)

    #横幅(最小値)
    x = int(np.min(X) - margin)
    #もし0を下回ればマージンなし
    if x < 0:
        x = x + margin
                
    #横幅
    w = int(np.max(X) - x + 1 + margin)
    #マージンで幅の最大値超えたらマイナスする
    if x + w > frame.shape[1]:
        w = w - x + w - frame.shape[1]
                
    #縦幅(最小値)
    y = int(np.min(Y) - margin)
    #マイナスならマージンなし
    if y < 0:
        y = y + margin

    #縦幅
    h = int(np.max(Y) - y + 1+ margin)
    #マージンで縦の最大値超えたらマイナスする
    if y + h > frame.shape[0]:
        h = h - y + h - frame.shape[0]
                
    #物体が検出範囲以上であれば検出
    if w > rangethreshold and h > rangethreshold:
        print("検出成功　　座標　x:{} y{} w {} h {}".format(x, y, w, h))
        detected_image = frame[y: y + h, x: x + w]
        print("detected_image取得")

        return detected_image
                
    #物体が検出範囲より小さければ検出しない
    else:
        pass