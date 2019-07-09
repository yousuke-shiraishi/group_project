import cv2
import detection
import datetime

if __name__ == '__main__':
    while True:
        img = detection.main()
        
        #now_dt = datetime.datetime.now()
        
        #now_dt = now_dt.strftime('%Y%m%d%H%M%S')
            
        #cv2.imwrite('./image/aquarius/aquarius__' + now_dt + '.jpg', img)
        
        k = cv2.waitKey(0)
        
        if k == 27:
            break