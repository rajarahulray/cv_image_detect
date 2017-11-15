import cv2
cap = cv2.VideoCapture(0)

while True:
    _,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7),3)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,21,7)
    im2, contours, hierarchy = cv2.findContours(th3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    ix = np.where(np.array(areas) > 300)[0]
    result = np.array([1,0,0,0,0,0,0,0,0,0])
    for i in ix:
        cnt = contours[i]
        xr,yr,wr,hr = cv2.boundingRect(cnt)
        if xr< 20 :
            xr = 25
            
            
        if yr < 20:
            yr = 25
            
        
        cv2.rectangle(img,(xr-10,yr-10),(xr+wr+10,yr+hr+10), (0,255,0),2)
        roi = th3[yr-20:yr+hr+20, xr-20:xr+wr+20]
        roi_re=cv2.resize(roi,(28,28))
        test = np.reshape(roi_re, (-1,np.product(roi_re.shape)))/255
        result=  sess.run(y,feed_dict = {X:test})
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'Number: '+str(result.argmax()),(xr-10,yr-10), font, 0.4, (255,0,0), 1, cv2.LINE_AA)

    cv2.imshow('Threshold',th3)
    cv2.imshow('orginal',img)
    
    if cv2.waitKey(41) & 0xff == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
