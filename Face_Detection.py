import cv2
import timeit

def videoDetector(cam, cascade):
    while True:
        start_t = timeit.default_timer()
        ret, img = cam.read()
        img = cv2.resize(img, dsize=None, fx=0.75, fy=0.75)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = cascade.detectMultiScale(gray,
                                           scaleFactor=1.1,
                                           minNeighbors=5,
                                           minSize=(20, 20))
        for box in results:
            x, y, w, h = box
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), thickness=2)

        terminate_t = timeit.default_timer()
        FPS = 'fps' + str(int(1./(terminate_t - start_t)))
        cv2.putText(img, FPS, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

        cv2.imshow('Face_Detection', img)

        if cv2.waitKey(1) > 0:
            break


cascade_filename = 'haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_filename)

cam = cv2.VideoCapture('sample.mp4')
img = cv2.imread('sample.jpg')

videoDetector(cam, cascade)



# Reference
# https://deep-eye.tistory.com/18
# https://deep-eye.tistory.com/46
# https://velog.io/@huttzza/%EC%8B%A4%EC%8B%9C%EA%B0%84-%EC%96%BC%EA%B5%B4-%EC%9D%B8%EC%8B%9D-%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8-1%EC%B0%A8-%EA%B5%AC%ED%98%84