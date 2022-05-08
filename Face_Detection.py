import cv2
import timeit

def videoDetector(cam, cascade, age_net, gender_net, MODEL_MEAN_VALUES, age_list, gender_list):
    while True:
        # start_t = timeit.default_timer()
        ret, img = cam.read()
        try:
            img = cv2.resize(img, dsize=None, fx=1.0, fy=1.0)
        except:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = cascade.detectMultiScale(gray,
                                           scaleFactor=1.1,
                                           minNeighbors=5,
                                           minSize=(20, 20))
        for box in results:
            x, y, w, h = box
            face = img[int(y):int(y+h), int(x):int(x+h)].copy()
            blob = cv2.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_preds.argmax()

            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_preds.argmax()

            info = gender_list[gender] + ' ' + age_list[age]

            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), thickness=2)
            cv2.putText(img, info, (x, y-15), 0, 0.5, (0, 255, 0), 1)

        # terminate_t = timeit.default_timer()
        # FPS = 'fps' + str(int(1./(terminate_t - start_t)))
        # cv2.putText(img, FPS, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

        cv2.imshow('Face_Detection', img)

        if cv2.waitKey(1) > 0:
            break


cascade_filename = 'haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_filename)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt',
                                   'age_net.caffemodel')
print("age : ", end='')
print(age_net)
gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt',
                                      'gender_net.caffemodel')
print("gender : ", end='')
print(gender_net)
age_list = ['(0 ~ 2)','(4 ~ 6)','(8 ~ 12)','(15 ~ 20)',
            '(25 ~ 32)','(38 ~ 43)','(48 ~ 53)','(60 ~ 100)']
gender_list = ['Male', 'Female']

cam = cv2.VideoCapture('sample.mp4')

videoDetector(cam,cascade,age_net,gender_net,MODEL_MEAN_VALUES,age_list,gender_list )


# Reference
# https://deep-eye.tistory.com/18
# https://deep-eye.tistory.com/46
# https://velog.io/@huttzza/%EC%8B%A4%EC%8B%9C%EA%B0%84-%EC%96%BC%EA%B5%B4-%EC%9D%B8%EC%8B%9D-%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8-1%EC%B0%A8-%EA%B5%AC%ED%98%84