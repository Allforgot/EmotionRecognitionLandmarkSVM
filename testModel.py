import cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt
import math
import glob

# emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"] #Emotion list
emotions = ["angry", "happy", "neutral", "sad", "surprise"] #Emotion list
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Or set this to whatever you named the downloaded file
# svm_model = "emotion_landmark_svm_model_181009_78.xml"
svm_model = "svm_model.xml"
svm = cv2.ml.SVM_load(svm_model)

data = {}   

def get_landmarks(image):
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(0,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        # xmean = np.mean(xlist)
        # ymean = np.mean(ylist)
        # 
        xlist_rotate, ylist_rotate = angle_correction(xlist, ylist)

        # xcentral = [(x-xmean) for x in xlist]
        # ycentral = [(y-ymean) for y in ylist]

        # xnorm = [(i-min(xlist)) / (max(xlist)-min(xlist)) for i in xlist]
        # ynorm = [(i-min(ylist)) / (max(ylist)-min(ylist)) for i in ylist]

        xlistNorm, ylistNorm = normalize(xlist_rotate, ylist_rotate)
        xlist_new, ylist_new = delLandmark(xlistNorm, ylistNorm)
        xmean = (max(xlist_new) + min(xlist_new)) / 2
        ymean = (max(ylist_new) + min(ylist_new)) / 2
        # xNormLen = xmean - min(xlist_new)
        # yNormLen = ymean - min(ylist_new)
        # xNormLen = max(xlist_new) - min(xlist_new)
        # yNormLen = max(ylist_new) - min(ylist_new)
        # xNormLen = 1
        # yNormLen = 1

        xcentral = [(x-xmean) for x in xlist_new]
        ycentral = [(y-ymean) for y in ylist_new]
        # normDist = get_norm_dist(xlist_rotate, ylist_rotate)
        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist_new, ylist_new):
            # landmarks_vectorised.append((w-xmean) / xNormLen)
            # landmarks_vectorised.append((z-ymean) / yNormLen)
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)
            # landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))
            landmarks_vectorised.append(math.atan2(y, x))
        # for x, y in zip(xlist, ylist):
        #     landmarks_vectorised.append(x)
        #     landmarks_vectorised.append(y)

        influentialDist = get_influential_dist(xlistNorm, ylistNorm)
        for i in influentialDist:
            landmarks_vectorised.append(i)

        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1:
        data['landmarks_vectorised'] = "error"
    return landmarks_vectorised, xlistNorm, ylistNorm

def get_influential_dist(xlist, ylist):
    pointA = [45, 36, 41, 49, 48, 44, 49, 62, 35]
    pointB = [54, 48, 48, 67, 64, 46, 59, 66, 53]
    influentialDist = []
    # normDist = get_norm_dist(xlist, ylist)
    for i in range(0, len(pointA)):
        pA_y = ylist[pointA[i]]
        pA_x = xlist[pointA[i]]
        pB_y = ylist[pointB[i]]
        pB_x = xlist[pointB[i]]
        dist = np.linalg.norm(np.asarray((pA_y, pA_x)) - np.asarray((pB_y, pB_x)))
        influentialDist.append(dist)
    return influentialDist

def delLandmark(xlist, ylist):
    landmark2del = [34,32,30,29,28,27,16,15,14,13,12,11,10,9,7,6,5,4,3,2,1,0]
    xlist_tmp = xlist.copy()
    ylist_tmp = ylist.copy()
    for i in range(0, len(landmark2del)):
        del(xlist_tmp[landmark2del[i]])
        del(ylist_tmp[landmark2del[i]])
    return xlist_tmp, ylist_tmp

def get_norm_dist(xlist, ylist):
    normPoint = [39, 42]
    normDist = np.linalg.norm(np.asarray((ylist[normPoint[0]], xlist[normPoint[0]])) - np.asarray((ylist[normPoint[1]], xlist[normPoint[1]])))
    return normDist

def angle_correction(xlist, ylist):
    # angleRotate = math.atan2(ylist[21] - ylist[22], xlist[22] - xlist[21])
    angleRotate = math.atan2(xlist[30] - xlist[27], ylist[30] - ylist[27])
    xlist_rotate = []
    ylist_rotate = []
    for x,y in zip(xlist,ylist):
        xlist_rotate.append(x*math.cos(angleRotate) - y*math.sin(angleRotate))
        ylist_rotate.append(x*math.sin(angleRotate) + y*math.cos(angleRotate))
    return xlist_rotate, ylist_rotate

def normalize(xlist, ylist):
    xMin = min(xlist)
    yMin = min(ylist)
    xLen = max(xlist) - xMin
    # yLen = max(ylist) - yMin
    scale = 200.0 / xLen
    # yScale = 200.0 * yLen / xLen
    xlistNorm = []
    ylistNorm = []
    for x, y in zip(xlist, ylist):
        xlistNorm.append((x - xMin) * scale)
        ylistNorm.append((y - yMin) * scale)
    # print("scale = ", scale)

    return xlistNorm, ylistNorm

def predict(image_name):
    image = cv2.imread(image_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
    clahe_image = clahe.apply(gray)
    landmarks, xlist, ylist = get_landmarks(clahe_image)
    data = np.float32(landmarks).reshape(-1, len(landmarks))
    result = svm.predict(data)[1]
    result = emotions[int(result)]
    return result

# image_name = "/home/allforgot/Documents/CompanyProject/EmotionRecognition/emotion_landmark/fer2013/disgust/img-3779-1.png"
# image_name = "img3.jpg"
# image = cv2.imread(image_name)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
# clahe_image = clahe.apply(gray)
# landmarks, xlist, ylist = get_landmarks(clahe_image)
# data = np.float32(landmarks).reshape(-1, len(landmarks))
# result = svm.predict(data)[1]
# result = emotions[int(result)]
# 
# result = predict("img3.jpg")
# print(result)
# for i in range(0,len(xlist)):
# 	cv2.circle(image, (int(xlist[i]), int(ylist[i])), 5, (0,0,255))


# plt.imshow(image)
# plt.show()

total = 0
correct = 0
for emotion in emotions:
    files = glob.glob("fer2013_new/%s/*" %emotion)
    for file in files:
        try:
            result = predict(file)
        except:
            continue
        print(emotion, ":", file, " == ", result)
        total += 1
        if (result == emotion):
            correct += 1
print("accuracy: ", correct / total)
print("total: ", total)
print("correct: ", correct)
