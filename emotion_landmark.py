import cv2
import glob
import random
import math
import numpy as np
import dlib
import itertools
from sklearn.svm import SVC

# emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"] #Emotion list
emotions = ["angry", "happy", "neutral", "sad", "surprise"] #Emotion list
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Or set this to whatever you named the downloaded file

data = {} #Make dictionary for all values
#data['landmarks_vectorised'] = []

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("CK+/%s/*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.80)] #get first 80% of file list
    prediction = files[-int(len(files)*0.20):] #get last 20% of file list
    return training, prediction
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
def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        print(" working on %s" %emotion)
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            clahe_image = clahe.apply(gray)
            # print(item)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one -> training")
            else:
                training_data.append(data['landmarks_vectorised']) #append image array to training data list
                training_labels.append(emotions.index(emotion))
        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            # print(item)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one -> prediction")
            else:
                prediction_data.append(data['landmarks_vectorised'])
                prediction_labels.append(emotions.index(emotion))
    return training_data, training_labels, prediction_data, prediction_labels

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

if __name__ == '__main__' :

    flag = 1  # 0 for test and 1 for model

    if flag == 0:
        clf = SVC(kernel='linear', probability=True, tol=1e-3)#, verbose = True) #Set the classifier as a support vector machines with polynomial kernel
        accur_lin = []
        for i in range(0,20):
            print("Making sets %s" %i) #Make sets by random sampling 80/20%
            training_data, training_labels, prediction_data, prediction_labels = make_sets()
            npar_train = np.array(training_data) #Turn the training set into a numpy array for the classifier
            npar_trainlabs = np.array(training_labels)
            print("training SVM linear %s" %i) #train SVM
            clf.fit(npar_train, training_labels)
            print("getting accuracies %s" %i) #Use score() function to get accuracy
            npar_pred = np.array(prediction_data)
            pred_lin = clf.score(npar_pred, prediction_labels)
            print("linear: ", pred_lin)
            accur_lin.append(pred_lin) #Store accuracy in a list
        print("Mean value lin svm: %s" %np.mean(accur_lin)) #FGet mean accuracy of the 10 runs

    elif flag == 1:
        training_data, training_labels, prediction_data, prediction_labels = make_sets()
        svm_param = dict(kernel_type=cv2.ml.SVM_LINEAR, svm_type=cv2.ml.SVM_C_SVC)
        svm = cv2.ml.SVM_create()
        svm.setKernel(cv2.ml.SVM_LINEAR)
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setC(2.73)

        train_data_np = np.float32(training_data)
        train_labels_np = np.array(training_labels).T
        train_labels_np.reshape(1,-1)
        print(train_data_np.shape)
        print(train_labels_np.shape)

        svm.train(samples=train_data_np, layout=cv2.ml.ROW_SAMPLE, responses=train_labels_np)
        svm.save('svm_model.xml')

        _, result = svm.predict(np.float32(prediction_data))
        mask = result == np.array([prediction_labels]).T
        correct = np.count_nonzero(mask)
        print(result)
        print(correct)
        print(np.array(prediction_labels))
        print("accuracy: ", correct*100.0/result.size)