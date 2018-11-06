package emotion_recognition;

import android.content.Context;
import android.graphics.Point;
import android.util.Log;

import com.yl.fecedetectdemo.R;

import org.opencv.core.Mat;
import org.opencv.ml.SVM;
import org.opencv.utils.Converters;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;


public class EmotionLandmark {
    static{
        System.loadLibrary("opencv_java3");
    }
    private static final String TAG = "EmotionLandmark";

    private ArrayList<Point> landmarks;
//    private static final String MODEL_NAME = "emotion_landmark_svm_model_181009_78.xml";
//    private static final String MODEL_NAME = "emotion_landmark_svm_model_181012_vectors_3.xml";
//    private static final String MODEL_PATH = "file:///android_assets/emotion_landmark_svm_model_181009_78.xml";

//    private static final String[] EMOTION_SET = {"angry", "disgusted", "fearful", "happy", "sad", "surprised", "neutral"};
//    private static final String[] EMOTION_SET = {"angry", "happy", "neutral", "sad", "surprised"};
//    private static final String[] EMOTION_SET = {"愤怒","厌恶","恐惧","高兴","悲伤","惊讶","中性"};
    private static final String[] EMOTION_SET = {"愤怒","高兴","中性","悲伤","惊讶"};

    private SVM svm;

    public EmotionLandmark() {  }

    public void setLandmarks(ArrayList<Point> landmarks) { this.landmarks = landmarks; }

    public void init(Context context) {
        try {
            InputStream is = context.getResources()
                    .openRawResource(R.raw.emotion_landmark_svm_model_181030_5emotions_ckp);
            File svmModelDir = context.getDir("emotion_landmark_svm", Context.MODE_PRIVATE);
            File svmModelFile = new File(svmModelDir, "emotion_landmark_svm_model_181030_5emotions_ckp.xml");
            FileOutputStream os = new FileOutputStream(svmModelFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();
            svm = SVM.load(svmModelFile.getAbsolutePath());
            Log.i(TAG, "SVM initial success.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Predict emotion
     * @return emotion result, one of EMOTION_SET
     */
    public String predict() {
        if (this.landmarks == null || this.landmarks.size() == 0) {
            Log.i(TAG, "No landmark.");
            return "No emotion.";
        }

        // Get feature from landmarks
        List<Float> featureFloatList = calFeature(this.landmarks);
        // Convert List<Float> to mat and reshape
        Mat landmarkMat = Converters.vector_float_to_Mat(featureFloatList);
        Mat landmarkReshape = landmarkMat.reshape(1, 1);

        // Predict result with OpenCV SVM
        float result = svm.predict(landmarkReshape);

        return EMOTION_SET[(int)result];
    }

    /**
     * Get the distance between android graphic point A and B
     * @param A [in] opencv point A, the head point
     * @param B [in] opencv point B, the end point
     * @return Float Distance
     */
    private float distBetweenAB(Point A, Point B) {
        return (float)Math.sqrt((B.x - A.x) * (B.x - A.x) + (B.y - A.y) * (B.y - A.y));
    }

    private float distBetweenAB(org.opencv.core.Point A, org.opencv.core.Point B) {
        return (float)Math.sqrt((B.x - A.x) * (B.x - A.x) + (B.y - A.y) * (B.y - A.y));
    }

    /**
     * Get arctan(y/x) as angle between android graphic point A and B
     * @param A [in] opencv point A, the head point
     * @param B [in] opencv point B, the end point
     * @return Float angle
     */
    private float angleBetweenAB(Point A, Point B) {
        return (float)Math.atan2((B.y - A.y), (B.x - A.x));
    }

    private float angleBetweenAB(org.opencv.core.Point A, org.opencv.core.Point B) {
        return (float)Math.atan2((B.y - A.y), (B.x - A.x));
    }

    /**
     * Calculate the central point of landmarks
     * @param landmarks [in]
     * @return the Central point
     */
    private org.opencv.core.Point calCentralPoint(ArrayList<org.opencv.core.Point> landmarks) {
        List<Double> xlist = new ArrayList<>();
        List<Double> ylist = new ArrayList<>();
        for (org.opencv.core.Point i : landmarks) {
            xlist.add(i.x);
            ylist.add(i.y);
        }
        double xMax = Collections.max(xlist);
        double xMin = Collections.min(xlist);
        double yMax = Collections.max(ylist);
        double yMin = Collections.min(ylist);

        return new org.opencv.core.Point((xMax+xMin)/2, (yMax+yMin)/2);
    }

    /**
     * Get margin [xMin, xMax, yMin, yMax] of the landmarks
     * @param landmarks [in]
     * @return ArrayList [xMin, xMax, yMin, yMax]
     */
    private List<Double> getMargin(ArrayList<org.opencv.core.Point> landmarks) {
        List<Double> xlist = new ArrayList<>();
        List<Double> ylist = new ArrayList<>();
        for (org.opencv.core.Point i : landmarks) {
            xlist.add(i.x);
            ylist.add(i.y);
        }
        double xMax = Collections.max(xlist);
        double xMin = Collections.min(xlist);
        double yMax = Collections.max(ylist);
        double yMin = Collections.min(ylist);

        List<Double> margin = new ArrayList<>();
        margin.add(xMin);
        margin.add(xMax);
        margin.add(yMin);
        margin.add(yMax);

        return margin;
    }

    /**
     * Get the classify landmark feature
     * @param landmarks [in]
     * @return feature list
     */
    private List<Float> calFeature(ArrayList<Point> landmarks) {
//        List<Integer> normPoint = Arrays.asList(39, 42);  // Two eye points
        List<Integer> landmarkDel = Arrays.asList(34,32,30,29,28,27,16,15,14,13,12,11,10,9,7,6,5,4,3,2,1,0);
        List<Integer> influentialPointA = Arrays.asList(45,36,41,49,48,44,49,62,35);
        List<Integer> influentialPointB = Arrays.asList(54,48,48,67,64,46,59,66,53);

        ArrayList<org.opencv.core.Point> landmarkRotate = angelRotate(landmarks);
        ArrayList<org.opencv.core.Point> landmarkScaled = scaleLandmark(landmarkRotate);

//        Point centralPoint = calCentralPoint(landmarks);
//        float normDist = distBetweenAB(landmarkRotate.get(normPoint.get(0)), landmarkRotate.get(normPoint.get(1)));
        List<Float> influentialDist = new ArrayList<>();
        for (int i = 0; i < influentialPointA.size(); i ++) {
            org.opencv.core.Point A = landmarkScaled.get(influentialPointA.get(i));
            org.opencv.core.Point B = landmarkScaled.get(influentialPointB.get(i));
            float dist = distBetweenAB(A, B);
            influentialDist.add(dist);
        }

        // Delete the useless landmark
        ArrayList<org.opencv.core.Point> landmarksNew = new ArrayList<>();
        landmarksNew.addAll(landmarkScaled);
        for (int i : landmarkDel) {
            landmarksNew.remove(i);
        }

        org.opencv.core.Point centralPoint = calCentralPoint(landmarksNew);  // Get the central point

        // Get feature of each point, [x, y, dist, angle]
        List<Float> featureFloatList = new ArrayList<>();
        for (org.opencv.core.Point point : landmarksNew) {
            float dist = distBetweenAB(centralPoint, point);
            float angle = angleBetweenAB(centralPoint, point);

            featureFloatList.add((float)point.x);
            featureFloatList.add((float)point.y);
            featureFloatList.add(dist);
            featureFloatList.add(angle);
        }

        featureFloatList.addAll(influentialDist);

        return featureFloatList;
    }

    /**
     * Correct the angle of landmark according to nose
     * @param landmarks [in]
     * @return the corrected landmarks
     */
    private ArrayList<org.opencv.core.Point> angelRotate(ArrayList<Point> landmarks) {
        double angle = Math.atan2(landmarks.get(30).x - landmarks.get(27).x,
                landmarks.get(30).y - landmarks.get(27).y);
        ArrayList<org.opencv.core.Point> landmarksNew = new ArrayList<>();
        for (Point point : landmarks) {
            float x = (float)(point.x * Math.cos(angle) - point.y * Math.sin(angle));
            float y = (float)(point.x * Math.sin(angle) + point.y * Math.cos(angle));
            landmarksNew.add(new org.opencv.core.Point(x, y));
        }
        return landmarksNew;
    }

    /**
     * Scale landmark to [200,  ]
     * @param landmarks [in]
     * @return the scaled landmarks
     */
    private ArrayList<org.opencv.core.Point> scaleLandmark(ArrayList<org.opencv.core.Point> landmarks) {
        List<Double> margin = getMargin(landmarks);  // margin = [xMin, xMax, yMin, yMax]
        double scale = 200 / (margin.get(1) - margin.get(0));
        ArrayList<org.opencv.core.Point> landmarkScaled = new ArrayList<>();
        for (org.opencv.core.Point point : landmarks) {
            double x = (point.x - margin.get(0)) * scale;
            double y = (point.y - margin.get(2)) * scale;
            landmarkScaled.add(new org.opencv.core.Point(x, y));
        }
        return landmarkScaled;
    }

}
