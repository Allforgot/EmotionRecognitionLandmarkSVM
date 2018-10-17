package emotion_recognition;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Range;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.utils.Converters;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


/**
 * emotion recognition
 *
 * @author tzx
 * @time 2018-08-31 15:24
 * @version v1.0
 */

public class Emotion {
    // Import libtensorflow_inference.so and libopencv_java3.so
    static {
        System.loadLibrary("tensorflow_inference");
        System.loadLibrary("opencv_java3");
    }

//    private static final String MODEL_FILE = "file:///android_asset/Gudi_model_100_epochs_20000_faces_0901_frozen_quantized_model.pb";  // Model file
//    private static final String MODEL_FILE = "file:///android_asset/optimzed_frozen_model_int8.pb";
//    private static final String INPUT_NODE_NAME = "network_input/X";  // The input node name of the model
//    private static final String OUTPUT_NODE_NAME = "network_output/Softmax";  // The output node name of the model
    private static final String MODEL_FILE = "file:///android_asset/jaffe_face_mobilenet_v2_140_224_steps2000.pb";
    private static final String INPUT_NODE_NAME = "Placeholder";  // The input node name of the model
    private static final String OUTPUT_NODE_NAME = "network_output";  // The output node name of the model
    private static final String EMOTION_SET[] = {"angry", "disgusted", "fearful", "happy", "sad", "surprised", "neutral"};
    private static final int NUM_CLASSES = 7;   // The number of the output classes

    // The width and height of the face to pass to the network
//    private static final int WIDTH = 48;
//    private static final int HEIGHT = 48;
    private static final int WIDTH = 224;
    private static final int HEIGHT = 224;

    private float networkInputs[];    // The input float[] of the network

    private TensorFlowInferenceInterface inferenceInterface;
    private Emotion.EmotionPredictThread emotionPredictThread;

    private Bitmap bmpInput;    // Store the input bitmap
    private String emotionResult;     // Store the predicted result
    private float score;     // The score of the emotion predicted

    private CascadeClassifier cascadeClassifier;     // OpenCV face detect

    /**
     * Define and initialize tensorflow interface
     *
     * @param context [in] the context of the Application, used to pass getAsstes() from Activity to
     *                TensorFlowInferenceInterface
     */
    public Emotion(Context context) {

        inferenceInterface = new TensorFlowInferenceInterface(context.getAssets(), MODEL_FILE); // Interface Definition
        Log.i("Emotion", "Tensorflow initials success.");
//        inferenceInterface.initializeTensorFlow(getAssets(), MODEL_FILE);   // Initialize the interface
        emotionPredictThread = new EmotionPredictThread();
//        initialOpenCVHaarcascade(context);
//
    }


    private void setNetworkInputs(float[] in) {
        this.networkInputs = in;
    }

    public void setBmpInput(Bitmap bmpInput) {
        this.bmpInput = bmpInput;
    }

    public String getEmotionResult() {
        return this.emotionResult;
    }

    public float getScore() {
        return this.score;
    }

    /**
     * Predict the result using the model, convert face image to gray first
     * Use OpenCV to convert bitmap to mat and preprocess the face before passing it into the
     * network. The main preprocess is Bitmap -> mat -> gray scaled -> resize with INTER_CUBIC
     * -> CvType.CV_32FC1 image type -> mat with 1 cols -> mat to List<Float> -> float[]
     *
     * @return emotion
     *//*
    public String predict() {
        Log.i("Emotion", "Start predict.");
        if (bmpInput == null)
            return "No emotion";

        Mat faceRgbMat = new Mat();     // Store the rgb mat convert from bitmap
        Mat faceGrayMat = new Mat();    // Store the gray mat convert from faceRgbMat
        Mat faceGrayResizedMat = new Mat();   // Store the resized gray mat
        final float networkInputs[] = new float[WIDTH * HEIGHT];    // Store the input data
        final float networkOutputs[] = new float[NUM_CLASSES];      // Store the output data of the network

        //============= Convert bitmap to mat and get gray image and resize using opencv ===========
        Utils.bitmapToMat(bmpInput, faceRgbMat);    // Bitmap -> Mat
//        if (bmpInput != null && bmpInput.isRecycled()) {    // Recycle bmpInput to avoid memory overflow
//            bmpInput.recycle();
//            bmpInput = null;
//        }
//        System.gc();    // Call system garbage collection to recycle the useless memory
        Imgproc.cvtColor(faceRgbMat, faceGrayMat, Imgproc.COLOR_BGR2GRAY);
        faceRgbMat = null;
        Size size = new Size(WIDTH, HEIGHT);
        Imgproc.resize(faceGrayMat, faceGrayResizedMat, size, 0, 0, Imgproc.INTER_CUBIC);
        faceGrayMat = null;

        //=================== Convert resized gray mat to float[] network_input ====================
        Mat faceCv32Fc1 = new Mat();   // Face mat in CvType.CV_32FC1 type
        faceGrayResizedMat.convertTo(faceCv32Fc1, CvType.CV_32FC1);
        Mat faceCols1 = faceCv32Fc1.reshape(1, WIDTH * HEIGHT);    // Face mat in 1 cols
        faceCv32Fc1 = null;
        List<Float> fs = new ArrayList<Float>();
        Converters.Mat_to_vector_float(faceCols1, fs);
        faceCols1 = null;
        Float[] F_ary = fs.toArray(new Float[0]);
        for (int i = 0; i < F_ary.length; i ++)
            networkInputs[i] = F_ary[i] / 255;
        F_ary = null;
        System.gc();

        //===================== Tensorflow feed run and fetch ========================
//        inferenceInterface.feed(INPUT_NODE_NAME, networkInputs, new long[]{WIDTH * HEIGHT});
        inferenceInterface.feed(INPUT_NODE_NAME, networkInputs, 1, WIDTH, HEIGHT, 1);
        Log.i("Emotion", "Tensorflow feed success.");
        inferenceInterface.run(new String[]{OUTPUT_NODE_NAME}, true);
        Log.i("Emotion", "Tensorflow run success.");
        inferenceInterface.fetch(OUTPUT_NODE_NAME, networkOutputs);
        Log.i("Emotion", "Tensorflow fetch success.");

        // Create a map with emotion name and the predicted values
        Map<String, Float> emotionMap = new HashMap<>();
        for (int i = 0; i < EMOTION_SET.length; ++ i)
            emotionMap.put(EMOTION_SET[i], networkOutputs[i]);

//        String emotion = getMaxKey(emotionMap);   // Get the emotion with largest value

//        if (emotion != null)
//            result = true;

        return getMaxKey(emotionMap);

    }*/

    /**
     * Predict with bitmap
     * @param bmpInput [in] the bitmap face passed in
     */
    public void predict(Bitmap bmpInput) {
        Log.i("Emotion", "Start predict thread.");
        // !emotionPredictThread.isStart to avoid the thread is start again and again, otherwise
        // the app will crash
        Mat faceRgbMat = new Mat();     // Store the rgb mat convert from bitmap
        Mat faceGrayMat = new Mat();    // Store the gray mat convert from faceRgbMat
        Mat faceGrayResizedMat = new Mat();   // Store the resized gray mat

        //============= Convert bitmap to mat and get gray image and resize using opencv ===========
        Utils.bitmapToMat(bmpInput, faceRgbMat);    // Bitmap -> Mat
//        if (bmpInput != null && bmpInput.isRecycled()) {    // Recycle bmpInput to avoid memory overflow
//            bmpInput.recycle();
//            bmpInput = null;
//        }
//        System.gc();    // Call system garbage collection to recycle the useless memory
        Imgproc.cvtColor(faceRgbMat, faceGrayMat, Imgproc.COLOR_BGR2GRAY);
        faceRgbMat = null;
        Size size = new Size(WIDTH, HEIGHT);
        Imgproc.resize(faceGrayMat, faceGrayResizedMat, size, 0, 0, Imgproc.INTER_CUBIC);

        float[] faceFloat = mat2float(faceGrayResizedMat, WIDTH, HEIGHT);  // Convert mat to float[WIDTH * HEIGHT]
        setNetworkInputs(faceFloat);

        if (emotionPredictThread != null && !emotionPredictThread.isStart)   // Start the prediction thread only once
            emotionPredictThread.start();
    }

    /**
     * Predict with bitmap with 3 channels
     * @param bitmap [in] bitmap input
     * @param imageMean [in] imageMean, 128
     * @param imageStd [in] imageStd, 128
     */
    public void predict(Bitmap bitmap, int imageMean, float imageStd) {
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, WIDTH, HEIGHT, false);
//        bitmap.recycle();
//        bitmap = null;
        int[] intValues = new int[WIDTH * HEIGHT];
        resizedBitmap.getPixels(intValues, 0, resizedBitmap.getWidth(), 0, 0,
                resizedBitmap.getWidth(), resizedBitmap.getHeight());
        float[] floatValues = new float[WIDTH * HEIGHT * 3];
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - imageMean) / imageStd;
            floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - imageMean) / imageStd;
            floatValues[i * 3 + 2] = ((val & 0xFF) - imageMean) / imageStd;
        }
        setNetworkInputs(floatValues);

        if (emotionPredictThread != null && !emotionPredictThread.isStart)  // Start the prediction thread only once
            emotionPredictThread.start();
    }

    /**
     * Predict with byte[] data read from camera, the face rectangle from ArcSoft is in arrayList
     * @param data [in] camera frame, NV21
     * @param mWidth [in] the width of the frame
     * @param mHeight [in] the height of the frame
     * @param arrayList [in] face rect
     */
    public void predict(byte[] data, int mWidth, int mHeight, ArrayList arrayList) {
        //================ Convert NV21 to gray mat ==========================
        Mat mYuv = new Mat(mHeight + mHeight / 2, mWidth, CvType.CV_8UC1);
        mYuv.put(0,0, data);
        Mat mGray = new Mat();
        Imgproc.cvtColor(mYuv, mGray, Imgproc.COLOR_YUV2GRAY_NV21);

        // Get face area with arrayList(face rectangle)
        Mat faceGrayMat = new Mat(mGray, new Range((int)arrayList.get(1), (int)arrayList.get(3)+1),
                new Range((int)arrayList.get(0), (int)arrayList.get(2)+1));

        Mat faceGrayResizedMat = new Mat();
        Size size = new Size(WIDTH, HEIGHT);
        Imgproc.resize(faceGrayMat, faceGrayResizedMat, size, 0, 0, Imgproc.INTER_CUBIC);
        float[] faceFloat = mat2float(faceGrayResizedMat, WIDTH, HEIGHT);  // Convert mat to float[WIDTH * HEIGHT]
        setNetworkInputs(faceFloat);

        if (emotionPredictThread != null && !emotionPredictThread.isStart)  // Start the prediction thread only once
            emotionPredictThread.start();
    }

    /**
     * Return the key of the largest value in the map
     * @param map [in] the input map
     * @return the key
     */
    private String getMaxKey(Map<String, Float> map) {
        List<Float> list = new ArrayList<>();
        for (String temp : map.keySet()) {
            float value = map.get(temp);
            list.add(value);
        }
        float max = 0;
        for (int i = 0; i < list.size(); i++) {
            float size = list.get(i);
            max = (max>size)?max:size;
        }
        for (String key : map.keySet()) {
            if (max == map.get(key)) {
                return key;
            }
        }
        return null;
    }

    /**
     * 求Map<K,V>中Value(值)的最大值
     *
     * @param map
     * @return
     */
    public static Object getMaxValue(Map<String, Float> map) {
        if (map == null)
            return null;
        int length =map.size();
        Collection<Float> c = map.values();
        Object[] obj = c.toArray();
        Arrays.sort(obj);
        return obj[length-1];
    }

    /**
     * Read OpenCV face detect xml file and initial cascadeClassifier
     * @param context
     */
/*    private void initialOpenCVHaarcascade(Context context) {
        try {
            InputStream is = context.getResources().openRawResource(org.opencv.R.raw.haarcascade_frontalface_default);
            File cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, "haarcascade_frontalface_default.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);

            byte buffer[] = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1)
                os.write(buffer, 0, bytesRead);
            is.close();
            os.close();

            // Load the cascade classifier
            cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
        } catch (Exception e) {
            Log.e("OpenCVActivity", "Error loading cascade", e);
        }
    }*/

    /**
     * Convert resized face gray mat to float[] as the input of the network
     * @param mat [in] the mat of the gray face
     * @param WIDTH [in] the width of the mat
     * @param HEIGHT [in] the height of the mat
     * @return float[WIDTH * HEIGHT]
     */
    private float[] mat2float(Mat mat, int WIDTH, int HEIGHT) {
        float[] result = new float[WIDTH * HEIGHT];
        Mat faceCv32Fc1 = new Mat();   // Face mat in CvType.CV_32FC1 type
        mat.convertTo(faceCv32Fc1, CvType.CV_32FC1);
        Mat faceCols1 = faceCv32Fc1.reshape(1, WIDTH * HEIGHT);    // Face mat in 1 cols
        faceCv32Fc1 = null;
        List<Float> fs = new ArrayList<Float>();
        Converters.Mat_to_vector_float(faceCols1, fs);
        faceCols1 = null;
        Float[] F_ary = fs.toArray(new Float[0]);
        for (int i = 0; i < F_ary.length; i++)
            result[i] = F_ary[i] / 255;
        F_ary = null;
        System.gc();

        return result;
    }

    class EmotionPredictThread extends Thread {
        private boolean isStart = false;    // The state of the emotionPredictThread to control
                                            // the thread start, only start the thread once in
                                            // predict()
        private volatile boolean exit = false;  // The exit flag of emotionPredictThread
        @Override
        public void run() {
            while (!exit) {
                isStart = true;
                Log.i("Emotion", "Start predict.");

                final float networkOutputs[] = new float[NUM_CLASSES];      // Store the output data of the network

                //===================== Tensorflow feed run and fetch ========================
//        inferenceInterface.feed(INPUT_NODE_NAME, networkInputs, new long[]{WIDTH * HEIGHT});
                inferenceInterface.feed(INPUT_NODE_NAME, Emotion.this.networkInputs, 1, WIDTH, HEIGHT, 3);
                Log.i("Emotion", "Tensorflow feed success.");
                inferenceInterface.run(new String[]{OUTPUT_NODE_NAME}, true);
                Log.i("Emotion", "Tensorflow run success.");
                inferenceInterface.fetch(OUTPUT_NODE_NAME, networkOutputs);
                Log.i("Emotion", "Tensorflow fetch success.");

                // Create a map with emotion name and the predicted values
                Map<String, Float> emotionMap = new HashMap<>();
                for (int i = 0; i < EMOTION_SET.length; ++i)
                    emotionMap.put(EMOTION_SET[i], networkOutputs[i]);

                Emotion.this.emotionResult = getMaxKey(emotionMap);
                Emotion.this.score = (float)getMaxValue(emotionMap);
            }
        }
    }

}
