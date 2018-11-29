package emotion_recognition;

import android.os.Environment;

import java.io.File;

public class Constants {

    public static String SVM_MODEL_NAME = "emotion_landmark_svm_model_181012_vectors_3.xml";

    public static String SHAPE_PREDICTOR_MODEL = "shape_predictor_68_face_landmarks.dat";

    private Constants() {
        // Constants should be private
    }

    /**
     * getFaceShapeModelPath
     * @param modelName [in] model name
     * @return default face shape model path
     */
    public static String getModelPath(String modelName) {
        File sdcard = Environment.getExternalStorageDirectory();
        String targetPath = sdcard.getAbsolutePath() + File.separator + modelName;
        return targetPath;
    }
}
