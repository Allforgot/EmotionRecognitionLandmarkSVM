package emotion_recognition;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Point;
import android.util.Log;

import com.tzutalin.dlib.Constants;
import com.tzutalin.dlib.FaceDet;
import com.tzutalin.dlib.VisionDetRet;
import com.yl.fecedetectdemo.R;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Detect face landmark using dlib
 *
 * Learn from project (https://github.com/gicheonkang/fast-face-android)
 *
 * @author tzx
 * 2018-09-30
 */

public class Landmark {
    private static final String TAG = "Landmark";

//    private static final String MODEL_FILE = "file:///android_asset/shape_predictor_68_face_landmarks.dat";

    private FaceDet mFaceDet;
    private Paint mFaceLandmarkPaint;   // Face bitmap with landmarks
    private ArrayList<Point> landmarks;   // The landmarks of the bitmap input
    private Bitmap faceBitmap;   // Origin face bitmap

    public Landmark () {
        mFaceDet = new FaceDet(Constants.getFaceShapeModelPath());    // Initial the FaceDet
        Log.i(TAG, "FaceDet initial success");

        // Initial paint
        mFaceLandmarkPaint = new Paint();
        mFaceLandmarkPaint.setColor(Color.GREEN);
        mFaceLandmarkPaint.setStrokeWidth(2);
        mFaceLandmarkPaint.setStyle(Paint.Style.STROKE);

    }

    private void setLandmarks(ArrayList<Point> landmarks) {
        this.landmarks = landmarks;
    }

    public ArrayList<Point> getLandmarks() {
        return this.landmarks;
    }

    public void setFaceBitmap(Bitmap bitmap) {
        this.faceBitmap = bitmap;
    }

    /**
     * Detect landmark from origin face bitmap
     */
    public void calculateLandmark() {
        Bitmap bitmap = this.faceBitmap;
        if (bitmap == null) {
            Log.e(TAG, "Bitmap passed in is null.");
            throw new NullPointerException("Bitmap null");
        }

//        ArrayList<Point> landmark;

        // bitmap contains only one face
        List<VisionDetRet> result = mFaceDet.detect(bitmap);  // Landmark detection
        Log.i(TAG, "FaceDet detect success.");
        if (result.size() != 0) {
            VisionDetRet ret = result.get(0);   // Only one face
            ArrayList<Point> landmark = ret.getFaceLandmarks();  // Get landmarks
            setLandmarks(landmark);
        }
        else {
            Log.i(TAG, "Landmark detect failed.");
            setLandmarks(null);
        }
    }

    /**
     * Draw landmarks to original face bitmap
     * @return bitmap face with landmark points
     */
    public Bitmap getFaceWithLandmark(Bitmap bitmap) {
        // If using (Bitmap mBitmap = this.faceBitmap) instead of the below code, a immutable
        // bitmap crash error will occur at (Canvas canvas = new Canvas(mBitmap)
//        Bitmap mBitmap = this.faceBitmap.copy(Bitmap.Config.ARGB_8888, true);
//        if (mBitmap == null) {
//            Log.e(TAG, "faceBitmap is null");
//            throw new NullPointerException("Bitmap null");
//        }
        Bitmap mBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        if (mBitmap == null) {
            Log.e(TAG, "faceBitmap is null");
            throw new NullPointerException("Bitmap null");
        }


        // If landmarks is null, return the original face
        ArrayList<Point> landmarks = this.landmarks;
        if (landmarks == null) {
            return mBitmap;
        }

        Canvas canvas = new Canvas(mBitmap);
        // Draw landmark
        for (Point point : landmarks) {
            canvas.drawCircle(point.x, point.y, 1, mFaceLandmarkPaint);
        }

        return mBitmap;
    }

}
