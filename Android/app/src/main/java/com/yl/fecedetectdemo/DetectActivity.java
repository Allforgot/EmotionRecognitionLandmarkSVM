package com.yl.fecedetectdemo;

import android.content.Context;
import android.content.pm.ActivityInfo;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Point;
import android.graphics.YuvImage;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.guo.android_extend.java.ExtByteArrayOutputStream;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

import emotion_recognition.EmotionLandmark;
import emotion_recognition.Landmark;


public class DetectActivity extends AppCompatActivity implements
        CameraBridgeViewBase.CvCameraViewListener2, View.OnClickListener {

    private CameraBridgeViewBase cameraView;
    private TextView emotionResultTextview;
    private CascadeClassifier classifier;
    private Mat mGray;
    private Mat mRgba;
    private int mAbsoluteFaceSize = 0;
    private boolean isFrontCamera;

    private Landmark landmark;
    private EmotionLandmark emotionLandmark;
    private String emotionResult = "无表情";
    private final static int SHOW_EMOTION = 100;

    private int frameCount;

    // 手动装载openCV库文件，以保证手机无需安装OpenCV Manager
    static {
        System.loadLibrary("opencv_java3");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        initWindowSettings();
        setContentView(R.layout.activity_detect);
        cameraView = (CameraBridgeViewBase) findViewById(R.id.camera_view);
        cameraView.setCvCameraViewListener(this); // 设置相机监听
        cameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT);
        emotionResultTextview = (TextView)findViewById(R.id.emotion_result);
        isFrontCamera = true;
//        cameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_BACK);

        initClassifier();
//        initialSVMModel();
        emotionLandmark = new EmotionLandmark();
        emotionLandmark.init(getApplicationContext());
        landmark = new Landmark();
        landmark.init();

        cameraView.enableView();
//        Button switchCamera = (Button) findViewById(R.id.switch_camera);
//        switchCamera.setOnClickListener(this); // 切换相机镜头，默认后置
    }

    private Handler handlerMain = new Handler() {
        @Override
        public void handleMessage(Message msg) {
            super.handleMessage(msg);
            switch (msg.what) {
                case SHOW_EMOTION:
                    emotionResultTextview.setText(emotionResult);
                    break;
            }
        }
    };

    @Override
    public void onClick(View v) {
        switch (v.getId()) {
            case R.id.switch_camera:
                cameraView.disableView();
                if (isFrontCamera) {
                    cameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_BACK);
                    isFrontCamera = false;
                } else {
                    cameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT);
                    isFrontCamera = true;
                }
                cameraView.enableView();
                break;
            default:
        }
    }

    // 初始化窗口设置, 包括全屏、横屏、常亮
    private void initWindowSettings() {
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
    }

    // 初始化人脸级联分类器，必须先初始化
    private void initClassifier() {
        try {
            InputStream is = getResources()
                    .openRawResource(R.raw.lbpcascade_frontalface);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File cascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
            FileOutputStream os = new FileOutputStream(cascadeFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();
            classifier = new CascadeClassifier(cascadeFile.getAbsolutePath());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    @Override
    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    @Override
    // 这里执行人脸检测的逻辑, 根据OpenCV提供的例子实现(face-detection)
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        frameCount += 1;
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        // 翻转矩阵以适配前后置摄像头
        if (isFrontCamera) {
            Core.flip(mRgba, mRgba, 1);
            Core.flip(mGray, mGray, 1);
        } else {
            Core.flip(mRgba, mRgba, -1);
            Core.flip(mGray, mGray, -1);
        }
        float mRelativeFaceSize = 0.2f;
        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
        }

        MatOfRect faces = new MatOfRect();
        if (classifier != null)
            classifier.detectMultiScale(mGray, faces, 1.1, 2, 2,
                    new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        Rect[] facesArray = faces.toArray();
        Scalar faceRectColor = new Scalar(0, 255, 0, 255);

        //=========== Emotion recognition ==============
        if (frameCount == 2) {
            Bitmap bitmap = Bitmap.createBitmap(mGray.cols(), mGray.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(mRgba, bitmap);
            Matrix matrix = new Matrix();
            matrix.postScale((float) 0.3, (float) 0.3);
//            matrix.postRotate(90);
            bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(),
                    matrix, true);
            landmark.setFaceBitmap(bitmap);
            landmark.calculateLandmark();
            emotionLandmark.setLandmarks(landmark.getLandmarks());
            emotionResult = emotionLandmark.predict();
            //========================================

            frameCount = 0;
        }
        handlerMain.sendEmptyMessage(SHOW_EMOTION);   // Update emotion result

//        canvas = new Canvas(mRgba);
        for (Rect faceRect : facesArray) {
            Imgproc.rectangle(mRgba, faceRect.tl(), faceRect.br(), faceRectColor, 3);
        }
        return mRgba;
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (cameraView != null) {
            cameraView.disableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraView.disableView();
    }
}
