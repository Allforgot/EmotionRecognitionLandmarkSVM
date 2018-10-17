## EmwitEmotionRecognitionDemo

中悦轻量表情识别演示程序

### 算法实现

1. 利用 dlib 计算人脸68个关键点
2. 从68个关键点得出用于表情分类的特征值
3. 利用 OpenCV 的 SVM 进行分类

### 数据集

1. jaffe

### 软件使用说明

1. 先将 shape_predictor_68_face_landmarks.dat 文件复制到手机存储根目录
2. 只能单方向横屏识别