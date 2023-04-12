# 导入所需的库
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

# 定义一些常量或参数
EYE_AR_THRESH = 0.2 # 眼部纵横比阈值，用于判断眼睛是否闭合
EYE_AR_CONSEC_FRAMES = 3 # 连续多少帧低于阈值才认为眼睛闭合
BLINK_DURATION = 0.1 # 假设每次眨眼持续0.1秒
P80_THRESH = 0.012 # p80阈值，用于判断是否出现疲劳驾驶

# 加载dlib库中的shape_predictor
predictor = dlib.shape_predictor("D:\shiyan\p\shape_predictor_68_face_landmarks.dat")
# 加载dlib库中的face_detector
detector = dlib.get_frontal_face_detector()
# 定义一个函数来计算眼部纵横比
def eye_aspect_ratio(eye):
    # 计算眼睛两个垂直方向上的距离
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # 计算眼睛水平方向上的距离
    C = dist.euclidean(eye[0], eye[3])
    # 计算眼部纵横比
    ear = (A + B) / (2.0 * C)
    return ear

# 定义一个函数来计算perclos值
def perclos(blink_count, total_frames, fps):
    # 计算眨眼总时间
    blink_duration = blink_count * BLINK_DURATION
    # 计算视频总时间
    total_duration = total_frames / fps
    # 计算perclos值
    perclos_value = blink_duration / total_duration
    return perclos_value

# 定义一个函数来检测眼睛状态并返回perclos值和疲劳驾驶标志
def detect_eye_state(frame, counter, total):
    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 检测人脸
    faces = detector(gray)
    for face in faces:
        # 获取人脸关键点坐标
        landmarks = predictor(gray, face)
        # 获取左眼和右眼的坐标点
        leftEye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
        rightEye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])
        # 计算左眼和右眼的眼部纵横比
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # 计算两只眼睛的平均眼部纵横比
        ear = (leftEAR + rightEAR) / 2.0
        # 用凸包函数来获取眼睛轮廓
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        # 在图像上绘制眼睛轮廓
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        # 判断眼部纵横比是否低于阈值
        if ear < EYE_AR_THRESH:
            # 如果低于阈值，计数器加一
            counter += 1
        else:
            # 如果高于阈值，判断计数器是否大于等于设定的帧数
            if counter >= EYE_AR_CONSEC_FRAMES:
                # 如果大于等于设定的帧数，总眨眼次数加一
                total += 1
            # 重置计数器
            counter = 0
        # 在图像上显示眨眼次数和眼部纵横比
        cv2.putText(frame, "Blinks: {}".format(total), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # 返回计数器，总眨眼次数，以及当前帧
    return counter, total, frame

# 定义一个函数来计算perclos值
def perclos(blink_count, window_duration):
    # 计算眨眼总时间
    blink_duration = blink_count * BLINK_DURATION
    # 计算perclos值
    perclos_value = blink_duration / window_duration
    return perclos_value

# 创建一个视频捕捉对象，参数为摄像头编号或视频文件路径
cap = cv2.VideoCapture(0)

# 获取视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)

# 初始化计数器和总眨眼次数
counter = 0
total = 0

# 初始化perclos值和疲劳驾驶标志
perclos_value = 0.0
fatigue_driving = False

# 定义一个时间窗口的大小和起始时间，单位为秒
window_size = 1.0
window_start = 0.0

# 循环读取视频帧，直到视频结束或按下q键退出
while True:
    # 获取一帧图像，ret为布尔值，表示是否成功获取，frame为图像数组
    ret, frame = cap.read()
    if not ret:
        break

    # 调用函数检测眼睛状态并返回计数器和总眨眼次数
    counter, total, frame = detect_eye_state(frame, counter, total)

    # 获取当前视频的时间位置，单位为秒
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    # 判断当前时间是否超过窗口的结束时间
    if current_time > window_start + window_size:
        # 如果超过窗口的结束时间，计算perclos值，并重置窗口的起始时间和眨眼次数
        perclos_value = perclos(total, window_size)
        window_start = current_time
        total = 0

        # 判断perclos值是否超过p80阈值，如果超过则认为出现疲劳驾驶，并在图像上显示警告信息
        if perclos_value > P80_THRESH:
            fatigue_driving = True
            cv2.putText(frame, "Fatigue Driving!", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        else:
            fatigue_driving = False

    # 在图像上显示perclos值和帧率
    cv2.putText(frame, "PERCLOS: {:.4f}".format(perclos_value), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "FPS: {:.2f}".format(fps), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 显示图像
    cv2.imshow("Frame", frame)

    # 等待按键输入，如果是q键则退出循环
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# 释放视频捕捉对象和窗口
cap.release()
cv2.destroyAllWindows()