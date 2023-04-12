# 导入所需的库
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
detector = dlib.get_frontal_face_detector()
# 定义一些常量
EYE_AR_THRESH = 0.3 # 眼部纵横比的阈值，低于该值则认为眼睛闭合
EYE_AR_CONSEC_FRAMES = 3 # 连续多少帧低于阈值则认为眨眼
BLINK_DURATION = 0.1 # 每次眨眼的持续时间，单位为秒
P80_THRESH = 0.2 # perclos的p80阈值，高于该值则认为出现疲劳驾驶
FATIGUE_ALERT_DURATION = 2 # 疲劳驾驶提醒的持续时间，单位为秒

# 定义一个函数来计算眼部纵横比
def eye_aspect_ratio(eye):
    # 计算两组垂直眼睛标志（x，y） - 坐标之间的欧氏距离
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # 计算水平之间的欧氏距离
    # 眼睛标志（x，y） - 坐标
    C = dist.euclidean(eye[0], eye[3])

    # 计算眼睛的纵横比
    ear = (A + B) / (2.0 * C)

    # 返回眼睛的纵横比
    return ear

# 定义一个函数来检测眼睛状态并返回计数器和总眨眼次数
def detect_eye_state(frame, counter, total):
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用dlib库检测人脸
    rects = detector(gray, 0)

    # 遍历每一个检测到的人脸
    for rect in rects:
        # 获取人脸区域的坐标和大小
        (x, y, w, h) = rect_to_bb(rect)
        # 绘制人脸区域矩形框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 提取人脸区域，并使用预测器获取68个特征点
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        # 提取左眼和右眼对应的特征点，并计算它们的纵横比
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # 计算两只眼睛的平均纵横比
        ear = (leftEAR + rightEAR) / 2.0

        # 使用cv2.convexHull函数计算左眼和右眼的凸包，然后绘制出来
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # 检查眼部纵横比是否低于阈值，如果低于则增加计数器
        if ear < EYE_AR_THRESH:
            counter += 1

            # 否则，眼睛纵横比不低于阈值
        else:
            # 如果之前的帧中眼睛是闭合的
            if counter >= EYE_AR_CONSEC_FRAMES:
                # 增加总眨眼次数
                total += 1

            # 重置眼睛帧计数器
            counter = 0

            # 在图像上显示眨眼次数和眼睛纵横比
        cv2.putText(frame, "Blinks: {}".format(total), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.4f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 计算每秒的帧数
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 计算每次眨眼的持续帧数
    blink_frames = int(BLINK_DURATION * fps)

    # 计算perclos值，即每分钟内闭眼时间占总时间的百分比
    perclos_value = (counter + total * blink_frames) / (60 * fps)

    # 在图像上显示perclos值和fps值
    cv2.putText(frame, "PERCLOS: {:.4f}".format(perclos_value), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "FPS: {:.2f}".format(fps), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 检查perclos值是否高于阈值，如果高于则认为出现疲劳驾驶，并在图像上显示提醒信息
    # 使用code blocks语法来包装代码
    while True:  # 添加一个循环
        if perclos_value > P80_THRESH:
            cv2.putText(frame, "疲劳驾驶，请注意休息！", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # 设置一个定时器，让提醒信息持续一定时间
            timer = FATIGUE_ALERT_DURATION * fps
        else:
            timer = max(timer - 1, 0)

        # 如果定时器不为零，则继续显示提醒信息
        if timer > 0:
            cv2.putText(frame, "疲劳驾驶，请注意休息！", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 显示图像
        cv2.imshow("Frame", frame)

        # 等待按键输入，如果是q键则退出循环
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break  # 这里的break是跳出循环
# 创建一个视频捕捉对象，并打开摄像头
cap = cv2.VideoCapture(0)

# 使用code blocks语法来包装代码
# 初始化一个计数器和总眨眼次数
counter = 0
total = 0

# 定义fps变量，表示每秒帧数
fps = 30 # 这里可以根据你的摄像头实际情况修改

# 初始化一个定时器，用于显示疲劳驾驶提醒信息
timer = FATIGUE_ALERT_DURATION * fps

# 创建一个循环，不断从摄像头获取图像并处理
while True:
    # 获取一帧图像，并检查是否成功获取
    ret, frame = cap.read()
    if not ret:
        # break语句和if语句对齐，比while语句多一个缩进
        break

    # 调用detect_eye_state函数，传入图像，计数器和总眨眼次数，并获取返回值
    counter, total = detect_eye_state(frame, counter, total)

# 释放摄像头资源
cap.release()

# 关闭所有窗口
cv2.destroyAllWindows()