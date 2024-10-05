import cv2
from loguru import logger

from dt import Detector


# 定义常量
RED = "red"
BLUE = "blue"
ARMOR_TYPE_STR = ["INVALID", "SMALL", "LARGE"]

light_params = {'min_ratio': 0.1, 'max_ratio': 1, 'max_angle': 30}
armor_params = {
    'min_light_ratio': 0.001,
    'min_small_center_distance': 1,
    'max_small_center_distance': 5000,
    'min_large_center_distance': 1,
    'max_large_center_distance': 10000,
    'max_angle': 30
}
detector = Detector(bin_thres=80, color=RED, light_params=light_params, armor_params=armor_params)

# 读取摄像头并进行检测
# url = 0
url = "img/1.mp4"
cap = cv2.VideoCapture(url)

if cap is not None:
    while True:
        ret, frame = cap.read()
        if ret:
            # input_image = cv2.imread('combine.png')
            detected_armors = detector.detect(frame)

            # 绘制检测结果
            out = detector.draw_results(frame.copy())

            logger.info(f"预测结果：\n{detected_armors}")

            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.imshow("frame", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    if cap is not None:
        cap.release()
