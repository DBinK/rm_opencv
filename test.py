import cv2
from loguru import logger

from detector import Detector


# 定义常量
RED = "red"
BLUE = "blue"
ARMOR_TYPE_STR = ["INVALID", "SMALL", "LARGE"]

# light_params 和 armor_params 应根据实际需求设置
light_params = {"min_ratio": 0.1, "max_ratio": 2000, "max_angle": 900}
armor_params = {
    "min_light_ratio": 0.1,
    "min_small_center_distance": 3,
    "max_small_center_distance": 1000,
    "min_large_center_distance": 10,
    "max_large_center_distance": 3000,
    "max_angle": 900,
}

detector = Detector(
    bin_thres=120, color=RED, light_params=light_params, armor_params=armor_params
)

# 读取图像并进行检测
input_image = cv2.imread("img/2.jpg")
# input_image = cv2.imread('combine.png')
detected_armors = detector.detect(input_image)

# 绘制检测结果
out = detector.draw_results(input_image)

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
            out = detector.draw_results(frame)

            logger.info(f"预测结果：\n{detected_armors}")

            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.imshow("frame", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    if cap is not None:
        cap.release()
