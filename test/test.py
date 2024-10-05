import cv2
from loguru import logger


def darker_img(img):
    """降低亮度"""
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # 转换为 HSV 颜色空间
    hsv_image[:, :, 2] = hsv_image[:, :, 2] * 0.5  # 将 V 通道乘以 0.5
    darker_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)  # 转换回 BGR
    return darker_image

img = cv2.imread("img/2.jpg")

cvs_img = cv2.convertScaleAbs(img, alpha=0.5)
hsv_img = darker_img(img)

cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img", img)

cv2.namedWindow("cvs_img", cv2.WINDOW_NORMAL)
cv2.imshow("cvs_img", cvs_img)

cv2.namedWindow("hsv_img", cv2.WINDOW_NORMAL)
cv2.imshow("hsv_img", hsv_img)

cv2.waitKey(0)
cv2.destroyAllWindows()