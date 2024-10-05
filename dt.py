import cv2
import numpy as np
from enum import Enum

# 定义颜色常量
RED = 0
BLUE = 1

# 定义装甲类型
class ArmorType(Enum):
    SMALL = 0
    LARGE = 1
    INVALID = 2

# 定义灯条类
class Light:
    def __init__(self, box):
        self.color = None
        self.top = None
        self.bottom = None
        self.length = 0
        self.width = 0
        self.tilt_angle = 0
        
        points = cv2.boxPoints(box)
        points = sorted(points, key=lambda p: p[1])  # 按y坐标排序
        self.top = (points[0] + points[1]) / 2
        self.bottom = (points[2] + points[3]) / 2
        
        self.center = (self.top + self.bottom) / 2  # 添加center属性
        self.length = np.linalg.norm(self.top - self.bottom)
        self.width = np.linalg.norm(points[0] - points[1])
        
        self.tilt_angle = np.arctan2(abs(self.top[0] - self.bottom[0]), abs(self.top[1] - self.bottom[1])) * 180 / np.pi

# 定义装甲板类
class Armor:
    def __init__(self, light1, light2):
        if light1.center[0] < light2.center[0]:
            self.left_light = light1
            self.right_light = light2
        else:
            self.left_light = light2
            self.right_light = light1

        self.center = (self.left_light.center + self.right_light.center) / 2
        self.type = None
        self.number_img = None
        self.number = ""
        self.confidence = 0
        self.classification_result = ""

# 定义检测器类
class Detector:
    def __init__(self, bin_thres, color, light_params, armor_params):
        self.binary_thres = bin_thres
        self.detect_color = color
        self.light_params = light_params
        self.armor_params = armor_params
        self.lights = []
        self.armors = []

    def detect(self, input_img):
        binary_img = self.preprocess_image(input_img)
        self.lights = self.find_lights(input_img, binary_img)
        self.armors = self.match_lights(self.lights)

        # 这里可以添加分类数字的代码
        # if self.armors:
        #     self.classifier.extract_numbers(input_img, self.armors)
        #     self.classifier.classify(self.armors)

        return self.armors

    def preprocess_image(self, rgb_img):
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        _, binary_img = cv2.threshold(gray_img, self.binary_thres, 255, cv2.THRESH_BINARY)
        return binary_img

    def find_lights(self, rgb_img, binary_img):
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lights = []

        for contour in contours:
            if len(contour) < 5:
                continue

            r_rect = cv2.minAreaRect(contour)
            light = Light(r_rect)

            if self.is_light(light):
                rect = cv2.boundingRect(contour)
                roi = rgb_img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
                sum_r = np.sum(roi[:, :, 2])  # 红色通道
                sum_b = np.sum(roi[:, :, 0])  # 蓝色通道
                light.color = RED if sum_r > sum_b else BLUE
                lights.append(light)

        return lights

    def is_light(self, light):
        ratio = light.width / light.length
        ratio_ok = self.light_params['min_ratio'] < ratio < self.light_params['max_ratio']
        angle_ok = light.tilt_angle < self.light_params['max_angle']
        return ratio_ok and angle_ok

    def match_lights(self, lights):
        armors = []
        for i, light1 in enumerate(lights):
            for light2 in lights[i + 1:]:
                if light1.color != self.detect_color or light2.color != self.detect_color:
                    continue

                if self.contain_light(light1, light2, lights):
                    continue

                armor_type = self.is_armor(light1, light2)
                if armor_type != ArmorType.INVALID:
                    armor = Armor(light1, light2)
                    armor.type = armor_type
                    armors.append(armor)

        return armors

    def contain_light(self, light1, light2, lights):
        points = np.array([light1.top, light1.bottom, light2.top, light2.bottom])
        bounding_rect = cv2.boundingRect(points)

        # 获取矩形的坐标和尺寸
        x, y, w, h = bounding_rect

        for test_light in lights:
            if test_light == light1 or test_light == light2:
                continue

            # 检查灯条的顶点和中心是否在矩形内
            if (x <= test_light.top[0] <= x + w and y <= test_light.top[1] <= y + h) or \
            (x <= test_light.bottom[0] <= x + w and y <= test_light.bottom[1] <= y + h) or \
            (x <= test_light.center[0] <= x + w and y <= test_light.center[1] <= y + h):
                return True

        return False


    def is_armor(self, light1, light2):
        # 计算两个灯的长度比
        light_length_ratio = min(light1.length, light2.length) / max(light1.length, light2.length)
        light_ratio_ok = light_length_ratio > self.armor_params['min_light_ratio']

        # 计算中心距离
        avg_light_length = (light1.length + light2.length) / 2
        center_distance = np.linalg.norm(light1.center - light2.center) / avg_light_length
        center_distance_ok = (
            self.armor_params['min_small_center_distance'] <= center_distance < self.armor_params['max_small_center_distance'] or
            self.armor_params['min_large_center_distance'] <= center_distance < self.armor_params['max_large_center_distance']
        )

        # 计算角度
        diff = light1.center - light2.center
        angle = abs(np.arctan2(diff[1], diff[0])) * 180 / np.pi
        angle_ok = angle < self.armor_params['max_angle']

        is_armor = light_ratio_ok and center_distance_ok and angle_ok
        if is_armor:
            return ArmorType.LARGE if center_distance > self.armor_params['min_large_center_distance'] else ArmorType.SMALL
        return ArmorType.INVALID
    
    def draw_results(self, img):
        # 绘制灯
        for light in self.lights:
            cv2.circle(img, tuple(map(int, light.top)), 1, (255, 255, 255), -1)
            cv2.circle(img, tuple(map(int, light.bottom)), 1, (255, 255, 255), -1)
            line_color = (255, 255, 0) if light.color == RED else (255, 0, 255)
            cv2.line(img, tuple(map(int, light.top)), tuple(map(int, light.bottom)), line_color, 1)

        # 绘制装甲板
        # for armor in self.armors:
        #     cv2.line(img, tuple(map(int, armor.left_light.top)), tuple(map(int, armor.right_light.bottom)), (0, 255, 0), 2)
        #     cv2.line(img, tuple(map(int, armor.left_light.bottom)), tuple(map(int, armor.right_light.top)), (0, 255, 0), 2)

            # 显示分类结果
            # cv2.putText(img, armor.classification_result, tuple(map(int, armor.left_light.top)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return img
    
# 使用示例
if __name__ == "__main__":
    light_params = {'min_ratio': 0.1, 'max_ratio': 5, 'max_angle': 300}
    armor_params = {
        'min_light_ratio': 0.1,
        'min_small_center_distance': 2,
        'max_small_center_distance': 500,
        'min_large_center_distance': 5,
        'max_large_center_distance': 1000,
        'max_angle': 300
    }
    detector = Detector(bin_thres=128, color=RED, light_params=light_params, armor_params=armor_params)

    # 读取图像并进行检测
    img = cv2.imread("img/2.jpg")
    armors = detector.detect(img)

    print(armors)

    # 绘制结果
    result_img = detector.draw_results(img.copy())

    cv2.namedWindow("result_img", cv2.WINDOW_NORMAL)
    cv2.imshow("result_img", result_img)
