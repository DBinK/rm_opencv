import cv2
import numpy as np
from loguru import logger

def show_img(img):
    if img is not None:
        cv2.namedWindow("tmp", cv2.WINDOW_NORMAL)
        cv2.imshow("tmp", img)

class Light:
    def __init__(self, rect):
        self.rect = rect
        # 使用 rect[0] 获取中心点坐标
        self.center = tuple(map(int, rect[0]))
        self.top = (self.center[0], int(self.center[1] - rect[1][1] / 2))
        self.bottom = (self.center[0], int(self.center[1] + rect[1][1] / 2))
        self.length = max(rect[1])
        self.width = min(rect[1])
        self.tilt_angle = rect[2]
        self.color = None
    def calculate_tilt_angle(self):
        """计算倾斜角度"""
        # cv2.minAreaRect 返回的第三个元素是旋转角度
        _, _, angle = cv2.minAreaRect(np.array(self.rect))
        # 如果角度大于 0，则调整为负数表示逆时针方向
        return -angle if angle > 0 else angle


class Armor:
    def __init__(self, left_light, right_light):
        self.left_light = left_light
        self.right_light = right_light
        self.type = None
        self.number_img = None


class Detector:
    def __init__(self, binary_thres=100, detect_color="blue", light_params={}, armor_params={}):
        self.binary_thres = binary_thres
        self.detect_color = detect_color
        self.l = light_params
        self.a = armor_params
        self.armors_ = []
        self.lights_ = []

    def detect(self, input_img):
        """识别装甲板"""
        binary_img = self.preprocess_image(input_img)
        show_img(binary_img)
        self.lights_ = self.find_lights(input_img, binary_img)
        self.armors_ = self.match_lights(self.lights_)
       
        if self.armors_:
            # 这里可以添加数字提取和分类的逻辑
            pass
        
        return self.armors_

    def preprocess_image(self, rgb_img):
        """预处理图像"""
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        _, binary_img = cv2.threshold(
            gray_img, self.binary_thres, 255, cv2.THRESH_BINARY
        )
        return binary_img

    def find_lights(self, rgb_img, binary_img):
        """查找灯条"""
        contours, _ = cv2.findContours(
            binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        lights = []
        for contour in contours:
            if len(contour) < 5:  # 确保 contour 是一个有效的轮廓点集
                continue
            r_rect = cv2.minAreaRect(contour)  # 获取最小外接矩形
            light = Light(r_rect)
            if self.is_light(light):
                roi = rgb_img[
                    int(light.center[1] - light.rect[1][1] / 2): int(light.center[1] + light.rect[1][1] / 2),
                    int(light.center[0] - light.rect[1][0] / 2): int(light.center[0] + light.rect[1][0] / 2),
                ]
                sum_r = np.sum(roi[:, :, 2])
                sum_b = np.sum(roi[:, :, 0])
                light.color = "red" if sum_r > sum_b else "blue"
                lights.append(light)

        logger.info(f"lights: {lights}")

        return lights

    def is_light(self, light):       
        """判断是否为灯条"""
        ratio = light.width / light.length  # 计算长宽比
        ratio_ok = self.l["min_ratio"] < ratio < self.l["max_ratio"]  # 计算长宽比
        angle_ok = light.calculate_tilt_angle() < self.l["max_angle"]  # 调用新方法获取角度
        return ratio_ok and angle_ok

    def match_lights(self, lights):
        """匹配灯条"""
        armors = []
        for i, light_1 in enumerate(lights):
            for light_2 in lights[i + 1:]:
                if (
                    light_1.color != self.detect_color
                    or light_2.color != self.detect_color
                ):
                    continue
                if not self.contain_light(light_1, light_2, lights):
                    armor_type = self.is_armor(light_1, light_2)
                    if armor_type != "invalid":
                        armor = Armor(light_1, light_2)
                        armor.type = armor_type
                        armors.append(armor)

        logger.info(f"armors: {armors}")

        return armors

    def contain_light(self, light_1, light_2, lights):
        """判断两个给定的灯（light_1和light_2）形成的最小包围矩形内是否包含其他灯"""
        points = [light_1.top, light_1.bottom, light_2.top, light_2.bottom]
        bounding_rect = cv2.boundingRect(np.array(points))
        for test_light in lights:
            if (
                test_light.center == light_1.center
                or test_light.center == light_2.center
            ):
                continue
            if (
                bounding_rect[0]
                <= test_light.center[0]
                <= bounding_rect[0] + bounding_rect[2]
                and bounding_rect[1]
                <= test_light.center[1]
                <= bounding_rect[1] + bounding_rect[3]
            ):
                return True
        return False

    def is_armor(self, light_1, light_2):
        """判断是否为装甲板"""
        length_ratio = min(light_1.length, light_2.length) / max(
            light_1.length, light_2.length
        )
        center_distance = np.linalg.norm(
            np.array(light_1.center) - np.array(light_2.center)
        )
        avg_length = (light_1.length + light_2.length) / 2
        center_distance /= avg_length
        diff = np.array(light_1.center) - np.array(light_2.center)
        angle = abs(np.arctan2(diff[1], diff[0])) * 180 / np.pi
        if (
            length_ratio > self.a["min_light_ratio"]
            and (
                (
                    self.a["min_small_center_distance"]
                    <= center_distance
                    < self.a["max_small_center_distance"]
                )
                or (
                    self.a["min_large_center_distance"]
                    <= center_distance
                    < self.a["max_large_center_distance"]
                )
            )
            and angle < self.a["max_angle"]
        ):
            return (
                "large"
                if center_distance > self.a["min_large_center_distance"]
                else "small"
            )
        return "invalid"

    def draw_results(self, img):
        """绘制结果"""
        for light in self.lights_:
            cv2.circle(img, light.top, 3, (255, 255, 255), 1)
            cv2.circle(img, light.bottom, 3, (255, 255, 255), 1)
            line_color = (255, 255, 0) if light.color == "red" else (255, 0, 255)
            cv2.line(img, light.top, light.bottom, line_color, 1)

        for armor in self.armors_:
            cv2.line(
                img, armor.left_light.top, armor.right_light.bottom, (0, 255, 0), 2
            )
            cv2.line(
                img, armor.left_light.bottom, armor.right_light.top, (0, 255, 0), 2
            )


if __name__ == "__main__":


    img_path = "img/2.jpg"
    #img_path = "img/1.png"
    # img_path = "combine.png"

    img = cv2.imread(img_path)
    
    detector = Detector(
        binary_thres=100,
        detect_color="red",
        light_params={
            "min_ratio": 0.5,  # 灯条长宽比
            "max_ratio": 2.5,  
            "max_angle": 180,  # 灯条倾斜角度
            "max_length": 500  # 灯条最大长度
        },
        armor_params={
            "min_light_ratio": 0.5,            # 装甲板灯条长宽比
            "min_small_center_distance": 0.5,  # 装甲板灯条中心距离
            "max_small_center_distance":1.5,   
            "min_large_center_distance": 2.5,  # 装甲板灯条中心距离
            "max_large_center_distance": 3.5,  
            "max_angle": 30                    # 装甲板倾斜角度
        }
    )

    result = detector.detect(img)
    detector.draw_results(img)

    logger.info(result)

    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow("result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()