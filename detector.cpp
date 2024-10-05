2// 版权所有 (c) 2022 ChenJun
// 采用MIT许可证。

// 引入OpenCV相关头文件
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

// 引入标准库相关头文件
#include <algorithm>
#include <cmath>
#include <vector>

// 引入自定义消息类型头文件
#include "armor_detector/detector.hpp"
#include "auto_aim_interfaces/msg/debug_armor.hpp"
#include "auto_aim_interfaces/msg/debug_light.hpp"

// 命名空间rm_auto_aim开始
namespace rm_auto_aim
{
// Detector类的构造函数
Detector::Detector(
  const int & bin_thres, const int & color, const LightParams & l, const ArmorParams & a)
: binary_thres(bin_thres), detect_color(color), l(l), a(a)
{
}
/**
 * @brief 主函数, 检测装甲板并返回装甲板列表
 * 
 * @return 返回装甲板列表
 * @note 函数执行顺序为：预处理图像 -> 寻找灯源 -> 匹配灯源 -> 提取数字 -> 分类数字 -> 返回装甲板列表
 * @note 函数返回装甲板列表后，会根据装甲板的类型进行分类，并计算装甲板的中心坐标和旋转角度
 */
std::vector<Armor> Detector::detect(const cv::Mat & input)
{
  binary_img = preprocessImage(input);      // 预处理图像并获取二值图像
  lights_ = findLights(input, binary_img);  // 在图像中查找灯源
  armors_ = matchLights(lights_);           // 匹配并组合灯源，形成可能的装甲板

  if (!armors_.empty()) {  // 如果检测到装甲板，则进一步提取和分类数字
    classifier->extractNumbers(input, armors_);
    classifier->classify(armors_);
  }

  return armors_;  // 返回检测到的装甲板列表
}

/**
 * 预处理输入的RGB图像，转换为二值图像
 * 
 * @param rgb_img RGB图像
 * @return 返回一个二值图像
 */
cv::Mat Detector::preprocessImage(const cv::Mat & rgb_img)
{
  cv::Mat gray_img;  // 将RGB图像转换为灰度图像
  cv::cvtColor(rgb_img, gray_img, cv::COLOR_RGB2GRAY);

  cv::Mat binary_img;  // 根据二值化阈值将灰度图像转换为二值图像
  cv::threshold(gray_img, binary_img, binary_thres, 255, cv::THRESH_BINARY);

  return binary_img;
}

/**
 * 在给定的RGB图像和二值图像中查找灯源并返回一个二值图像
 * 
 * @param rgb_img RGB图像
 * @param binary_img 二值图像
 * @return 返回一个二值图像
 */
std::vector<Light> Detector::findLights(const cv::Mat & rbg_img, const cv::Mat & binary_img)
{
  using std::vector;
  vector<vector<cv::Point>> contours;
  vector<cv::Vec4i> hierarchy;
  // 在二值图像中查找轮廓
  cv::findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  vector<Light> lights;
  this->debug_lights.data.clear();

  // 遍历所有轮廓，寻找符合条件的灯源
  for (const auto & contour : contours) {
    if (contour.size() < 5) continue;

    auto r_rect = cv::minAreaRect(contour);
    auto light = Light(r_rect);

    if (isLight(light)) {
      auto rect = light.boundingRect();
      // 确保矩形在图像范围内
      if (  // 避免断言失败
        0 <= rect.x && 0 <= rect.width && rect.x + rect.width <= rbg_img.cols && 0 <= rect.y &&
        0 <= rect.height && rect.y + rect.height <= rbg_img.rows) {
        int sum_r = 0, sum_b = 0;
        auto roi = rbg_img(rect);
        // 遍历ROI内的每个像素点，计算红色和蓝色像素值之和
        for (int i = 0; i < roi.rows; i++) {
          for (int j = 0; j < roi.cols; j++) {
            if (cv::pointPolygonTest(contour, cv::Point2f(j + rect.x, i + rect.y), false) >= 0) {
              // 如果点在轮廓内
              sum_r += roi.at<cv::Vec3b>(i, j)[0];
              sum_b += roi.at<cv::Vec3b>(i, j)[2];
            }
          }
        }
        // 红色像素值之和大于蓝色像素值之和，则认为是红色灯源，否则为蓝色灯源
        light.color = sum_r > sum_b ? RED : BLUE;
        lights.emplace_back(light);
      }
    }
  }

  return lights;
}

/**
 * 判断给定的灯条对象是否符合识别为灯条的标准
 * 主要通过灯条对象的长宽比和倾斜角度来判断
 * 
 * @param light 灯条对象引用
 * @return 返回一个布尔值，表示该灯条对象是否被识别为灯条
 */
bool Detector::isLight(const Light & light)
{
  // 计算灯条的长宽比（短边 / 长边）
  float ratio = light.width / light.length;
  bool ratio_ok = l.min_ratio < ratio && ratio < l.max_ratio;
  bool angle_ok = light.tilt_angle < l.max_angle;
  bool is_light = ratio_ok && angle_ok;

  // 填充调试信息
  auto_aim_interfaces::msg::DebugLight light_data;
  light_data.center_x = light.center.x;
  light_data.ratio = ratio;
  light_data.angle = light.tilt_angle;
  light_data.is_light = is_light;
  this->debug_lights.data.emplace_back(light_data);

  return is_light;
}

/**
 * 根据灯条列表匹配装甲板对象
 * 
 * @param lights 灯条列表
 * @return 返回一个装甲板对象列表
 */
std::vector<Armor> Detector::matchLights(const std::vector<Light> & lights)
{
  std::vector<Armor> armors;
  this->debug_armors.data.clear();

  // 遍历所有可能的灯条配对组合
  for (auto light_1 = lights.begin(); light_1 != lights.end(); light_1++) {
    for (auto light_2 = light_1 + 1; light_2 != lights.end(); light_2++) {
      if (light_1->color != detect_color || light_2->color != detect_color) continue;

      if (containLight(*light_1, *light_2, lights)) {
        continue;
      }

      auto type = isArmor(*light_1, *light_2);
      if (type != ArmorType::INVALID) {
        auto armor = Armor(*light_1, *light_2);
        armor.type = type;
        armors.emplace_back(armor);
      }
    }
  }

  return armors;
}

/**
 * 检查边界中是否有其他灯由两个灯形成
 * 
 * @param light_1 第一个灯对象
 * @param light_2 第二个灯对象
 * @param lights 所有灯对象列表
 * @return 返回一个布尔值，表示边界中是否有其他灯
 */
bool Detector::containLight(
  const Light & light_1, const Light & light_2, const std::vector<Light> & lights)
{
  // 收集两个灯的四个点形成点集，并计算边界框
  auto points = std::vector<cv::Point2f>{light_1.top, light_1.bottom, light_2.top, light_2.bottom};
  auto bounding_rect = cv::boundingRect(points);

  // 遍历所有灯，检查是否在边界框内
  for (const auto & test_light : lights) {
    if (test_light.center == light_1.center || test_light.center == light_2.center) continue;

    if (
      bounding_rect.contains(test_light.top) || bounding_rect.contains(test_light.bottom) ||
      bounding_rect.contains(test_light.center)) {
      return true;
    }
  }

  return false;
}

/**
 * 判断两个灯是否构成装甲板，并确定装甲板类型
 * 
 * @param light_1 第一个灯对象
 * @param light_2 第二个灯对象
 * @return 返回装甲板类型（大、小或无效）
 */
ArmorType Detector::isArmor(const Light & light_1, const Light & light_2)
{
  // 计算两个灯长度比（短边/长边）
  float light_length_ratio = light_1.length < light_2.length ? light_1.length / light_2.length
                                                             : light_2.length / light_1.length;
  bool light_ratio_ok = light_length_ratio > a.min_light_ratio;

  // 计算两个灯中心距离（单位：灯的平均长度）
  float avg_light_length = (light_1.length + light_2.length) / 2;
  float center_distance = cv::norm(light_1.center - light_2.center) / avg_light_length;
  bool center_distance_ok = (a.min_small_center_distance <= center_distance &&
                             center_distance < a.max_small_center_distance) ||
                            (a.min_large_center_distance <= center_distance &&
                             center_distance < a.max_large_center_distance);

  // 计算两灯中心连线的角度
  cv::Point2f diff = light_1.center - light_2.center;
  float angle = std::abs(std::atan(diff.y / diff.x)) / CV_PI * 180;
  bool angle_ok = angle < a.max_angle;

  // 判断是否为装甲板
  bool is_armor = light_ratio_ok && center_distance_ok && angle_ok;

  // 判断装甲板类型
  ArmorType type;
  if (is_armor) {
    type = center_distance > a.min_large_center_distance ? ArmorType::LARGE : ArmorType::SMALL;
  } else {
    type = ArmorType::INVALID;
  }

  // 填充调试信息
  auto_aim_interfaces::msg::DebugArmor armor_data;
  armor_data.type = ARMOR_TYPE_STR[static_cast<int>(type)];
  armor_data.center_x = (light_1.center.x + light_2.center.x) / 2;
  armor_data.light_ratio = light_length_ratio;
  armor_data.center_distance = center_distance;
  armor_data.angle = angle;
  this->debug_armors.data.emplace_back(armor_data);

  return type;
}

/**
 * 获取所有装甲板的数字图像
 * 
 * @return 返回一个OpenCV的Mat对象，包含所有装甲板的数字图像, 如果装甲板列表为空，则返回一个20x28大小的单通道灰度图
 */
cv::Mat Detector::getAllNumbersImage()
{
  if (armors_.empty()) {
    return cv::Mat(cv::Size(20, 28), CV_8UC1);
  } else {
    std::vector<cv::Mat> number_imgs;
    number_imgs.reserve(armors_.size());
    for (auto & armor : armors_) {
      number_imgs.emplace_back(armor.number_img);
    }
    cv::Mat all_num_img;
    cv::vconcat(number_imgs, all_num_img);
    return all_num_img;
  }
}

/**
 * 绘制检测结果
 * 
 * @param img 输入图像
 */
void Detector::drawResults(cv::Mat & img)
{
  // 绘制灯
  for (const auto & light : lights_) {
    cv::circle(img, light.top, 3, cv::Scalar(255, 255, 255), 1);
    cv::circle(img, light.bottom, 3, cv::Scalar(255, 255, 255), 1);
    auto line_color = light.color == RED ? cv::Scalar(255, 255, 0) : cv::Scalar(255, 0, 255);
    cv::line(img, light.top, light.bottom, line_color, 1);
  }

  // 绘制装甲板
  for (const auto & armor : armors_) {
    cv::line(img, armor.left_light.top, armor.right_light.bottom, cv::Scalar(0, 255, 0), 2);
    cv::line(img, armor.left_light.bottom, armor.right_light.top, cv::Scalar(0, 255, 0), 2);
  }

  // 显示数字及其置信度
  for (const auto & armor : armors_) {
    cv::putText(
      img, armor.classfication_result, armor.left_light.top, cv::FONT_HERSHEY_SIMPLEX, 0.8,
      cv::Scalar(0, 255, 255), 2);
  }
}

}  // namespace rm_auto_aim
