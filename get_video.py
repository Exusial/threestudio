import cv2
import os
from PIL import Image
import numpy as np

# 定义图片路径
image_folder = '/home/zjp/zjp/threestudio/outputs/dreamavatar-vsd/{}/save'.format('Elsa@20230807-213758')  # 替换为你的路径
#image_folder = '/home/penghy/diffusion/threestudio/outputs/dreamavatar-vsd/{}/save'.format('Captain_American_Full_Body@20230717-155644')  # 替换为你的路径
video_name = 'output.mp4'

# 获取文件夹中的图片列表
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
# 对图片列表进行排序，以确保按正确的顺序进行读取
images.sort(key=lambda img: os.path.getmtime(os.path.join(image_folder, img)))
print(images)

# 读取第一张图片以获取图像尺寸
sample = Image.open(os.path.join(image_folder, images[0]))
image_array = np.array(sample)
height, width, layers = image_array.shape

# 选择适当的编解码器，并定义视频文件的输出参数
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者使用 'X264'
video = cv2.VideoWriter(video_name, fourcc, 10.0, (width, height))

# 遍历每一张图片，并添加到视频中
for image in images:
    img_path = os.path.join(image_folder, image)
    img_pil = Image.open(img_path)
    img_array = np.array(img_pil)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # 将图像从 RGB 格式转换为 BGR 格式
    video.write(img_array)

cv2.destroyAllWindows()
video.release()