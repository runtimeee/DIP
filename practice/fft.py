import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from PIL import Image

# 加载图像并转换为灰度
image = Image.open('/workspace/DIP/project1/src_imgs/hit.png').convert('L')
image_array = np.array(image)

# 应用DFT
dft_result = fft2(image_array)

# 中心化处理
dft_shifted = fftshift(dft_result)

# 计算幅度谱并进行对数变换以便于可视化
magnitude_spectrum = np.log(np.abs(dft_shifted) + 1)

# 创建一个图像对象，用于保存文件而不显示
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(magnitude_spectrum, cmap='gray')
ax.axis('off')  # 不显示坐标轴

# 保存图片
plt.savefig('magnitude_spectrum.png', bbox_inches='tight', pad_inches=0)

# 关闭图像，防止内存泄漏
plt.close(fig)