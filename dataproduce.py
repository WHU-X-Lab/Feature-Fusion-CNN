import Aalbumentations as A
import cv2
import os
# 图像处理函数 （图像路径， 存储路径，处理概率）
def trans_data(img_path, save_path, P):
    img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # image1 = A.HorizontalFlip(P,image=img)['image']
    image1 = A.Compose([
        #  # 对比度受限直方图均衡
        # #（Contrast Limited Adaptive Histogram Equalization）
        # A.CLAHE(),
        # # 随机旋转 90°
        #  A.RandomRotate90(),
        # # 转置
        #  A.Transpose(),
        # # 随机仿射变换
        #  A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
        # # 模糊
        #  A.Blur(blur_limit=3),
        # # 光学畸变
        #  A.OpticalDistortion(),
        # # 网格畸变
        # A. GridDistortion(),
        # # 随机改变图片的 HUE、饱和度和值
        #  A.HueSaturationValue()
        # 垂直轴翻转
        # A.HorizontalFlip(P),
        # 水平轴翻转
        # A.VerticalFlip(P),
        # 旋转角度     
        # A.Rotate(limit=[90,90], interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=1),
        # A.Rotate(limit=[-90,-90], interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=1),
        # A.Rotate(limit=[-45,-45], interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=1),
        A.Rotate(limit=[45,45], interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=1),

        ],p=P)(image=img)['image']
    # image1 = cv2.cvtColor(image1,cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, image1)
    

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    image_dir = r'origin_diff\1'
    # image_dir = r'test_img'
    save_path = r'origin_diff\1'
    # save_path = r'test_img'
    for jpeg in os.listdir(image_dir):
        test_img = os.path.join(image_dir, jpeg)
        jpeg1 = 'HrzF_'+jpeg
        jpeg2 = 'Vrt_'+ jpeg
        jpeg3 = 'Fli_'+ jpeg 
        jpeg4 = 'R90_'+ jpeg
        jpeg5 = 'R-90_'+ jpeg
        jpeg6 = 'R-45_'+ jpeg
        jpeg7 = 'R45_'+ jpeg
        save_path_img = os.path.join(save_path,jpeg4)
        trans_data(test_img,save_path_img,1)
