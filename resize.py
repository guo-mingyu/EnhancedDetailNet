from PIL import Image

def resize_image(input_path, output_path):
    # 打开图像
    image = Image.open(input_path)

    # 缩放图像
    resized_image = image.resize((32, 32), Image.ANTIALIAS)

    # 保存缩小后的图像
    resized_image.save(output_path)

# 示例用法
input_image_path = './data/test.jpg'
output_image_path = 'test_dog.jpg'
resize_image(input_image_path, output_image_path)
