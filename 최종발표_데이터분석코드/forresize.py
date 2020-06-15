from PIL import Image
source_image = "C:/Users/YEOREUM/Desktop/ss.jpg"
target_image = "C:/Users/YEOREUM/Desktop/ss2.jpg"
image = Image.open(source_image)
# resize 할 이미지 사이즈 
resize_image = image.resize((64,64))
# 저장할 파일 Type : JPEG, PNG 등 
# 저장할 때 Quality 수준 : 보통 95 사용 
resize_image.save(target_image, "JPEG", quality=95 )