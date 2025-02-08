import cv2
import os
from ultralytics import YOLO
import pytesseract

# بارگذاری مدل YOLO
model = YOLO('license_plate_detector.pt')

# بارگذاری تصویر
image_path = 'Cars1.png'
image = cv2.imread(image_path)

# استخراج نام فایل اصلی بدون پسوند
base_name, ext = os.path.splitext(os.path.basename(image_path))

# اجرای YOLO روی تصویر
results = model(image)
plate_class_id = 0

for result in results:
    for coordinates in result.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = coordinates
        if score > 0.5 and int(class_id) == plate_class_id:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # برش پلاک
            plate_image = image[int(y1):int(y2), int(x1):int(x2)]
            
            # ساخت نام جدید برای تصویر برش خورده
            cropped_filename = f'cropped_{base_name}{ext}'
            
            # ذخیره تصویر
            cv2.imwrite(cropped_filename, plate_image)
            cv2.imshow('License Plate', plate_image)
            cv2.waitKey(0)

# OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# تشخیص متن پلاک
plate_text = pytesseract.image_to_string(plate_image, config='--psm 8').strip()

print(f'Detected plate text: {plate_text}')
print(f'Cropped image saved as: {cropped_filename}')
