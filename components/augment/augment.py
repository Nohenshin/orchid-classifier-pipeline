import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Augment flowers dataset by rotation')
    parser.add_argument('--input_dir', required=True,
                        help='Thư mục gốc chứa các thư mục con theo loài hoa')
    parser.add_argument('--output_dir', required=True,
                        help='Thư mục đầu ra (cùng cấu trúc thư mục con)')
    parser.add_argument('--num_rotations', type=int, default=3,
                        help='Số góc xoay ngẫu nhiên cho mỗi ảnh (0-360)')
    return parser.parse_args()

def augment_image(img_path, output_dir, base_name, num_rotations):
    img = cv2.imread(img_path)
    if img is None:
        return
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    for i in range(num_rotations):
        angle = np.random.uniform(0, 360)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0,0,0))
        out_name = f"{base_name}_rot{i}.jpg"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, rotated)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    class_dirs = [d for d in os.listdir(args.input_dir)
                  if os.path.isdir(os.path.join(args.input_dir, d))]
    for class_name in class_dirs:
        class_input = os.path.join(args.input_dir, class_name)
        class_output = os.path.join(args.output_dir, class_name)
        os.makedirs(class_output, exist_ok=True)
        
        images = [f for f in os.listdir(class_input) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        for img_file in tqdm(images, desc=f"Augment {class_name}"):
            src_path = os.path.join(class_input, img_file)
            base = os.path.splitext(img_file)[0]
            # Copy ảnh gốc
            cv2.imwrite(os.path.join(class_output, img_file), cv2.imread(src_path))
            # Tạo ảnh xoay
            augment_image(src_path, class_output, base, args.num_rotations)
    print("Hoàn tất augmentation.")

if __name__ == '__main__':
    main()