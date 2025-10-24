import os
import argparse
from PIL import Image
from pathlib import Path

def crop_and_save_images(input_dir, output_dir):
    """
    ?????????????????4??????4??
    ????????????????????
    """
    # ??????
    output_folders = {
        'rgb': os.path.join(output_dir, 'unigs_rgb'),
        'depth': os.path.join(output_dir, 'unigs_depth'),
        'normal': os.path.join(output_dir, 'unigs_normal'),
        'seg': os.path.join(output_dir, 'unigs_seg')
    }
    
    for folder in output_folders.values():
        os.makedirs(folder, exist_ok=True)
    
    # ???????
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # ????????????
    for file_path in Path(input_dir).iterdir():
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            try:
                # ????
                with Image.open(file_path) as img:
                    width, height = img.size
                    
                    # ?????????
                    sub_img_height = height // 4
                    
                    # ?????????
                    for i, category in enumerate(['rgb', 'depth', 'normal', 'seg']):
                        # ?????? (left, upper, right, lower)
                        crop_area = (0, i * sub_img_height, width, (i + 1) * sub_img_height)
                        sub_img = img.crop(crop_area)
                        
                        # ???????
                        output_filename = f"{file_path.stem}_{category}{file_path.suffix}"
                        output_path = os.path.join(output_folders[category], output_filename)
                        
                        # ????
                        sub_img.save(output_path)
                        print(f"Saved: {output_path}")
                        
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='????4??????')
    parser.add_argument('input_dir', help='???????')
    parser.add_argument('output_dir', help='???????')
    
    args = parser.parse_args()
    
    crop_and_save_images(args.input_dir, args.output_dir)
    print("?????")
