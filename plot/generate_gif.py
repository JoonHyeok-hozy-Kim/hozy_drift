import os
import glob
import argparse
import re
from PIL import Image

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def create_gif(input_dir, output_filename, duration=200, scale=0.3):
    valid_extensions = ('*.png', '*.jpg', '*.jpeg')
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        
    if not image_paths:
        print(f"Image not found at : {input_dir}")
        return

    image_paths.sort(key=natural_sort_key)
    
    print(f"✅ {len(image_paths)} image(s) are found. Generating gif file...")
    for i, path in enumerate(image_paths[:5]):
        print(f"  [{i+1}] {os.path.basename(path)}")
    if len(image_paths) > 5:
        print("  ... (skipped) ...")
    
    frames = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        if scale != 0.1:
            ww = int(img.width * scale)
            hh = int(img.height * scale)
            try:
                resample_method = Image.Resampling.LANCZOS
            except AttributeError:
                resample_method = Image.LANCZOS
            img = img.resize((ww, hh), resample_method)
            
        frames.append(img)
        
    frames[0].save(
        output_filename,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=duration,  
        loop=0              
    )
    print(f"🎉 Success! The file '{output_filename}' has been created.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="디렉토리 내의 이미지들을 정렬하여 GIF로 변환합니다.")
    parser.add_argument('--dir', type=str, required=True, help="이미지가 저장된 디렉토리 경로 (필수)")
    args = parser.parse_args()
    
    output_dir = "results"
    output_dir = os.path.join(output_dir, "plot")
    
    curr_file_name = os.path.basename(__file__).split('.')[0]
    output_dir = os.path.join(output_dir, curr_file_name)
    
    script_dir = args.dir.split('/')[-4]
    output_dir = os.path.join(output_dir, script_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    target_name = args.dir.split('/')[-2]
    output_filename = target_name + ".gif"
    output_file_path = os.path.join(output_dir, output_filename)
    
    
    create_gif(args.dir, output_file_path)