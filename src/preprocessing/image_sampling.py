from PIL import Image
import os
import random
import shutil

def create_sample(input_dir:str, output_dir:str, sample_size:int, seed:int=42):
    '''
    This function shuffles the files in the specified directory and takes a sample of the specified size.

    Inputs: 
      input_dir (str): The input directory of the files.
      output_dir (str): The output directory for the files.
      sample_size (int): The number of samples.
      seed (int): The shuffling seed.
    '''

    os.makedirs(output_dir, exist_ok=True)

    files = os.listdir(input_dir)
    
    random.Random(seed).shuffle(files)
    print(f"Files shuffled with seed {seed}.")

    sample_files = files[:sample_size]

    for file in sample_files:
        shutil.copy(os.path.join(input_dir, file), os.path.join(output_dir, file))
    
    print("Sample creation completed.")
        

def resize_image(input_path:str, output_path:str, output_size:tuple=(64, 64)):
    '''
    This function resizes the image using LANCZOS downscaling.

    Inputs:
      image_path (str): The image input path.
      output_path (str): The image output path.
      output_size (tuple): The new image size.
    '''
    with Image.open(input_path) as img:
        # Convert to RGB if the image is in a different mode
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        aspect_ratio = img.width / img.height
        
        # Determine new dimensions while maintaining aspect ratio
        if aspect_ratio > 1:
            new_width = output_size[0]
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = output_size[1]
            new_width = int(new_height * aspect_ratio)
        
        # Resize the image using LANCZOS
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Create a new image with the target size and paste the resized image
        new_img = Image.new("RGB", output_size, (0, 0, 0))
        paste_x = (output_size[0] - new_width) // 2
        paste_y = (output_size[1] - new_height) // 2
        new_img.paste(img_resized, (paste_x, paste_y))
        
        # Save or return the image
        if output_path:
            new_img.save(output_path)
        
        return new_img


def downscale_images(input_dir:str, output_dir:str, output_size:tuple):
    '''
    This function loops through every image in the input directory and
    resizes the image using LANCZOS downscaling.

    Inputs:
      image_path (str): The image input path.
      output_path (str): The image output path.
      output_size (tuple): The new image size.
    '''

    os.makedirs(output_dir, exist_ok=True)

    files = os.listdir(input_dir)

    n = len(files)
        
    # Downscale images with LANCZOS
    for filename in files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"resized_{filename}")
        resize_image(input_path, output_path, output_size)
    
    print(f"Processed all {n} files")