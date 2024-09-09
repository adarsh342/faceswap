# run.py

import os
from PIL import Image
from face_swapper import download_model, process

def main():
    # User input for image paths
    source_img_paths = input("Enter the paths for source images, separated by commas: ").split(',')
    target_img_path = input("Enter the path for the target image: ")

    # Load images
    source_img = [Image.open(img_path.strip()) for img_path in source_img_paths]
    target_img = Image.open(target_img_path.strip())

    # Download model from Hugging Face if not already downloaded
    model_url = "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128_fp16.onnx"
    model_path = download_model(model_url, "./checkpoints/inswapper_128.onnx", force_download=True)

    # Process the images
    result_image = process(source_img, target_img, "-1", "-1", model_path)

    # Save the result
    result_image.save("result.png")
    print(f'Result saved successfully: result.png')

if __name__ == "__main__":
    main()
