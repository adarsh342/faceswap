import os
import cv2
import copy
import numpy as np
from PIL import Image
from typing import List, Union
import insightface
import onnxruntime

def download_model(model_url: str, model_path: str, force_download=False):
    if not os.path.exists(model_path) or force_download:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.system(f"wget -O {model_path} {model_url}")
    return model_path

def getFaceSwapModel(model_path: str):
    model = insightface.model_zoo.get_model(model_path)
    return model

def getFaceAnalyser(model_path: str, providers, det_size=(320, 320)):
    face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", root="./checkpoints", providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    return face_analyser

def get_many_faces(face_analyser, frame:np.ndarray):
    try:
        face = face_analyser.get(frame)
        return sorted(face, key=lambda x: x.bbox[0])
    except IndexError:
        return None

def swap_face(face_swapper, source_faces, target_faces, source_index, target_index, temp_frame):
    source_face = source_faces[source_index]
    target_face = target_faces[target_index]
    return face_swapper.get(temp_frame, target_face, source_face, paste_back=True)

def process(source_img: Union[Image.Image, List], target_img: Image.Image, source_indexes: str, target_indexes: str, model: str):
    providers = onnxruntime.get_available_providers()
    face_analyser = getFaceAnalyser(model, providers)
    face_swapper = getFaceSwapModel(model)
    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    target_faces = get_many_faces(face_analyser, target_img)
    num_target_faces = len(target_faces)
    num_source_images = len(source_img)

    if target_faces is not None:
        temp_frame = copy.deepcopy(target_img)
        if isinstance(source_img, list) and num_source_images == num_target_faces:
            print("Replacing faces in target image from the left to the right by order")
            for i in range(num_target_faces):
                source_faces = get_many_faces(face_analyser, cv2.cvtColor(np.array(source_img[i]), cv2.COLOR_RGB2BGR))
                if source_faces is None:
                    raise Exception("No source faces found!")
                temp_frame = swap_face(face_swapper, source_faces, target_faces, i, i, temp_frame)
        elif num_source_images == 1:
            source_faces = get_many_faces(face_analyser, cv2.cvtColor(np.array(source_img[0]), cv2.COLOR_RGB2BGR))
            if source_faces is None:
                raise Exception("No source faces found!")
            for i in range(num_target_faces):
                temp_frame = swap_face(face_swapper, source_faces, target_faces, 0, i, temp_frame)
        else:
            raise Exception("Unsupported face configuration")
        result = temp_frame
    else:
        print("No target faces found!")
    
    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    return result_image

if __name__ == "__main__":
    # Upload files using the Colab interface
    from google.colab import files
    uploaded = files.upload()
    
    # Set filenames based on uploaded files
    source_img_path = 'source.jpg'  # Replace with the actual filename if different
    target_img_path = 'target.jpg'  # Replace with the actual filename if different

    # Load images
    source_img = [Image.open(source_img_path)]
    target_img = Image.open(target_img_path)

    # Download model from Hugging Face if not already downloaded
    model_url = "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128_fp16.onnx"
    model_path = download_model(model_url, "./checkpoints/inswapper_128.onnx", force_download=True)

    # Process the images
    result_image = process(source_img, target_img, "-1", "-1", model_path)
    
    # Save the result
    result_image.save("result.png")
    print(f'Result saved successfully: result.png')
