import os
import cv2
import copy
import numpy as np
from PIL import Image
import insightface
import onnxruntime

# Download the required model
def download_model(model_url: str, model_path: str, force_download=False):
    if not os.path.exists(model_path) or force_download:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.system(f"wget -O {model_path} {model_url}")
    return model_path

# Load the face swap model
def getFaceSwapModel(model_path: str):
    model = insightface.model_zoo.get_model(model_path)
    return model

# Initialize face analyser
def getFaceAnalyser(model_path: str, providers, det_size=(320, 320)):
    face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", root="./checkpoints", providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    return face_analyser

# Get faces from an image
def get_many_faces(face_analyser, frame: np.ndarray):
    try:
        face = face_analyser.get(frame)
        return sorted(face, key=lambda x: x.bbox[0])
    except IndexError:
        return None

# Swap the faces between source and target
def swap_face(face_swapper, source_faces, target_faces, source_index, target_index, temp_frame):
    source_face = source_faces[source_index]
    target_face = target_faces[target_index]
    return face_swapper.get(temp_frame, target_face, source_face, paste_back=True)

# Process the face swapping
def process(source_img: Image.Image, target_img: Image.Image, model_path: str):
    providers = onnxruntime.get_available_providers()
    face_analyser = getFaceAnalyser(model_path, providers)
    face_swapper = getFaceSwapModel(model_path)

    # Convert target image to OpenCV format
    target_img_cv = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    target_faces = get_many_faces(face_analyser, target_img_cv)

    if target_faces is not None:
        temp_frame = copy.deepcopy(target_img_cv)
        # Use the first face for simplicity
        source_faces = get_many_faces(face_analyser, cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR))
        if source_faces is None:
            raise Exception("No source faces found!")
        for i in range(len(target_faces)):
            temp_frame = swap_face(face_swapper, source_faces, target_faces, 0, i, temp_frame)
        result = temp_frame
    else:
        raise Exception("No target faces found!")
    
    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    return result_image

if __name__ == "__main__":
    # Define paths for the source and target images
    source_img_path = "/content/source.jpg"  # Manually provide this path in Colab
    target_img_path = "/content/target.jpg"  # Manually provide this path in Colab

    # Load images
    source_img = Image.open(source_img_path)
    target_img = Image.open(target_img_path)

    # Download face-swapping model
    model_url = "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128_fp16.onnx"
    model_path = download_model(model_url, "./checkpoints/inswapper_128.onnx", force_download=True)

    # Perform face-swapping
    result_image = process(source_img, target_img, model_path)

    # Save the result
    result_image.save("result.png")
    print("Face swap completed. Result saved as result.png")
