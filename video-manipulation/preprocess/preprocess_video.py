import argparse
import os
import torch
import shutil
from pathlib import Path
import cv2
from PIL import Image
from transformers import AutoModelForOpticalFlow, AutoImageProcessor
from tqdm import tqdm


def load_raft_model(model_name="facebook/raft-small", weights_dir="weights"):
    os.makedirs(weights_dir, exist_ok=True)
    model = AutoModelForOpticalFlow.from_pretrained(model_name, cache_dir=weights_dir)
    processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=weights_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    return model, processor, device


def compute_flow(model, processor, img1, img2, device):
    if not isinstance(img1, Image.Image):
        img1 = Image.fromarray(img1)
    if not isinstance(img2, Image.Image):
        img2 = Image.fromarray(img2)
    inputs = processor(images=[img1, img2], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        flow = outputs.predicted_flow[0] if hasattr(outputs, 'predicted_flow') else outputs.flow[0]
    return flow.cpu()


def get_frame_paths(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        cap = cv2.VideoCapture(str(input_path))
        temp_dir = Path(input_path.parent) / "temp_frames"
        temp_dir.mkdir(exist_ok=True)
        frame_paths = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = temp_dir / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(str(frame_path))
            frame_count += 1
        cap.release()
        return frame_paths, True
    else:
        ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        paths = [str(p) for p in sorted(input_path.iterdir()) if p.suffix.lower() in ext]
        return paths, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Video file or image directory")
    parser.add_argument("--output", required=True, help="Output directory for .pt files")
    parser.add_argument("--weights", default="weights", help="Directory for model weights")
    parser.add_argument("--model", default="facebook/raft-small", help="Hugging Face model name")
    args = parser.parse_args()
    
    model, processor, device = load_raft_model(args.model, args.weights)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_paths, is_temp = get_frame_paths(args.input)
    temp_dir = Path(frame_paths[0]).parent if is_temp else None
    
    for i in tqdm(range(len(frame_paths) - 1)):
        img1 = cv2.cvtColor(cv2.imread(frame_paths[i]), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(frame_paths[i + 1]), cv2.COLOR_BGR2RGB)
        if img1 is None or img2 is None:
            continue
        flow = compute_flow(model, processor, img1, img2, device)
        torch.save(flow, output_dir / f"flow_{Path(frame_paths[i]).stem}.pt")
    
    if is_temp:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
