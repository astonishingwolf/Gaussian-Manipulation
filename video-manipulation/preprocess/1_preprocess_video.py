"""
python video-manipulation/preprocess/preprocess_video.py --input videos/00376.mp4 --output optical_flows --visualize

"""

import argparse
import os
import sys
import torch
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

DATA_DIR = './data'

def load_raft_model(weights_dir="weights", raft_path=None):
    raft_path = Path(raft_path or Path.cwd() / "RAFT")
    if not raft_path.exists():
        raise FileNotFoundError(f"RAFT not found at {raft_path}")
    sys.path.insert(0, str(raft_path))
    try:
        from core.raft import RAFT
    except ImportError:
        sys.path.insert(0, str(raft_path / "core"))
        from raft import RAFT
    os.makedirs(weights_dir, exist_ok=True)
    class Args:
        def __init__(self):
            self.small = False
            self.dropout = 0
            self.alternate_corr = False
            self.corr_levels = 4
            self.corr_radius = 4
            self.mixed_precision = False
        def __contains__(self, key):
            return hasattr(self, key)
    model = RAFT(Args())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights_path = next((p for p in [Path(weights_dir)/"raft-things.pth", raft_path/"models"/"raft-things.pth", raft_path/"raft-things.pth"] if p.exists()), None)
    if not weights_path:
        raise FileNotFoundError(f"RAFT weights not found. Download to {weights_dir}")
    state_dict = torch.load(weights_path, map_location=device)
    if 'module.' in list(state_dict.keys())[0]:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model, device

def compute_flow_raft(model, img1, img2, device):
    from core.utils.utils import InputPadder
    img1_t = torch.from_numpy(img1).permute(2,0,1).float().to(device)
    img2_t = torch.from_numpy(img2).permute(2,0,1).float().to(device)
    padder = InputPadder(img1_t.shape)
    img1_t, img2_t = padder.pad(img1_t[None], img2_t[None])
    with torch.no_grad():
        _, flow_up = model(img1_t, img2_t, iters=20, test_mode=True)
    return padder.unpad(flow_up[0]).cpu()

def compute_flow_opencv(img1, img2):
    g1, g2 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return torch.from_numpy(flow).permute(2,0,1)

def visualize_flow(flow, raft_path=None):
    if isinstance(flow, torch.Tensor):
        flow_np = flow.permute(1,2,0).numpy() if flow.dim()==3 and flow.shape[0]==2 else flow.numpy()
    else:
        flow_np = flow
    if raft_path:
        try:
            sys.path.insert(0, str(raft_path))
            from core.utils.flow_viz import flow_to_image
            return flow_to_image(flow_np)
        except:
            pass
    h, w = flow_np.shape[:2]
    mag = np.sqrt(flow_np[:,:,0]**2 + flow_np[:,:,1]**2)
    ang = np.arctan2(flow_np[:,:,1], flow_np[:,:,0])
    mag = np.clip(mag / (mag.max() + 1e-6), 0, 1) if mag.max() > 0 else np.zeros_like(mag)
    hsv = np.zeros((h,w,3), dtype=np.uint8)
    hsv[:,:,0] = ((ang + np.pi) / (2*np.pi) * 180).astype(np.uint8)
    hsv[:,:,1] = 255
    hsv[:,:,2] = (mag * 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def get_frame_paths(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        cap = cv2.VideoCapture(str(input_path))
        temp_dir = Path(input_path.parent) / "temp_frames"
        temp_dir.mkdir(exist_ok=True)
        frame_paths = []
        for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, frame = cap.read()
            if not ret:
                break
            p = temp_dir / f"frame_{i:06d}.jpg"
            cv2.imwrite(str(p), frame)
            frame_paths.append(str(p))
        cap.release()
        return frame_paths, True
    else:
        ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        return [str(p) for p in sorted(input_path.iterdir()) if p.suffix.lower() in ext], False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", default="weights")
    parser.add_argument("--raft-path", default="./RAFT")
    parser.add_argument("--method", choices=["raft","opencv"], default="raft")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    input_path = Path(DATA_DIR)/args.input if not Path(args.input).is_absolute() else Path(args.input)
    output_path = Path(DATA_DIR)/args.output if not Path(args.output).is_absolute() else Path(args.output)
    input_name = input_path.stem if input_path.is_file() else input_path.name
    output_dir = output_path / input_name
    output_dir.mkdir(parents=True, exist_ok=True)
    breakpoint()
    flow_dir = output_dir / "flow"
    flow_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = (output_dir/"vis") if args.visualize else None
    if viz_dir:
        viz_dir.mkdir(parents=True, exist_ok=True)
    if args.method == "raft":
        raft_path = Path(args.raft_path)
        model, device = load_raft_model(args.weights, raft_path)
        flow_fn = lambda i1, i2: compute_flow_raft(model, i1, i2, device)
    else:
        flow_fn = compute_flow_opencv
        raft_path = None
    frame_paths, is_temp = get_frame_paths(input_path)
    temp_dir = Path(frame_paths[0]).parent if is_temp else None
    for i in tqdm(range(len(frame_paths)-1)):
        img1 = cv2.cvtColor(cv2.imread(frame_paths[i]), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(frame_paths[i+1]), cv2.COLOR_BGR2RGB)
        if img1 is None or img2 is None:
            continue
        flow = flow_fn(img1, img2)
        frame_num = f"{i:06d}"
        torch.save(flow, flow_dir/f"{frame_num}.pt")
        if args.visualize and viz_dir:
            cv2.imwrite(str(viz_dir/f"{frame_num}.png"), cv2.cvtColor(visualize_flow(flow, raft_path), cv2.COLOR_RGB2BGR))
    if is_temp:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
