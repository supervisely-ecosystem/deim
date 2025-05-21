from engine.core import YAMLConfig
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
import os
from time import time
from tqdm import tqdm
import numpy as np


def load_model(config_path, checkpoint, device, w, h):
    # Download checkpoint if it's a URL
    if checkpoint.startswith("http"):
        base_name = os.path.basename(checkpoint)
        checkpoint_path = os.path.join("weights", base_name)
        if not os.path.exists(checkpoint_path):
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            print(f"Downloading checkpoint from {checkpoint} to {checkpoint_path}")
            torch.hub.download_url_to_file(checkpoint, checkpoint_path)
    else:
        checkpoint_path = checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    cfg.yaml_cfg["eval_spatial_size"] = [h, w]

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    model = cfg.model
    # model.load_state_dict(state, strict=False)
    model.deploy().to(device)
    postprocessor = cfg.postprocessor.deploy().to(device)
    return model, postprocessor

def inference(model, postprocessor, im_pil, device, W, H):
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)

    transforms = T.Compose([
        T.Resize((H, W)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    outputs = model(im_data)
    outputs = postprocessor(outputs, orig_size)
    return outputs

def prepare_img(im_pil, W, H):
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)

    transforms = T.Compose([
        T.Resize((H, W)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil).unsqueeze(0).to(device)
    return im_data, orig_size

def transform(im_pil, transforms):
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)
    im_data = transforms(im_pil).unsqueeze(0).to(device)
    return im_data, orig_size

def inference_2(model, postprocessor, im_data, orig_size):
    outputs = model(im_data)
    outputs = postprocessor(outputs, orig_size)
    return outputs

device = "cuda"
file_path = "coco_sample.jpg"
# file_path = "box.png"
W, H = 1920, 1088
config_path = "configs/deim_dfine/deim_hgnetv2_n_coco.yml"
# config_path = "configs/deim_rtdetrv2/deim_r18vd_120e_coco.yml"
checkpoint = "weights/deim_dfine_hgnetv2_n_coco_160e.pth"

model, postprocessor = load_model(config_path, checkpoint, device, W, H)

print(f"image: {file_path}")
im_pil = Image.open(file_path).convert("RGB")

# warmup
outputs = inference(model, postprocessor, im_pil, device, W, H)
outputs = inference(model, postprocessor, im_pil, device, W, H)

# inference 1 run
t0 = time()
outputs = inference(model, postprocessor, im_pil, device, W, H)
print(f"Inference time: {(time() - t0)*1000:.2f} ms")
labels, boxes, scores = outputs

# speed test with 100 runs
def random_image(w, h):
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    im_pil = Image.fromarray(img)
    return im_pil

transforms = T.Compose([
    T.Resize((H, W)),
    T.ToTensor(),
])
N = 100
times = []
bar = tqdm(total=N, desc="Speed test")
for _ in range(N):
    im_pil_rnd = random_image(W, H)
    # im_data, orig_size = prepare_img(im_pil, W, H)
    t0 = time()
    im_data, orig_size = transform(im_pil_rnd, transforms)
    outputs = inference_2(model, postprocessor, im_data, orig_size)
    dt = (time() - t0) * 1000
    times.append(dt)
    bar.update(1)
    bar.set_postfix({"time (ms)": dt})

print(f"Average inference time over {N} runs: {sum(times)/len(times):.2f} ms")
# FPS
fps = 1000 / (sum(times) / len(times))
print(f"FPS: {fps:.2f}")

def draw(images, labels, boxes, scores, thrh=0.4):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]

        for j, b in enumerate(box):
            draw.rectangle(list(b), outline='red')
            draw.text((b[0], b[1]), text=f"{lab[j].item()} conf={round(scrs[j].item(), 2)}", fill='blue', )

        im.save('torch_results.png')

draw([im_pil], labels, boxes, scores, thrh=0.4)