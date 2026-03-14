import os
import sys
import cv2
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.append(r"LG-Robustsam")

from lg_robust_sam import sam_model_registry
from lg_robust_sam.utils.transforms import ResizeLongestSide

def show_points(coords, labels, ax, marker_size=220):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]

    if len(pos_points) > 0:
        ax.scatter(pos_points[:, 0], pos_points[:, 1],
                   color='red', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)

    if len(neg_points) > 0:
        ax.scatter(neg_points[:, 0], neg_points[:, 1],
                   color='green', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)

def clean_mask_for_vis(mask, thresh=0.5, open_kernel=3, close_kernel=5):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    bin_mask = (mask > thresh).astype(np.uint8)

    if open_kernel > 0:
        k_open = np.ones((open_kernel, open_kernel), np.uint8)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, k_open)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    main_mask = np.zeros_like(bin_mask, dtype=np.uint8)

    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        main_mask[labels == largest_label] = 1
    else:
        main_mask = bin_mask.copy()

    if close_kernel > 0:
        k_close = np.ones((close_kernel, close_kernel), np.uint8)
        main_mask = cv2.morphologyEx(main_mask, cv2.MORPH_CLOSE, k_close)

    return main_mask.astype(np.float32)

def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255], dtype=np.float32)

    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    h, w = mask.shape[-2:]
    mask = mask.reshape(h, w).astype(np.float32)

    alpha = mask * 0.65
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)

    colored_mask = np.zeros((h, w, 4), dtype=np.float32)
    colored_mask[..., :3] = color
    colored_mask[..., 3] = alpha

    ax.imshow(colored_mask)

def robust_imread(image_path):
    data = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
opt = parser.parse_args()

image_path = r"demo-img/FOG.jpg"
checkpoint_path = r"ckpt_stage2_epoch20_clean.pth"

point_list = [
    np.array([[145, 76], [127, 186], [240, 155]], dtype=np.float32),
]

device = f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu"

class Opt:
    pass

model_opt = Opt()
model_opt.gpu = opt.gpu
model_opt.model_size = "tiny_vit"
model_opt.checkpoint_path = checkpoint_path

model = sam_model_registry["tiny_vit"](opt=model_opt, checkpoint=checkpoint_path)
model = model.to(device)
model.eval()

sam_transform = ResizeLongestSide(model.image_encoder.img_size)

image = robust_imread(image_path)
prompt = point_list[0]

image_t = torch.tensor(image, dtype=torch.uint8).unsqueeze(0).to(device)
image_t = torch.permute(image_t, (0, 3, 1, 2))
image_t_transformed = sam_transform.apply_image_torch(image_t.float())

point_t = torch.tensor(prompt, dtype=torch.float32, device=device)
input_label = torch.ones(prompt.shape[0], dtype=torch.int64, device=device)

data_dict = {
    "image": image_t_transformed,
    "point_coords": sam_transform.apply_coords_torch(point_t, image_t.shape[-2:]).unsqueeze(0),
    "point_labels": input_label.unsqueeze(0),
    "original_size": image_t.shape[-2:]
}

with torch.no_grad():
    batched_output = model.predict(
        model_opt,
        [data_dict],
        multimask_output=False,
        return_logits=False
    )

output_mask = batched_output[0]["masks"][0][0]
cleaned_mask = clean_mask_for_vis(output_mask, thresh=0.5, open_kernel=3, close_kernel=5)

save_dir = os.path.join("demo_result", "lgrobustsam_point_tiny_vit")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "FOG.png")

plt.figure(figsize=(10, 10))
plt.imshow(image)
show_points(prompt, np.ones(prompt.shape[0], dtype=np.int32), plt.gca())
show_mask(cleaned_mask, plt.gca())
plt.axis("off")
plt.savefig(save_path, bbox_inches="tight", dpi=150, pad_inches=0)
plt.close()

print("结果已保存到:", save_path)