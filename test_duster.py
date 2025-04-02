import os
import numpy as np
import torch
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb, enlarge_seg_masks
from dust3r.utils.device import to_numpy
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"
weights_path = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
output_dir = "outputs"
print('Outputting stuff in', output_dir)

input_dir = "/home/stefano/Data/davis/JPEGImages/480p/car-turn"

# Process images in the input directory with default parameters
if os.path.isdir(input_dir):    # input_dir is a directory of images
    input_files = [os.path.join(input_dir, fname) for fname in sorted(os.listdir(input_dir))]
else:
    raise ValueError(f"Invalid input directory: {input_dir}")

image_size = 512
silent = False
dynamic_mask_path = None

# fps = 0  # FPS for video processing
# num_frames = 200  # Maximum number of frames for video processing

# from a list of images, run dust3r inference

imgs, width, height, video_fps = load_images(
    input_dir,
    size=image_size,
    verbose=not silent,
    dynamic_mask_root=dynamic_mask_path,
    # fps=fps,
    # num_frames=num_frames,
    return_img_size=True
)

scenegraph_type = 'swinstride'
winsize = 3
scenegraph_type = scenegraph_type + "-" + str(winsize) + "-noncyclic"
pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
print(f"Number of pairs: {len(pairs)}")

first_pair = pairs[0]
first_img = first_pair[0]
img = first_img['img'].squeeze(0).permute(1, 2, 0)
# print(img.shape, img.dtype)
mask = first_img['mask'].squeeze(0)
# print(mask.shape, mask.dtype)
dynamic_mask = first_img['dynamic_mask'].squeeze(0)
# print(dynamic_mask.shape, dynamic_mask.dtype)

# # plot things side by side
# fig, ax = plt.subplots(1, 3, figsize=(15, 5))
# ax[0].imshow(rgb(to_numpy(img)))
# ax[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
# ax[2].imshow(dynamic_mask, cmap='gray', vmin=0, vmax=1)
# plt.show()

model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(device)
batch_size = 1
output = inference(pairs, model, device, batch_size=batch_size, verbose=not silent)

# print(output.keys())

view1 = output['view1']
view2 = output['view2']
pred1 = output['pred1']
pred2 = output['pred2']

view1_img = view1['img'][0].permute(1, 2, 0).cpu().numpy()
view1_mask = view1['mask'][0].cpu().numpy()
view1_dynamic_mask = view1['dynamic_mask'][0].cpu().numpy()

# # plot things side by side
# fig, ax = plt.subplots(1, 3, figsize=(15, 5))
# ax[0].imshow(view1_img)
# ax[1].imshow(view1_mask, cmap='gray', vmin=0, vmax=1)
# ax[2].imshow(view1_dynamic_mask, cmap='gray', vmin=0, vmax=1)
# plt.show()

# print(view1_img.shape, view1_mask.shape, view1_dynamic_mask.shape)
print(pred1.keys())
pred1_pts3d = pred1['pts3d'][0]
pred1_conf = pred1['conf'][0]
pred1_match_feature = pred1['match_feature'][0]
pred1_cross_atten_maps_k = pred1['cross_atten_maps_k'][0]
print(pred1_pts3d.shape, pred1_conf.shape, pred1_match_feature.shape, pred1_cross_atten_maps_k.shape)
# view2_img = view2['img']
# print(view1_img.shape, view2_img.shape)

points_3d = pred1_pts3d.reshape(-1, 3).cpu().numpy()
points_rgb = np.clip(view1_img.reshape(-1, 3), 0, 1)
# subsample by a factor of 100
points_3d = points_3d[::100]
points_rgb = points_rgb[::100]
# plot 3d points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c=points_rgb, s=1)
plt.show()

# global aligner