# References
Using 3 late days for phase 2 submission
## Models

### Object Detection
- **YOLO11x** — Ultralytics YOLO11, extra-large variant. Used for primary vehicle/pedestrian/sign detection.
  https://github.com/ultralytics/ultralytics
- **YOLO-World** (YOLOv8x-worldv2) — Open-vocabulary real-time object detection via vision-language prompts.
  https://github.com/AILab-CVC/YOLO-World
  arXiv: https://arxiv.org/abs/2401.17270

### Depth Estimation
- **Depth Anything V2** (Metric Outdoor Large) — Monocular metric depth estimation fine-tuned for outdoor scenes.
  HuggingFace: `depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf`
  https://github.com/DepthAnything/Depth-Anything-V2
  arXiv: https://arxiv.org/abs/2406.09414

### Lane Detection
- **YOLOPv2** — Panoptic driving perception: simultaneous lane detection, drivable area, and object detection.
  https://github.com/CAIC-AD/YOLOPv2
  arXiv: https://arxiv.org/abs/2208.11434
- **CLRerNet** — Confidence-guided lane detection with row-anchor representation.
  https://github.com/hirotomusiker/CLRerNet
  arXiv: https://arxiv.org/abs/2305.08366

### Pose Estimation
- **YOLO11x-Pose** — Ultralytics YOLO11 pose variant for 2D human keypoint detection.
  https://github.com/ultralytics/ultralytics
- **RTMW** (RTMPose Wholebody) — Real-time multi-person whole-body pose estimation (body + hands + face).
  https://github.com/open-mmlab/mmpose
  arXiv: https://arxiv.org/abs/2303.07399


### Instance Segmentation / Lane Detection
- **Mask R-CNN** — Region-based convolutional neural network for instance segmentation. Custom-trained on road-lane dataset for lane instance segmentation.
  https://github.com/facebookresearch/detectron2
  arXiv: https://arxiv.org/abs/1703.06870

---

## Datasets

- **BDD100K** — Large-scale diverse driving dataset (100K videos, North American roads). Used to train/validate YOLO detection models.
  https://bdd-data.berkeley.edu/
- **CULane** — Large-scale lane detection benchmark with 9 challenging scenarios.
  https://xingangpan.github.io/projects/CULane.html
- **KITTI** — Autonomous driving benchmark dataset for 3D detection, depth, and odometry.
  https://www.cvlibs.net/datasets/kitti/
- **nuScenes** — Full sensor suite autonomous driving dataset (cameras, lidar, radar).
  https://www.nuscenes.org/
- **Omni3D** — Large-scale benchmark for monocular 3D object detection (indoor + outdoor).
  https://github.com/facebookresearch/omni3d
- **FlyingThings3D** — Synthetic dataset for optical flow and scene flow training.
  https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
- **Cocktail14** — Multi-dataset training mixture used for RTMW wholebody pose.
  https://github.com/open-mmlab/mmpose
- **Road-Lane Instance Segmentation** — Roboflow Universe dataset used for custom Mask R-CNN lane training.
  https://universe.roboflow.com/

---

## Ego Motion / Visual Odometry

### DPVO (Dense Patch Visual Odometry)
Teed, Z., Lipson, L., & Deng, J. (2022). Deep Patch Visual Odometry. *Advances in Neural Information Processing Systems (NeurIPS 2023)*.
https://github.com/princeton-vl/DPVO
arXiv: https://arxiv.org/abs/2208.04726

---

## 3D Object Detection

### CubeRCNN / Omni3D
Brazil, G., Kumar, A., Straub, J., Liu, T., Johnson, J., & Gkioxari, G. (2023). Omni3D: A Large Benchmark and Model for 3D Object Detection in the Wild. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2023)*.
https://github.com/facebookresearch/omni3d
arXiv: https://arxiv.org/abs/2207.10660

---

## Vision-Language Models

### CLIP (OpenAI)
Radford, A., Kim, J.W., Hallacy, C., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *International Conference on Machine Learning (ICML 2021)*.
https://github.com/openai/CLIP
arXiv: https://arxiv.org/abs/2103.00020
*Used for: vehicle sub-type classification (sedan/SUV/hatchback/pickup) and speed limit sign validation (rejection of non-speed-limit signs).*

---

## OCR

### EasyOCR
Jaided AI. (2020). EasyOCR: Ready-to-use OCR with 80+ languages supported.
https://github.com/JaidedAI/EasyOCR
*Used for: reading speed limit values from detected sign crops.*

---

## Libraries and Frameworks

| Library | Version / Notes | Link |
|---------|-----------------|------|
| PyTorch | Core deep learning framework | https://pytorch.org/ |
| torchvision | Vision models and transforms | https://github.com/pytorch/vision |
| Ultralytics | YOLO11 / YOLO-World inference | https://github.com/ultralytics/ultralytics |
| Transformers (HuggingFace) | Depth Anything V2 inference | https://github.com/huggingface/transformers |
| MMPose | RTMW wholebody pose estimation | https://github.com/open-mmlab/mmpose |
| MMDetection | Mask R-CNN backbone | https://github.com/open-mmlab/mmdetection |
| OpenCV (`cv2`) | Video I/O, image processing, visualization | https://opencv.org/ |
| supervision | Detection tracking utilities, ByteTrack wrapper | https://github.com/roboflow/supervision |
| NumPy | Numerical arrays and geometry math | https://numpy.org/ |
| SciPy | Signal processing, interpolation | https://scipy.org/ |
| PyYAML | Config file parsing | https://pyyaml.org/ |
| msgpack / msgpack-numpy | Compact binary serialization of per-frame detections | https://github.com/msgpack/msgpack-python |
| Pillow | Image loading and color space utilities | https://python-pillow.org/ |
| tqdm | Progress bars for batch processing | https://github.com/tqdm/tqdm |

---

## Tools

- **Blender 5.1** — 3D rendering engine used for synthetic scene reconstruction and visualization.
  https://www.blender.org/
- **ffmpeg** — Video encoding and format conversion.
  https://ffmpeg.org/
- **uv** — Fast Python package manager and virtual environment tool.
  https://github.com/astral-sh/uv
