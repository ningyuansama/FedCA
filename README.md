# YOLO-based GPR Defect Detection

This repository provides a YOLO-based object detection framework for Ground Penetrating Radar (GPR) defect detection. The project includes model definition, training scripts, inference scripts, mAP evaluation scripts, and federated learning client/server code.

The pretrained weights and test dataset are not included in this repository due to file size limitations. They are provided separately through Baidu Netdisk.

---

## Project Structure

```text
.
├── __pycache__/
├── model_data/          # Model-related files, such as class names
├── nets/                # YOLO network architecture
├── utils/               # Utility functions for training, data loading, and evaluation
├── utils_coco/          # COCO-related utility functions
├── client.py            # Federated learning client script
├── server.py            # Federated learning server script
├── train.py             # Local training script
├── yolo.py              # YOLO inference configuration file
├── get_map.py           # mAP evaluation script
├── predict.py           # prediction script
├── voc_annotation.py    # VOC annotation list generation script
├── summary.py           # Model structure summary script
├── requirements.txt     # Python dependencies
└── d2l.yml              # Conda environment configuration file
```

---

## Test Data and Pretrained Weights

The test dataset and pretrained model weights are provided through Baidu Netdisk.

Download link:

```text
https://pan.baidu.com/s/1rRsbEy75OxPGlsThibcrKg?pwd=t74w
```

Extraction code:

```text
t74w
```

After downloading, place the test dataset and weight files in the project directory. A recommended structure is shown below:

```text
.
├── test_data/
│   └── VOC2007/
│       ├── Annotations/
│       ├── JPEGImages/
│       └── ImageSets/
│           └── Main/
│               └── test.txt
├── model_data/
│   ├── gpr_classes.txt
│   └── weight.pth
├── yolo.py
├── get_map.py
└── ...
```

The test dataset should follow the Pascal VOC format:

```text
test_data/VOC2007/Annotations              # XML annotation files
test_data/VOC2007/JPEGImages               # Test images
test_data/VOC2007/ImageSets/Main/test.txt  # Test image list
```

---

## Environment Setup

You can install the required dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

Alternatively, you can create a Conda environment using `d2l.yml`:

```bash
conda env create -f d2l.yml
conda activate d2l
```

If your Conda environment name is different, please replace `d2l` with your own environment name.

---

## Configure the Model Weight Path

Before running inference or evaluation, you need to modify the model weight path in `yolo.py`.

Open `yolo.py` and find the model configuration section. It is usually similar to:

```python
_defaults = {
    "model_path": "model_data/weight.pth",
    "classes_path": "model_data/gpr_classes.txt",
}
```

Modify `model_path` to the path of the downloaded weight file. For example:

```python
"model_path": "model_data/best_weights.pth"
```

If the weight file is stored in another folder, modify the path accordingly:

```python
"model_path": "weights/best.pth"
```

Please also make sure that the class file path is correct:

```python
"classes_path": "model_data/gpr_classes.txt"
```

The class names in `voc_classes.txt` must be consistent with the training setting. For example:

```text
DT_GAP
DT_LACUNAS
DT_SUBSIDENCE
DT_CRACK
```

---

## Configure the Test Dataset Path

Before running mAP evaluation, you need to modify the test dataset path in `get_map.py`.

Open `get_map.py` and find the dataset path setting. It is usually similar to:

```python
VOCdevkit_path = 'VOCdevkit'
```

Modify it to the root directory of the downloaded test dataset. For example:

```python
VOCdevkit_path = 'test_data'
```

The directory should then look like this:

```text
test_data/
└── VOC2007/
    ├── Annotations/
    ├── JPEGImages/
    └── ImageSets/
        └── Main/
            └── test.txt
```

If your dataset is stored in another path, such as:

```text
data/test_data/VOC2007/
```

then modify the path as follows:

```python
VOCdevkit_path = 'data/test_data'
```

---

## Run mAP Evaluation

After configuring the weight path in `yolo.py` and the dataset path in `get_map.py`, run:

```bash
python get_map.py
```

The script will automatically load the test images and annotation files, generate detection results, and calculate AP and mAP.

After evaluation, the following output directory will usually be generated:

```text
map_out/
├── detection-results/   # Predicted detection results
├── ground-truth/        # Ground-truth annotation files
├── images-optional/     # Optional visualization images
└── results/             # Evaluation results
```

The final mAP results can be found in:

```text
map_out/results/results.txt
```

---

## Single Image Prediction

To test detection on a single image, run:

```bash
python predict.py
```

Then input the image path according to the terminal prompt, for example:

```text
test_data/VOC2007/JPEGImages/example.jpg
```

The program will perform object detection on the input image and display or save the prediction result.

---

## Local Training

To train the model locally, run:

```bash
python train.py
```

Before training, please check the following settings:

1. The dataset path is correctly configured.
2. The class file `model_data/voc_classes.txt` is correct.
3. The pretrained weight path is correct if pretrained weights are used.
4. The batch size, learning rate, input size, and other training parameters in `train.py` are suitable for your device.

---

## Federated Learning Training

This repository also includes federated learning scripts:

```text
server.py   # Federated learning server
client.py   # Federated learning client
```

A typical federated learning process is as follows.

First, start the server:

```bash
python server.py
```

Then, start the clients:

```bash
python client.py
```

For multi-client training, you may need to modify the following settings in `client.py`:

1. Client ID
2. Local dataset path
3. GPU device ID
4. Local training parameters
5. Server address and port

Please make sure the server is running before starting the clients.

---

## Quick Start for Testing

A complete testing workflow is shown below:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the test dataset and pretrained weights from Baidu Netdisk

# 3. Modify the model weight path in yolo.py
# Example:
# "model_path": "model_data/best_epoch_weights.pth"

# 4. Modify the test dataset path in get_map.py
# Example:
# VOCdevkit_path = 'test_data'

# 5. Run mAP evaluation
python get_map.py
```

After the evaluation is completed, check the result file:

```text
map_out/results/results.txt
```

---

## Notes

1. The pretrained weights and test dataset must be downloaded before running evaluation.
2. The `model_path` in `yolo.py` must point to the correct weight file.
3. The `VOCdevkit_path` in `get_map.py` must point to the correct test dataset root directory.
4. The test dataset must follow the Pascal VOC format.
5. The class order in `voc_classes.txt` must be the same as that used during training.
6. If images or XML files cannot be found, please check the structure of `test_data/VOC2007`.
7. If the model output is incorrect or the mAP is abnormal, please check whether the weight file and class file match.

---

## Citation

If you use this code, dataset, or pretrained weights in your research, please cite the corresponding paper or project once it becomes available.

---

## Contact

For questions about the code, dataset, or pretrained weights, please contact the project maintainer.
