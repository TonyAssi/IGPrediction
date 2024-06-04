---
license: apache-2.0
base_model: google/vit-base-patch16-224
tags:
- Image Regression
datasets:
- "-"
metrics:
- accuracy
model-index:
- name: "-"
  results: []
---

# Title
## IG Prediction

This model was trained with [IGPrediction](https://github.com/TonyAssi/IGPrediction). It predicts how many likes an image will get.

```python
from IGPredict import predict_ig
predict_ig(repo_id='-',image_path='image.jpg')
```

---

## Dataset
Dataset:\
Value Column:\
Train Test Split:

---

## Training
Base Model: [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)\
Epochs:\
Learning Rate:

---

## Usage

### Download
```bash
git clone https://github.com/TonyAssi/IGPrediction.git
cd IGPrediction
```

### Installation
```bash
pip install -r requirements.txt
```

### Import 
```python
from IGPredict import ig_download, upload_dataset, train_ig_model, upload_ig_model, predict_ig
```

### Download Instagram Images
- **username** Instagram username
- **num_images** maximum number of images to download
```python
ig_download(username='instagarm_username', num_images=100)
```
Instagram images will be downloaded to *'./images'* folder, each one named like so *"index-likes.jpg"*. E.g. *"3-17.jpg"* is the third image and has 17 likes.

### Upload Dataset
- **dataset_name** name of dataset to be uploaded
- **token** go [here](https://huggingface.co/settings/tokens) to create a new 🤗 token
```python
upload_dataset(dataset_name='-', token='YOUR_HF_TOKEN')
```
Go to your  🤗 profile to find your uploaded dataset, it should look similar to [tonyassi/tony__assi-ig-ds](https://huggingface.co/datasets/tonyassi/tony__assi-ig-ds).


### Train Model
- **dataset_id** 🤗 dataset id
- **test_split** test split of the train/test split
- **num_train_epochs** training epochs
- **learning_rate** learning rate
```python
train_ig_model(dataset_id='-',
               test_split=-,
               num_train_epochs=-,
               learning_rate=-)

```
The trainer will save the checkpoints in the 'results' folder. The model.safetensors are the trained weights you'll use for inference (predicton).

### Upload Model
This function will upload your model to the 🤗 Hub.
- **model_id** the name of the model id
- **token** go [here](https://huggingface.co/settings/tokens) to create a new 🤗 token
- **checkpoint_dir** checkpoint folder that will be uploaded
```python
upload_ig_model(model_id='-',
             token='YOUR_HF_TOKEN',
             checkpoint_dir='./results/checkpoint-940')
```

### Inference (Prediction)
- **repo_id** 🤗 repo id of the model
- **image_path** path to image
```python
predict_ig(repo_id='-',
        image_path='image.jpg')
```
The first time this function is called it'll download the safetensor model. Subsequent function calls will run faster.