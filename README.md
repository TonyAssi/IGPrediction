# IG Prediction

by [Tony Assi](https://www.tonyassi.com/)

Predict Instagram likes of an image using image regression. The framework takes care of the scraping, model training, and inference.

Built with PyTorch, 🤗 Transformers, and [Image Regression Trainer](https://github.com/TonyAssi/ImageRegression).

## Download
```bash
git clone https://github.com/TonyAssi/IGPrediction.git
cd IGPrediction
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Import 
```python
from IGPredict import ig_download, upload_dataset, train_ig_model, upload_ig_model, predict_ig
```

### Download Instagram Images
- **username** Instagram username
- **num_images** maximum number of images to download
```python
ig_download(username='tony__assi', num_images=100)
```
Instagram images will be downloaded to *'./images'* folder, each one named like so *"index-likes.jpg"*. E.g. *"3-17.jpg"* is the third image and has 17 likes.

### Upload Dataset
- **dataset_name** name of dataset to be uploaded
- **token** go [here](https://huggingface.co/settings/tokens) to create a new 🤗 token
```python
upload_dataset(dataset_name='tony__assi-ig-ds', token='YOUR_HF_TOKEN')
```
Go to your  🤗 profile to find your uploaded dataset, it should look similar to [tonyassi/tony__assi-ig-ds](https://huggingface.co/datasets/tonyassi/tony__assi-ig-ds).

### Train Model
- **dataset_id** 🤗 dataset id
```python
train_ig_model('tonyassi/tony__assi-ig-ds')
```
You can also set the training parameters:
- **dataset_id** 🤗 dataset id
- **test_split** test split of the train/test split
- **num_train_epochs** training epochs
- **learning_rate** learning rate
```python
train_ig_model(dataset_id='tonyassi/tony__assi-ig-ds',
               test_split=0.2, # default
               num_train_epochs=10, # default
               learning_rate=1e-4) # default
```
The trainer will save the checkpoints in the *"results"* folder. The model.safetensors are the trained weights you'll use for inference (predicton).

### Upload Model
This function will upload your model to the 🤗 Hub, which will be useful for inference.
- **model_id** the name of the model id
- **token** go [here](https://huggingface.co/settings/tokens) to create a new 🤗 token
- **checkpoint_dir** checkpoint folder that will be uploaded (located inside *"results"* folder)
```python
upload_ig_model(model_id='tony__assi-ig-prediction',
                token='YOUR_HF_TOKEN',
                checkpoint_dir='./results/checkpoint-100')
```
Go to your 🤗 profile to find your uploaded model, it should look similar to [tonyassi/tony__assi-ig-prediction](https://huggingface.co/tonyassi/tony__assi-ig-prediction).

### Inference (Prediction)
- **repo_id** 🤗 repo id of the model
- **image_path** path to image
```python
predicted_likes = predict_ig(repo_id='tonyassi/tony__assi-ig-prediction',
                             image_path='images.jpg')
print(predicted_likes)
```
The first time this function is called it'll download the safetensor model. Subsequent function calls will run faster.


