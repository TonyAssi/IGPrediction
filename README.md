# IG Prediction

by [Tony Assi](https://www.tonyassi.com/)

Predict Instagram likes of an image using image regression. The framework takes care of the scraping, model training, and inference.

Image Regression model training fine-tunes [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) with a custom PyTorch model that takes an image as input and outputs the predicted number of likes. Built with PyTorch, 🤗 Transformers, and [Image Regression Trainer](https://github.com/TonyAssi/ImageRegression).

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
Download Instagram images and likes to *'./images'* folder using [Instaloader](https://github.com/instaloader/instaloader).
- **username** Instagram username
- **num_images** maximum number of images to download
```python
ig_download(username='tony__assi', num_images=100)
```

### Train Model
- **dataset_id** 🤗 dataset id (see [Dataset](https://github.com/TonyAssi/ImageRegression?tab=readme-ov-file#dataset))
- **value_column_name** column name of prediction values in dataset
- **test_split** test split of the train/test split
- **output_dir** the directory where the checkpoints will be saved
- **num_train_epochs** training epochs
- **learning_rate** learning rate
```python
train_model(dataset_id='tonyassi/clothing-sales-ds',
            value_column_name='sales',
            test_split=0.2,
            output_dir='./results',
            num_train_epochs=10,
            learning_rate=1e-4)

```
The trainer will save the checkpoints in the output_dir location. The model.safetensors are the trained weights you'll use for inference (predicton).

### Upload Model
This function will upload your model to the 🤗 Hub, which will be useful for inference.
- **model_id** the name of the model id
- **token** go [here](https://huggingface.co/settings/tokens) to create a new 🤗 token
- **checkpoint_dir** checkpoint folder that will be uploaded
```python
upload_model(model_id='sales-prediction',
             token='YOUR_HF_TOKEN',
             checkpoint_dir='./results/checkpoint-940')
```
Go to your 🤗 profile to find your uploaded model, it should look similar to [tonyassi/sales-prediction](https://huggingface.co/tonyassi/sales-prediction).

### Inference (Prediction)
- **repo_id** 🤗 repo id of the model
- **image_path** path to image
```python
predict(repo_id='tonyassi/sales-prediction',
        image_path='image.jpg')
```
The first time this function is called it'll download the safetensor model. Subsequent function calls will run faster.

## Dataset

The model trainer takes a 🤗 dataset id as input so your dataset must be uploaded to 🤗. It should have a column of images and a column of values (floats or ints). Check out [Create an image dataset](https://huggingface.co/docs/datasets/en/image_dataset) if you need help creating a 🤗 dataset. Your dataset should look like [tonyassi/clothing-sales-ds](https://huggingface.co/datasets/tonyassi/clothing-sales-ds) (the values column can be named whatever you'd like).

<img width="868" alt="Screenshot 2024-05-18 at 12 11 32 PM" src="https://github.com/TonyAssi/ImageRegression/assets/42156881/06ed6954-de6f-45ab-84a3-57781d39722b">
