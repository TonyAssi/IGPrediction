import instaloader
import os
import shutil
import csv
from huggingface_hub import create_repo
from datasets import load_dataset
import ImageRegression


def ig_download(username, num_images):
    # Initialize Instaloader
    L = instaloader.Instaloader()

    # Load the profile
    profile = instaloader.Profile.from_username(L.context, username)

    # Create directories to store the downloaded and final copied posts
    download_dir = 'downloaded_posts'
    final_dir = 'images'
    if os.path.exists(download_dir):
        shutil.rmtree(download_dir)
    os.makedirs(download_dir, exist_ok=True)

    os.makedirs(final_dir, exist_ok=True)

    # Loop through each post in the profile
    num_downloaded_images = 1
    for index, post in enumerate(profile.get_posts()):

        # Skip if the post is a video
        if post.is_video:
            print(f"Skipping video post {index} with shortcode {post.shortcode}")
            continue

        # Determine the filename
        likes = post.likes
        if(likes == -1): likes=1 # If the instagram account hides likes then likes will be -1
        extension = '.mp4' if post.is_video else '.jpg'
        custom_filename = f"{num_downloaded_images}-{likes}{extension}"
        custom_path = os.path.join(download_dir, custom_filename)
        
        # Download the post using Instaloader with a custom target
        print(f"Downloading post {index} with shortcode {post.shortcode}")
        L.download_post(post, target=download_dir)
        
        # Debug: List files in output directory after download
        downloaded_files = os.listdir(download_dir)
        print(f"Files in {download_dir} after download: {downloaded_files}")
        
        # Assume the most recent file is the correct one
        newest_file = max([os.path.join(download_dir, f) for f in downloaded_files if f.endswith(extension)], key=os.path.getctime)
        
        print(f"Copying file {newest_file} to {custom_path}")
        shutil.copyfile(newest_file, custom_path)

        print(f"Downloaded and saved: {custom_path}\n")
        
        # Copy the file to the final directory
        final_path = os.path.join(final_dir, custom_filename)
        shutil.copyfile(custom_path, final_path)
        print(f"Copied to final directory: {final_path}\n")

        num_downloaded_images += 1

        # Stop downloading images if max images is reached
        if(num_downloaded_images > num_images):
            break

    shutil.rmtree(download_dir)


def create_metadata_csv():
    folder_path = './images'

    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return
    
    # Get a list of files in the folder
    files = [filename for filename in os.listdir(folder_path) if filename.endswith('.jpg')]
    
    # Sort the files based on the numerical part before the hyphen in the filename
    files.sort(key=lambda x: int(x.split('-')[0]))
    
    # Create or open the metadata.csv file for writing
    with open(os.path.join(folder_path, 'metadata.csv'), 'w', newline='') as csvfile:
        fieldnames = ['file_name', 'likes']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header row
        writer.writeheader()
        
        # Iterate through the sorted files
        for filename in files:
            # Extract the likes from the filename
            likes = filename.split('-')[1].split('.')[0]
            
            # Write the filename and likes to the CSV file
            writer.writerow({'file_name': filename, 'likes': likes})

def upload_dataset(dataset_name, token):
    create_metadata_csv()
    
    # Create a dataset repo
    dataset_id = create_repo(dataset_name, token=token, repo_type="dataset").repo_id

    # Load images
    dataset = load_dataset('imagefolder', data_dir='./images',  split='train')

    # Push dataset to Huggingface
    dataset.push_to_hub(dataset_id, token=token)

    print('Dataset was uploaded to '+ dataset_id)

def train_ig_model(dataset_id, test_split=0.2, num_train_epochs=10, learning_rate=1e-4):
    ImageRegression.train_model(dataset_id=dataset_id,
                                test_split=test_split,
                                num_train_epochs=num_train_epochs,
                                learning_rate=learning_rate,
                                value_column_name='likes',
                                output_dir='./results')

def upload_ig_model(model_id, token, checkpoint_dir):
    ImageRegression.upload_model(model_id, token, checkpoint_dir)

def predict_ig(repo_id, image_path):
    return ImageRegression.predict(repo_id=repo_id, image_path=image_path)











