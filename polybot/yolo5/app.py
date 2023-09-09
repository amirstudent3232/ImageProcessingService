from pathlib import Path
from flask import Flask, request
from detect import run
import uuid
import yaml
from loguru import logger
import os
import boto3
import time
import pymongo

images_bucket = os.environ['BUCKET_NAME']

with open("data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']

# Initialize the S3 client
s3 = boto3.client('s3')

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # Generates a UUID for this current prediction HTTP request. This id can be used as a reference in logs to
    # identify and track individual prediction requests.
    prediction_id = str(uuid.uuid4())

    logger.info(f'prediction: {prediction_id}. start processing')

    # Receives a URL parameter representing the image to download from S3
    img_name = request.args.get('imgName')

    # TODO download img_name from S3, store the local image path in original_img_path
    #  The bucket name should be provided as an env var BUCKET_NAME.
    filename = img_name.split('/')[-1]  # Get the filename alone as srt
    local_dir = 'photos/'  # str of dir to save to
    os.makedirs(local_dir, exist_ok=True)  # make sure the dir exists
    original_img_path = local_dir + filename  # assign the full path of the file to download
    s3.download_file(images_bucket, img_name, original_img_path)  # download the file

    logger.info(f'prediction id: {prediction_id}, path: \"{original_img_path}\" Download img completed')

    # Predicts the objects in the image
    run(
        weights='yolov5s.pt',
        data='data/coco128.yaml',
        source=original_img_path,
        project='static/data',
        name=prediction_id,
        save_txt=True
    )

    logger.info(f'prediction: {prediction_id}, path: {original_img_path}. done')

    # This is the path for the predicted image with labels The predicted image typically includes bounding boxes
    # drawn around the detected objects, along with class labels and possibly confidence scores.
    predicted_img_path = Path(f'static/data/{prediction_id}/{filename}')  # get the result path
    # TODO Uploads the predicted image (predicted_img_path) to S3 (be careful not to override the original image).
    predicted_img_name = f'predicted_{filename}'  # assign the new name
    os.rename(f'/usr/src/app/static/data/{prediction_id}/{filename}',
              f'/usr/src/app/static/data/{prediction_id}/{predicted_img_name}')  # rename the file before upload
    s3_path_to_upload_to = '/'.join(img_name.split('/')[:-1]) + f'/{predicted_img_name}'  # assign the path on s3 as str
    file_to_upload = f'/usr/src/app/static/data/{prediction_id}/{predicted_img_name}'  # assign the path locally as str
    s3.upload_file(file_to_upload, images_bucket, s3_path_to_upload_to)  # upload the file to same path with new name s3
    os.rename(f'/usr/src/app/static/data/{prediction_id}/{predicted_img_name}',
              f'/usr/src/app/static/data/{prediction_id}/{filename}')  # rename the file back after upload

    # Parse prediction labels and create a summary
    pred_summary_path = Path(f'static/data/{prediction_id}/labels/{original_img_path.split(".")[0]}.txt')
    if pred_summary_path.exists():
        with open(pred_summary_path) as f:
            labels = f.read().splitlines()
            labels = [line.split(' ') for line in labels]
            labels = [{
                'class': names[int(l[0])],
                'cx': float(l[1]),
                'cy': float(l[2]),
                'width': float(l[3]),
                'height': float(l[4]),
            } for l in labels]

        logger.info(f'prediction: {prediction_id}/{original_img_path}. prediction summary:\n\n{labels}')

        prediction_summary = {
            'prediction_id': prediction_id,
            'original_img_path': original_img_path,
            'predicted_img_path': predicted_img_path,
            'labels': labels,
            'time': time.time()
        }

        cluster = os.environ['CLUSTERSTRING']
        mongo_client = pymongo.MongoClient(cluster)

        db = mongo_client['shermanDB']
        collection = db['shermanCollection']

        result = collection.insert_one(prediction_summary)

        return prediction_summary
    else:
        return f'prediction: {prediction_id}/{original_img_path}. prediction result not found', 404


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081)




"""import time
from datetime import datetime
from pathlib import Path
from flask import Flask, request
from detect import run
import uuid
import yaml
from loguru import logger
import os
import boto3
from pymongo import MongoClient

images_bucket = os.environ['BUCKET_NAME']

with open("data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Generates a UUID for this current prediction HTTP request. This id can be used as a reference in logs to identify and track individual prediction requests.
    prediction_id = str(uuid.uuid4())

    logger.info(f'prediction: {prediction_id}. start processing')

    # Receives a URL parameter representing the image to download from S3
    img_name = request.args.get('imgName')

    # TODO download img_name from S3, store the local image path in original_img_path
    #  The bucket name should be provided as an env var BUCKET_NAME.

    # start
    def download_from_s3(bucket_name, img_name):
        s3 = boto3.client('s3')
        local_img_path = f"static/data/{img_name}"  # Change according to your directory structure
        s3.download_file(bucket_name, img_name, local_img_path)
        return local_img_path
    # end
    original_img_path = ...

    logger.info(f'prediction: {prediction_id}/{original_img_path}. Download img completed')

    # Predicts the objects in the image
    run(
        weights='yolov5s.pt',
        data='data/coco128.yaml',
        source=original_img_path,
        project='static/data',
        name=prediction_id,
        save_txt=True
    )

    logger.info(f'prediction: {prediction_id}/{original_img_path}. done')

    # This is the path for the predicted image with labels
    # The predicted image typically includes bounding boxes drawn around the detected objects, along with class labels and possibly confidence scores.
    predicted_img_path = Path(f'static/data/{prediction_id}/{original_img_path}')

    # TODO Uploads the predicted image (predicted_img_path) to S3 (be careful not to override the original image).

    # start
    def upload_to_s3(bucket_name, local_path):
        s3 = boto3.client('s3')
        s3_path = "predicted_" + Path(local_path).name
        with open(local_path, 'rb') as f:
            s3.upload_fileobj(f, bucket_name, s3_path)

    # end

    # Parse prediction labels and create a summary
    pred_summary_path = Path(f'static/data/{prediction_id}/labels/{original_img_path.split(".")[0]}.txt')
    if pred_summary_path.exists():
        with open(pred_summary_path) as f:
            labels = f.read().splitlines()
            labels = [line.split(' ') for line in labels]
            labels = [{
                'class': names[int(l[0])],
                'cx': float(l[1]),
                'cy': float(l[2]),
                'width': float(l[3]),
                'height': float(l[4]),
            } for l in labels]

        logger.info(f'prediction: {prediction_id}/{original_img_path}. prediction summary:\n\n{labels}')

        prediction_summary = {
            'prediction_id': prediction_id,
            'original_img_path': original_img_path,
            'predicted_img_path': predicted_img_path,
            'labels': labels,
            'time': time.time()
        }

        # TODO store the prediction_summary in MongoDB

        # start

        def store_in_mongo(summary):
            client = MongoClient()  # Update with your MongoDB URI if it's not the default one
            db = client.predictions_db  # Update with your database name
            collection = db.predictions  # Update with your collection name
            collection.insert_one(summary)

        # end

        return prediction_summary
    else:
        return f'prediction: {prediction_id}/{original_img_path}. prediction result not found', 404


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081)

# original_img_path = download_from_s3(images_bucket, img_name)

# upload_to_s3(images_bucket, str(predicted_img_path))

# store_in_mongo(prediction_summary)"""