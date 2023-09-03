import time
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
    def upload_predicted_image_to_s3(predicted_img_path, original_img_name):
        """
        Uploads the predicted image to S3 ensuring not to overwrite the original.

        :param predicted_img_path: Local path to the predicted image.
        :param original_img_name: Original image name as it was in S3.
        :return: The S3 URL of the uploaded image.
        """

        # Fetch the bucket name from the environment variable
        bucket_name = os.environ['BUCKET_NAME']

        # Create a unique key for the predicted image to ensure it doesn't override the original.
        # This example uses a timestamp for uniqueness. You can also use other methods like appending a UUID.
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        s3_key = f"predictions/{original_img_name}_{timestamp}"

        # Create an S3 client
        s3 = boto3.client('s3')

        # Upload the predicted image to S3
        s3.upload_file(predicted_img_path, bucket_name, s3_key)

        # Return the S3 URL of the uploaded image (this step is optional, depending on your needs)
        s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
        return s3_url

    # Example usage:
    s3_url = upload_predicted_image_to_s3('/local/path/to/predicted_image.jpg', 'original_image_name.jpg')
    print(f"Predicted image uploaded to: {s3_url}")

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
    def upload_predicted_image_to_s3(predicted_img_path, s3_bucket, original_img_name):
        """
        Uploads the predicted image to S3 without overwriting the original image.

        :param predicted_img_path: Local path of the predicted image.
        :param s3_bucket: The name of the S3 bucket to upload to.
        :param original_img_name: The name of the original image for reference.
        :return: S3 object key (path) for the uploaded image.
        """

        # Generate a unique identifier to append to the original filename.
        unique_id = uuid.uuid4().hex
        s3_key = f"predictions/{original_img_name}_predicted_{unique_id}.jpg"

        # Create an S3 client
        s3 = boto3.client('s3')

        # Upload the predicted image using the unique S3 key
        s3.upload_file(predicted_img_path, s3_bucket, s3_key)

        return s3_key

    # Usage example:
    s3_bucket = os.environ['BUCKET_NAME']
    predicted_img_local_path = "/path/to/your/predicted/image.jpg"
    original_img_name = "original_image_name"

    s3_object_key = upload_predicted_image_to_s3(predicted_img_local_path, s3_bucket, original_img_name)
    print(f"Predicted image uploaded to S3 with key: {s3_object_key}")

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
        def store_prediction_in_mongo(prediction_summary):
            # Connect to MongoDB running in Docker (default IP is 172.17.0.2 for the first container, but you might need to adjust depending on your setup)
            client = MongoClient("mongodb://172.17.0.2:27017/")

            # Select your database (replace "your_database" with your desired DB name)
            db = client["your_database"]

            # Select the collection (replace "your_collection" with your desired collection name)
            predictions = db["your_collection"]

            # Insert the prediction_summary
            predictions.insert_one(prediction_summary)

        # Example usage
        prediction_summary = {
            'prediction_id': '123456',
            'original_img_path': '/path/to/original/img.jpg',
            'predicted_img_path': '/path/to/predicted/img.jpg',
            'labels': [{'class': 'dog', 'confidence': 0.95}],
            'time': 1234567890.0
        }

        store_prediction_in_mongo(prediction_summary)

        # end

        return prediction_summary
    else:
        return f'prediction: {prediction_id}/{original_img_path}. prediction result not found', 404


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081)