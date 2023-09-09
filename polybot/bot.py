import telebot
from loguru import logger
import os
import time
from telebot.types import InputFile
from polybot.img_proc import Img
import requests
import boto3
import json

class Bot:

    def __init__(self, token, telegram_chat_url):
        # create new instance of the TeleBot class.
        # all communication with Telegram servers are done using self.telegram_bot_client
        self.telegram_bot_client = telebot.TeleBot(token)

        # remove any existing webhooks configured in Telegram servers
        self.telegram_bot_client.remove_webhook()
        time.sleep(0.5)

        # set the webhook URL
        self.telegram_bot_client.set_webhook(url=f'{telegram_chat_url}/{token}/', timeout=60)

        logger.info(f'Telegram Bot information\n\n{self.telegram_bot_client.get_me()}')

    def send_text(self, chat_id, text):
        self.telegram_bot_client.send_message(chat_id, text)

    def send_text_with_quote(self, chat_id, text, quoted_msg_id):
        self.telegram_bot_client.send_message(chat_id, text, reply_to_message_id=quoted_msg_id)

    def is_current_msg_photo(self, msg):
        return 'photo' in msg

    def download_user_photo(self, msg):
        """
        Downloads the photos that sent to the Bot to `photos` directory (should be existed)
        :param quality: integer representing the file quality. Allowed values are [0, 1, 2]
        :return:
        """
        if not self.is_current_msg_photo(msg):
            raise RuntimeError(f'Message content of type \'photo\' expected')

        file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
        data = self.telegram_bot_client.download_file(file_info.file_path)
        folder_name = file_info.file_path.split('/')[0]

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        with open(file_info.file_path, 'wb') as photo:
            photo.write(data)

        return file_info.file_path

    def send_photo(self, chat_id, img_path):
        if not os.path.exists(img_path):
            raise RuntimeError("Image path doesn't exist")

        self.telegram_bot_client.send_photo(
            chat_id,
            InputFile(img_path)
        )

    def handle_message(self, msg):
        """Bot Main message handler"""
        logger.info(f'Incoming message: {msg}')
        self.send_text(msg['chat']['id'], f'Your original message: {msg["text"]}')


class QuoteBot(Bot):
    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')

        if msg["text"] != 'Please don\'t quote me':
            self.send_text_with_quote(msg['chat']['id'], msg["text"], quoted_msg_id=msg["message_id"])


class ImageProcessingBot(Bot):
    def __init__(self, token, telegram_chat_url):
        super().__init__(token, telegram_chat_url)

    def handle_message(self, message):
        if not self.is_current_msg_photo(message):
            raise RuntimeError(f'Message content of type \'photo\' expected')

        filter_name = message.get('caption', '').lower()
        if 'salt' in filter_name and 'pepper' in filter_name:
            filter_name = 'salt_n_pepper'

        if filter_name in ['blur', 'contour', 'rotate', 'segment', 'salt_n_pepper']:
            self.process_and_send_image(message, filter_name)
        else:
            self.send_text(message['chat']['id'], "Invalid filter name. Please provide a valid filter name.")

    def process_and_send_image(self, message, filter_name):
        user_id = message['from']['id']
        image_path = self.download_user_photo(message)
        try:
            img = Img(image_path)

            if filter_name == 'blur':
                img.blur()
            elif filter_name == 'contour':
                img.contour()
            elif filter_name == 'rotate':
                img.rotate()
            elif filter_name == 'segment':
                img.segment()
            elif filter_name == 'salt_n_pepper':
                img.salt_n_pepper()

            # Save and send the processed image
            processed_image_path = img.save_img()
            self.send_photo(user_id, processed_image_path)
        except RuntimeError as e:
            self.send_text(user_id, str(e))

class ObjectDetectionBot(Bot):
    def __init__(self, token, telegram_chat_url):
        super().__init__(token, telegram_chat_url)
        self.s3_client = boto3.client('s3')
        self.default_response = "Sorry, I didn't understand that. Type /help for available commands."

    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')

        if self.is_current_msg_photo(msg):
            photo_download = self.download_user_photo(msg)
            s3_bucket = "S3name"
            img_name = f'tg-photos/{photo_download}'
            self.s3_client.upload_file(photo_download, s3_bucket, img_name)
            yolo_summary = self.yolo5_request(img_name)  # Get YOLOv5 summary
            self.send_summary_to_user(msg['chat']['id'], yolo_summary)  # Send the summary to the user

    def send_summary_to_user(self, chat_id, summary):
        # Format the YOLOv5 summary as a string
        summary_str = json.dumps(summary, indent=4)

        # Send the summary to the user
        self.send_text(chat_id, summary_str)

    def yolo5_request(self, s3_photo_path):
        yolo5_api = "http://localhost:8081/predict"
        response = requests.post(f"{yolo5_api}?imgName={s3_photo_path}")

        if response.status_code == 200:
            try:
                return response.json()  # Attempt to parse the JSON response
            except json.JSONDecodeError as e:
                logger.error(f'Failed to decode JSON response: {e}')
                return {"error": "Invalid JSON response from YOLOv5 API"}

        else:
            logger.error(f'Error response from YOLOv5 API: {response.status_code} - {response.text}')
            return {"error": f"Error response from YOLOv5 API: {response.status_code}"}

