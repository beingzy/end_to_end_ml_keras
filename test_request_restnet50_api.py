import os
import sys
import requests
import argparse


def generate_payload(image_file):
    """return dictionary as payload to send to server
       * image_file: string, path for image
    """
    if not os.path.exists(image_file):
        raise IOError("{} does not exist!".format(image_file))

    image = open(image_file, mode='rb').read()
    return {'image': image}


def request_predict(image_file):
    payload = generate_payload(image_file)
    r = requests.post(REST_API_PREDICT_URL, files=payload).json()
    if r["success"]:
        for (ii, result) in enumerate(r["predictions"]):
            msg = "{}. {}: {:.4f}".format(ii + 1, result["label"],
                result["probability"])
            print(msg)
    else:
        print("Request failed.")


if __name__ == "__main__":
    REST_API_PREDICT_URL = "http://localhost:5000/predict"
    parser = argparse.ArgumentParser(__name__)
    parser.add_argument('--image', dest="image_file")
    args = parser.parse_args()

    image_file = args.image_file or os.path.join('data', 'test_dog.jpg')

    request_predict(image_file)
