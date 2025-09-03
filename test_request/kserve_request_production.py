import json
import logging
import os

import requests

logging.basicConfig(level=logging.INFO)


def get_access_token(username, password, token_url):
    data = {
        "username": username,
        "password": password,
    }

    response = requests.post(token_url, data=data)
    if response.status_code != 200:
        raise Exception(
            f'Error {response.status_code}. {response.json()["detail"]}'
        )
    return response.json()['access_token']


def make_inference_request(endpoint_url, access_token, data):
    logging.info(f'Kserve endpoint: {endpoint_url}')
    resp = requests.post(
        endpoint_url,
        data=json.dumps(data),
        headers={'Authorization': f'Bearer {access_token}'},
    )
    try:
        resp.raise_for_status()  # Raise an exception for 4xx and 5xx status codes
        logging.info("URL check successful.")
        logging.info(resp.text)
    except Exception as e:
        logging.error(f"Error checking URL: {e}")
        exit(1)  # Exit the script with a non-zero exit code
    return resp


def main():
    EMERALDS_TOKEN_URL = "https://emeralds-token.apps.emeralds.ari-aidata.eu"
    KSERVE_MODEL_ENDPOINT = "https://zone-density-model.models.kubeflow.emeralds.ari-aidata.eu/v1/models/zone-density-model:predict"
    MLOPS_PLATFORM_USERNAME_PRODUCTION = os.environ[
        'MLOPS_PLATFORM_USERNAME_PRODUCTION'
    ]
    MLOPS_PLATFORM_PASSWORD_PRODUCTION = os.environ[
        'MLOPS_PLATFORM_PASSWORD_PRODUCTION'
    ]

    access_token = get_access_token(
        MLOPS_PLATFORM_USERNAME_PRODUCTION,
        MLOPS_PLATFORM_PASSWORD_PRODUCTION,
        EMERALDS_TOKEN_URL,
    )

    data = {
        "instances": [
            [3, "13:30", "Rumbula"],  # day=3 (Wed), 13:30, zone string
            [6, "21:15", "Rumbula"],
        ]
    }

    make_inference_request(KSERVE_MODEL_ENDPOINT, access_token, data)


if __name__ == "__main__":
    main()
