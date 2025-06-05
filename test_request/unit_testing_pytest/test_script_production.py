import pytest
from unittest.mock import MagicMock
import requests
import os
import logging
logging.basicConfig(level=logging.INFO)

from test_request.kserve_request_production import get_access_token, make_inference_request

def test_get_access_token_success(mocker):
    # Mock the requests.post method
    mock_post = mocker.patch('test_request.kserve_request_production.requests.post')

    # Configure the mock to return a response with a specific status code and JSON data
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'access_token': 'fake-token'}
    mock_post.return_value = mock_response

    # Call the function
    token = get_access_token('user', 'pass', 'https://fake-url.com')

    # Assert that the token is as expected
    assert token == 'fake-token'
    logging.info("Test get_access_token_success was successful")

def test_get_access_token_failure(mocker):
    # Mock the requests.post method
    mock_post = mocker.patch('test_request.kserve_request_production.requests.post')

    # Configure the mock to return a response with a 400 status code
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.json.return_value = {'detail': 'Invalid credentials'}
    mock_post.return_value = mock_response

    # Call the function and verify that it raises an exception
    with pytest.raises(Exception) as exc_info:
        get_access_token('user', 'pass', 'https://fake-url.com')
    assert 'Error 400. Invalid credentials' in str(exc_info.value)
    logging.info("Test test_get_access_token_failure was successful")

def test_make_inference_request_success(mocker):
    # Mock the requests.post method
    mock_post = mocker.patch('test_request.kserve_request_production.requests.post')

    # Configure the mock to return a successful response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = 'Success'
    mock_response.raise_for_status.return_value = None  # No exception
    mock_post.return_value = mock_response

    # Call the function
    response = make_inference_request(
        'https://fake-endpoint.com',
        'fake-token',
        {'instances': [[1, 2, 3, 4]]}
    )

    # Assert that the response text is as expected
    assert response.text == 'Success'
    logging.info("Test test_make_inference_request_success was successful")

def test_make_inference_request_failure(mocker):
    # Mock the requests.post method
    mock_post = mocker.patch('test_request.kserve_request_production.requests.post')

    # Configure the mock to simulate a failed request
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Server error")
    mock_post.return_value = mock_response

    # Since the function exits on error, we need to catch the SystemExit exception
    with pytest.raises(SystemExit) as exc_info:
        make_inference_request(
            'https://fake-endpoint.com',
            'fake-token',
            {'instances': [[1, 2, 3, 4]]}
        )
    # Assert that the script exits with code 1
    assert exc_info.value.code == 1
    logging.info("Test test_make_inference_request_failure was successful")

def test_integration():
    EMERALDS_TOKEN_URL = "https://emeralds-token.apps.emeralds.ari-aidata.eu"
    KSERVE_MODEL_ENDPOINT = (
        "https://model-example.models.kubeflow.emeralds.ari-aidata.eu/v1/models/model-example:predict"
    )
    username = os.environ.get('MLOPS_PLATFORM_USERNAME_PRODUCTION')
    password = os.environ.get('MLOPS_PLATFORM_PASSWORD_PRODUCTION')

    assert username is not None, "Username environment variable not set"
    assert password is not None, "Password environment variable not set"

    access_token = get_access_token(username, password, EMERALDS_TOKEN_URL)

    data = {"instances": [[6.8, 2.8, 4.8, 1.4], [1, 1, 1, 1]]}

    response = make_inference_request(KSERVE_MODEL_ENDPOINT, access_token, data)

    assert response.status_code == 200
    logging.info("Test test_integration was successful: %s", response.text)
