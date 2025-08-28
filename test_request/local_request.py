import json

import requests

# URL of the model prediction endpoint
url = "http://localhost:8080/v1/models/crowd-density-model:predict"

# Data to be sent for prediction
###############################
### original code
#data = {"instances": [[6.8, 2.8, 4.8, 1.4], [1, 1, 1, 1]]}
###############################
#Changed code
data = {
    "instances": [
        [3, "13:30", "Rumbula"],   # day=3 (Wed), 13:30, zone string
        [6, "21:15", "Rumbula"]
    ]
}
# Convert data to JSON
json_data = json.dumps(data)

# Send the POST request to the model server
response = requests.post(url, data=json_data)

# Check if the request was successful
if response.status_code == 200:
    # Get the prediction result
    prediction = response.json()
    print("Prediction:", prediction)
else:
    print("Error:", response.status_code, response.text)
