import io
import logging
import os
from typing import Dict, Union

import joblib
import numpy as np
########### extra imports
from datetime import datetime  
import tensorflow as tf        
###########
from kserve import InferRequest, Model, ModelServer
from kserve.errors import InvalidInput
from minio import Minio

logging.basicConfig(level=logging.INFO)

MINIO_ACCESS_KEY = os.environ['MINIO_ACCESS_KEY']
MINIO_SECRET_ACCESS_KEY = os.environ['MINIO_SECRET_ACCESS_KEY']
MINIO_API_URL = os.environ['MINIO_API_URL']
MINIO_BUCKET = os.environ['MINIO_BUCKET']

########### model (h5) & mapping (joblib)
FOLDER_PREFIX = 'uc3/'
MODEL_PATH = os.environ.get('MODEL_PATH', FOLDER_PREFIX + 'NN_En_Ex.h5')
ZONE_MAP_PATH = os.environ.get('ZONE_MAP_PATH', FOLDER_PREFIX + 'forward_zone_map.joblib')
###########

########### Extra function
def hhmm_to_window(hhmm: str) -> int:
    """Convert "HH:MM" string to 15-minute window index (0–95)."""
    t = datetime.strptime(hhmm, "%H:%M")
    return (t.hour * 60 + t.minute) // 15
###########

class CrowdPred(Model):
 """
    Crowd prediction model predicts the number of entries & exits per zone, day of the week and timewindow for Riga.

    Args:
        name (str): The name of the inference service.

    Attributes:
        model: Zone-Embedding Keras MLP regression model.

    Methods:
        load(): Loads the model from the object storage.
        predict(request: Dict) -> Dict: Makes predictions based on the input request.

    """


    def __init__(self, name: str):
        super().__init__(name)
        self.ready = False
        self.model = None
        ########### Extra attibute
        self.forward_zone_map = None
        ###########  
        self.minio_client = Minio(MINIO_API_URL, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_ACCESS_KEY)

        ########### Extra constants for min–max scaling
        self._DAY_MIN, self._DAY_MAX = 1, 7
        self._TW_MIN,  self._TW_MAX  = 1, 95
        ###########

    def load(self):
        """
        Loads the model (.h5) and the forward_zone_map.joblib from MinIO.
        """
        # response = self.minio_client.get_object(MINIO_BUCKET, MODEL_PATH)
        # with io.BytesIO(response.data) as f:
        #     self.model = joblib.load(f)
        with io.BytesIO(self.minio_client.get_object(MINIO_BUCKET, MODEL_PATH).data) as f:
            self.model = tf.keras.models.load_model(f)
        with io.BytesIO(self.minio_client.get_object(MINIO_BUCKET, ZONE_MAP_PATH).data) as f:
            self.forward_zone_map = joblib.load(f)
         
        self.ready = True

    def preprocess(self, payload: Union[Dict, InferRequest], headers: Dict[str, str] = None) -> np.ndarray:
        if isinstance(payload, Dict) and "instances" in payload:
            logging.info("Inference request: %s", str(payload))
            inputs = payload["instances"]
            inputs = np.array(inputs)
            return inputs
        else:
            raise InvalidInput(
                "Invalid payload. Expected: " '{"instances": [[day(1-7), "HH:MM", "zone"], ...]}'
            )

    def predict(self, input_array: np.ndarray, headers: Dict[str, str] = None) -> Dict:
        """
        Makes predictions based on the input request 
        Returns Dict: A dictionary containing the 'prediction' field with the predicted value.
        """
        # result = []
        # for array in input_array:
        #     prediction = self.model.predict(array.reshape(1, -1))
        #     result.append(prediction[0])
        # return {"prediction": str(np.array(result))}

        ##### Changed
        # 1. split columns
        day_raw  = input_array[:, 0].astype(int)
        time_raw = input_array[:, 1].astype(str)
        zone_raw = input_array[:, 2].astype(str)

        # 2. feature engineering
        vec_hhmm = np.vectorize(hhmm_to_window)
        time_window = vec_hhmm(time_raw).astype(int)

        # scale numeric
        day_scaled = (day_raw - self._DAY_MIN) / (self._DAY_MAX - self._DAY_MIN)
        tw_scaled  = (time_window - self._TW_MIN) / (self._TW_MAX - self._TW_MIN)
        num_batch  = np.stack([day_scaled, tw_scaled], axis=1).astype("float32")

        # categorical ➜ int index
        vec_zone = np.vectorize(self.forward_zone_map.__getitem__)
        em_batch = vec_zone(zone_raw).astype("int32").reshape(-1, 1)

        # 3. inference
        preds = self.model.predict([num_batch, em_batch])

        return {"prediction": preds.tolist()}  




if __name__ == "__main__":
    # run the inference service
    logging.info(f"Starting CrowdPred {datetime.now()}")
    model = CrowdPred("crowd-density-model")
    model.load()
    ModelServer().start([model])
