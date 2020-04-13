from os import path
import efficientnet.keras as efn
import keras
import efficientnet
from models import MODELS
from multiprocessing.pool import ThreadPool
save_path = "."

def download_and_save_model(model_def):
    model_fn = model_def[0]
    model_name = model_def[1]

    # Download the model
    keras_model = model_fn()

    # # Save the model weights and config in a single file
    keras_model.save(path.join(save_path, model_name + ".h5"))

    return model_name

# Download
results = ThreadPool(8).imap_unordered(download_and_save_model, MODELS)

for name in results:
    print("\n", name, "saved\n")