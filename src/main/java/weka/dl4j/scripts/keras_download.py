from os import path, makedirs
from models import MODELS
from multiprocessing.pool import ThreadPool
h5_path = "output_h5"
summary_path = "output_summary"

def download_and_save_model(model_def):
    model_fn = model_def[0]
    model_name = model_def[1]

    # Download the model
    keras_model = model_fn()

    # # Save the model weights and config in a single file
    keras_model.save(path.join(h5_path, model_name + ".h5"))

    with open(path.join(summary_path, model_name + ".txt"), mode='w') as f:
        keras_model.summary(print_fn= lambda x: f.write(x + "\n"))

    return model_name

makedirs(h5_path, exist_ok=True)
makedirs(summary_path, exist_ok=True)

# Download
results = ThreadPool(8).imap_unordered(download_and_save_model, MODELS)

for name in results:
    print("\n", name, "saved\n")