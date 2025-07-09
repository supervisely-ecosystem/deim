import os
from dotenv import load_dotenv
import supervisely as sly

from supervisely_integration.serve.deim import DEIM

if sly.is_development():
    load_dotenv("local.env")
    # load_dotenv("supervisely.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

model_n = 1

# 1. Pretrained model
if model_n == 1:
    model = DEIM(
        model="DEIM D-FINE-S",
        device="cuda",
    )

# 2. Local checkpoint
elif model_n == 2:
    model = DEIM(
        model="my_models/best.pth",
        device="cuda",
    )

# 3. Remote Custom Checkpoint (Team Files)
elif model_n == 3:
    model = DEIM(
        model="/experiments/1322_Animals (Rectangle)/47676_DEIM/checkpoints/best.pth",
        device="cuda:0",
    )

image_path = "supervisely_integration/demo/img/coco_sample.jpg"
predictions = model(input=image_path)
print(f"Predictions: {len(predictions)}")
