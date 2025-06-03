import os

from dotenv import load_dotenv

import supervisely as sly

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api.from_env()

team_id = sly.env.team_id()
workspace_id = sly.env.workspace_id()


model = api.nn.deploy(
    model="DEIM/DEIM D-FINE-N",
    # model="/experiments/2067_Animals (W)/14748_DEIM/checkpoints/best.pth",
    # model="/experiments/2067_Animals (W)/14748_DEIM/checkpoints/checkpoint0024.pth",
    device="cuda:0",
)

predictions = model.predict(
    input=["supervisely_integration/demo/img/coco_sample.jpg"],
)
print(predictions)
