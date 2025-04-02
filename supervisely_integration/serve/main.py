import os

from dotenv import load_dotenv

import supervisely as sly
from supervisely_integration.serve.deim import DEIM

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

model = DEIM(
    use_gui=True,
    use_serving_gui_template=True,
)
model.serve()
