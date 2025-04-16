"""This script is testing on DeepSeek Vision Model API using LM-deploy."""

# !pip install lmdeploy

from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

engine_config = TurbomindEngineConfig(
    cache_max_entry_count=0.3
)
pipe = pipeline(
    'deepseek-ai/deepseek-vl-1.3b-chat',
    backend_config=engine_config
)

image_path = "/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/slices/grid_black_frame/bts_black_frame_1/item_0.png"
response = pipe((
    'Extract text from this image',
    load_image(image_path)
))

print(response)