"""This script is testing on DeepSeek Vision Model from git."""

# !pip install git+https://github.com/deepseek-ai/deepseek-vl.git

import torch
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

model: VLChatProcessor = VLChatProcessor.from_pretrained(
    "deepseek-ai/deepseek-vl-1.3b-chat"
)
tokenizer = model.tokenizer

vl_gpt: MultiModalityCausalLM = MultiModalityCausalLM.from_pretrained(
    "deepseek-ai/deepseek-vl-1.3b-chat",
    trust_remote_code=True,
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

conversation = [
    {
        "role": "User",
        "content": "<image_placeholder>What is the text in the image?",
        "images": ["/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/slices/grid_black_frame/bts_black_frame_1/item_0.png"]
    },
    {
        "role": "Assistant",
        "content": ""
    }
]

prep_inputs = model(
    conversations=conversation,
    images=load_pil_images(conversation),
).to(vl_gpt.device)

inputs_embeds = vl_gpt.prepare_inputs_embeds(**prep_inputs)

outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prep_inputs.attention_mask,
    pad_token_id = tokenizer.eos_token_id,
    bos_token_id = tokenizer.bos_token_id,
    eos_token_id = tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True,
)

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print(answer)