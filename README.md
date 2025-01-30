# comcts_replica
```python
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python code/run_comcts.py \
    --image_dir_path ./demo_data/images \
    --data_path ./demo_data/comcts_input_data.json \
    --output_path ./output/comcts_data.jsonl \
    --max_iterations 20 \
    --exploration_weight 0.5 \
    --gpt_version 'gpt-4o' \
    --openai_api_key "Your_Open_API_Key" \
    --qwen2_vl_7b_model_path 'Qwen/Qwen2-VL-7B-Instruct' \
    --qwen2_vl_72b_model_path 'Qwen/Qwen2-VL-72B-Instruct' \
    --llama3_vision_11b_model_path 'meta-llama/Llama-3.2-11B-Vision-Instruct' 
```

run_comcts.py
```python
from openai import OpenAI
import json
# import base64
from tqdm import tqdm
import os
# import math
import argparse
from utils import *
from model import *
from comcts import *
import pdb
import time

def infer_comcts(args):
    data_path = args.data_path 
    if data_path.endswith('.jsonl'):
        data = read_jsonl(data_path)
    else:
        with open(data_path, 'r') as f:
            data = json.load(f)

    output_path = args.output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ans_file = open(output_path, "w")
    failed_search_path = args.output_path.replace('.jsonl', '_failed.jsonl')
    failed_search_file = open(failed_search_path, "w")

    # print(args.num_chunks, args.chunk_idx)
    data = get_chunk(data, args.num_chunks, args.chunk_idx)    
    
    client = OpenAI(
        base_url=args.openai_base_url,
        api_key=args.openai_api_key,        
    )

    activated_models, model_dict = init_model(args)

    for d in tqdm(data):
        comcts = CoMCTS(args, '', '', max_iterations=args.max_iterations)
        comcts.search(d, client, activated_models, model_dict, ans_file, failed_search_file)        

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--image_dir_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--qwen2_vl_72b_model_path", type=str, default=None)
    parser.add_argument("--qwen2_vl_7b_model_path", type=str, default=None)
    parser.add_argument("--qwen2_vl_2b_model_path", type=str, default=None)
    parser.add_argument("--llama3_vision_11b_model_path", type=str, default=None)
    parser.add_argument("--llava_next_8b_model_path", type=str, default=None)
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--openai_base_url", type=str, default='https://api.openai.com/v1')
    parser.add_argument("--gpt_version", type=str, default=None)
    parser.add_argument("--use_multi_thread", action='store_true')
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--eval_expert", type=list, default=['gpt-4o', 'qwen2_vl_72b']) 
    parser.add_argument("--exploration_weight", type=float, default=0.5)
    parser.add_argument("--max_iterations", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0)
    args = parser.parse_args()

    infer_comcts(args)

```
