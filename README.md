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

```python

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import MllamaForConditionalGeneration



def init_model(args):

    activated_models = []
    model_dict={}
    if args.gpt_version is not None:
        activated_models.append(args.gpt_version)

    # qwen2vl_7b
    if args.qwen2_vl_7b_model_path is not None:
        print('init qwen2 vl 7b model')
        qwen2_vl_7b_model  = Qwen2VLForConditionalGeneration.from_pretrained(
            args.qwen2_vl_7b_model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation='flash_attention_2',
        )
        min_pixels = 256*28*28
        max_pixels = 1280*28*28
        qwen2_vl_7b_processor = AutoProcessor.from_pretrained(args.qwen2_vl_7b_model_path, min_pixels=min_pixels, max_pixels=max_pixels)

        activated_models.append('qwen2_vl_7b')
        model_dict['qwen2_vl_7b'] = {'model': qwen2_vl_7b_model, 'processor': qwen2_vl_7b_processor}

    # qwen2vl_2b
    if args.qwen2_vl_2b_model_path is not None:
        print('init qwen2 vl 2b model')
        qwen2_vl_2b_model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.qwen2_vl_2b_model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation='flash_attention_2',
        )
        min_pixels = 256*28*28
        max_pixels = 1280*28*28
        qwen2_vl_2b_processor = AutoProcessor.from_pretrained(args.qwen2_vl_2b_model_path, min_pixels=min_pixels, max_pixels=max_pixels)

        activated_models.append('qwen2_vl_2b')
        model_dict['qwen2_vl_2b'] = {'model': qwen2_vl_2b_model, 'processor': qwen2_vl_2b_processor}

    # qwen2vl_72b
    if args.qwen2_vl_72b_model_path is not None:
        print('init qwen2 vl 72b model')
        qwen2_vl_72b_model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.qwen2_vl_72b_model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation='flash_attention_2',
        )
        min_pixels = 256*28*28
        max_pixels = 1280*28*28
        qwen2_vl_72b_processor = AutoProcessor.from_pretrained(args.qwen2_vl_72b_model_path, min_pixels=min_pixels, max_pixels=max_pixels)

        activated_models.append('qwen2_vl_72b')
        model_dict['qwen2_vl_72b'] = {'model': qwen2_vl_72b_model, 'processor': qwen2_vl_72b_processor}

    # llama3.2_vision_11b
    if args.llama3_vision_11b_model_path is not None:
        print('init llama3.2 vision 11b model')
        llama_vision_11b_model = MllamaForConditionalGeneration.from_pretrained(
            args.llama3_vision_11b_model_path, torch_dtype=torch.bfloat16, device_map="auto",
        )
        llama_vision_11b_processor = AutoProcessor.from_pretrained(args.llama3_vision_11b_model_path)

        activated_models.append('llama_vision_11b')
        model_dict['llama_vision_11b'] = {'model': llama_vision_11b_model, 'processor': llama_vision_11b_processor}

    if args.llava_next_8b_model_path is not None:
        llava_next_8b_model = LlavaNextForConditionalGeneration.from_pretrained(
            args.llava_next_8b_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation='flash_attention_2'
            # use_flash_attention_2=True,
        )
        llava_next_8b_processor = LlavaNextProcessor.from_pretrained(args.llava_next_8b_model_path)

```

JUDGE_PROMPT = """Evaluate whether the model's answer matches the correct result. 

- If it does not align, respond with 'No'.
- If there is a logical error in the reasoning steps, respond with 'No'.
- If the model's answer aligns with the correct result, respond with 'Yes'. 

Provide only 'Yes' or 'No' as the output, with no explanation.

The question is: {question}

The model's answer is: {model_answer}

The correct result is: {gt_answer}"""



PROMPT = """Generate an image description based on the question.
Then, provide a rationale to analyze the question.
Next, generate a step-by-step reasoning process to solve the problem. Ensure the steps are logical and concise.
Finally, provide a concise summary of the final answer in the following format: 'The final answer is: xxx'. If the question is multiple-choice, provide the options along with their content. If it is free-form, directly present the final result. Do not provide any explanation.

Format your response with the following sections, separated by ###:
### Image Description:
### Rationales:
### Let's think step by step.
### Step 1:
### Step 2:
...
### The final answer is: 

{question}"""



LOCATE_ERROR_PROMPT = ''''### Question:
{question}

### Ground truth answer:
{gt}

### Reasoning steps:
{reasoning}

Given the question and reasoning steps listed above, along with the corresponding ground truth answer, please evaluate the correctness of the image description, rationales, and each step of the reasoning process.

Requirements:
1. Output the decision ("correct", "neutral", "incorrect") for each step following the format of "Final Decision:\nImage Description: [your decision]; Rationales: [your decision]; Let's think step by step: [your decision]; Step 1: [your decision]; Step 2: [your decision]; ...";
2. Do not provide any explanation.'''



GPT_PREFIX_PROMPT = """Generate an image description based on the question.
Then, provide a rationale to analyze the question.
Next, generate a step-by-step reasoning process to solve the problem. Ensure the steps are logical and concise.
Finally, provide a concise summary of the final answer in the following format: 'The final answer is: xxx'. If the question is multiple-choice, provide the options along with their content. If it is free-form, directly present the final result. Do not provide any explanation.

Format your response with the following sections, separated by ###:
### Image Description:
### Rationales:
### Let's think step by step.
### Step 1:
### Step 2:
...
### The final answer is: 

{question}

Please complete the response based on the reasoning prefix without altering its content.

Reasoning prefix: {reasoning_prefix}"""
```

```python
from openai import OpenAI
import json
import base64
from tqdm import tqdm
import os
import math
import argparse
from PIL import Image
from collections import Counter
import heapq
import math

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def read_jsonl(file):
    with open(file, 'r') as f:
        data = [json.loads(line) for line in f]
    return data


def question_process(d):
    if 'question' in d.keys():
        question = d['question']
    elif '\nAnswer the question with a short answer.' in d["conversations"][0]['value']:
        question = d["conversations"][0]['value'].replace('\nAnswer the question with a short answer.', '')
    elif "\nAnswer with the option's letter from the given choices directly." in d["conversations"][0]['value']:
        question = d["conversations"][0]['value'].replace("\nAnswer with the option's letter from the given choices directly.", '')
    elif "\nAnswer the question using a single word or phrase." in d["conversations"][0]['value']:
        question = d["conversations"][0]['value'].replace("\nAnswer the question using a single word or phrase.", '')
    elif "<image>\nFirst perform reasoning, then finally select the question from the choices in the following format: Answer: xxx.\n" in d["conversations"][0]['value']:
        question = d["conversations"][0]['value'].replace('<image>\nFirst perform reasoning, then finally select the question from the choices in the following format: Answer: xxx.\n', '')
    elif "<image>\nBased on the image, directly select the correct answer for the following question:\n" in d["conversations"][0]['value']:
        question = d["conversations"][0]['value'].replace('<image>\nBased on the image, directly select the correct answer for the following question:\n', '')
    else:
        question = d["conversations"][0]['value']

    if not question.startswith('Question:'):
        question = 'Question: ' + question

    return question

def find_img_path(d,args):
    if os.path.exists(os.path.join(args.image_dir_path, d['image'])):
        img_path = os.path.join(args.image_dir_path, d['image'])
    elif os.path.exists(d['image']):
        img_path = d['image']
    else:
        raise ValueError(f"Image path not found: {d['image']}")

    return img_path


def gpt_forward(client, prompt, base64_image=None, temperature=0.9):
    content = [{
                    "type": "text",
                    "text": prompt
                }]
    if base64_image is not None:
        content.append({
            "type": "image_url",
            "image_url":{
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ]
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=message,
        temperature = temperature
    )

    return completion.choices[0].message.content


def get_correctness(judge_output):
    if 'yes' in judge_output.lower() and 'no' not in judge_output.lower():
        return 1
    else:
        return -1

def qwen2_vl_forward(model, processor, question, prefix_prompt, img_path, temperature=0.9):
    messages = [
        {
            'role': "system",
            "content": 'You are a helpful assistant.'
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_path,
                },
                {"type": "text", "text": question},
            ],
        },
    ]

    image = Image.open(img_path)
    h, w = image.size
    if h < 28 or w < 28:
        factor = 28 / h if h < w else 28 / w
        if h < w:
            image = image.resize((28, int(w * factor)))
        else:
            image = image.resize((int(h * factor), 28))

    texts = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefix_prompt]
                
    inputs = processor(
        text=texts,
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=1024, repetition_penalty=1, temperature=temperature)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return prefix_prompt + output_texts

def llama_forward(model, processor, question, prefix_prompt, img_path, temperature=0.9):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_path,
                },
                {"type": "text", "text": question},
            ],
        },
    ]
    image = Image.open(img_path)
    texts = processor.apply_chat_template(messages, add_generation_prompt=True) + prefix_prompt

    inputs = processor(image, texts, return_tensors="pt").to('cuda')

    generated_ids = model.generate(**inputs, max_new_tokens=1024, temperature=temperature)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return prefix_prompt + output_texts

def check_data(steps):
    steps = steps.split('###')
    len_steps = len(steps)
    for i, step in enumerate(steps):
        if i == 0:
            if step == '':
                continue
            else:
                return False

        elif i == 1:
            if step.strip().startswith('Image Description:'):
                continue
            else:
                return False

        elif i == 2:
            if step.strip().startswith('Rationales:'):
                continue
            else:
                return False
        
        elif i == 3:
            if step.strip().startswith("Let's think step by step"):
                continue
            else:
                return False
        
        elif i > 3 and i < len_steps - 1:
            if step.strip().startswith(f"Step {i-3}"):
                continue
            else:
                return False

        elif i == len_steps-1:
            if step.strip().startswith("The final answer is:") and i > 4:
                continue
            else:
                return False
    return True

def check_validity(response):
    if not check_data(response):
        return False
    if len(response.split('### The final answer is:')) == 2:
        return True
    return False

def get_depth(response):
    # image description is depth 1
    steps = response.split('###')
    return len(steps) - 1


def get_step(response, depth):
    res = response.split('###')
    return '###' + res[depth]


def step_correctness_to_list(response, depth):
    step_correctness_list = []
    output_scores = response.split('Final Decision:')[-1].strip()
    output_scores_list = output_scores.split(';')
    for score in output_scores_list:
        if 'incorrect' in score.lower():
            step_correctness_list.append(-1)
        elif 'neutral' in score.lower():
            step_correctness_list.append(0)
        elif 'correct' in score.lower():
            step_correctness_list.append(1)
    if len(step_correctness_list) != depth-1:
        return [-2]
    return step_correctness_list
    

def prune_response(response, idx): 
    steps = response.split('###')
    len_steps = len(steps) - 1
    if idx == 0:
        return ''
    elif idx == 1:
        index = response.find('### Rationales:')
        if index != -1:
            return response[:index]
        else:
            return response
    elif idx == 2:
        index = response.find("### Let's think step by step") if response.find("### Let's think step by step") != -1 else response.find("###Let's think step by step")
        if index != -1:
            return response[:index]
    elif idx > 2 and idx < len_steps-1:
        index = response.find(f'### Step {idx-2}')
        if index != -1:
            return response[:index]
    elif idx == len_steps-1:
        index = response.find(f'### The final answer is:')
        if index != -1:
            return response[:index]
    else:
        return ''


def prune_tree(comcts_dict, start_index, threshold=0):
    pruned_comcts_dict = dict()
    step_correctness_list = comcts_dict['step_correctness']
    first_less_than_zero_idx = -1
    for i, value in enumerate(step_correctness_list):
        if i < start_index:
            continue
        if value < threshold:
            first_less_than_zero_idx = i
            break

    if first_less_than_zero_idx == -1 or first_less_than_zero_idx == 0:
        comcts_dict['valid'] = -1
        return comcts_dict
    
    pruned_response = prune_response(comcts_dict['response'], first_less_than_zero_idx)
    pruned_step_correctness = step_correctness_list[:first_less_than_zero_idx]

    pruned_comcts_dict['response'] = pruned_response
    pruned_comcts_dict['step_correctness'] = pruned_step_correctness
    pruned_comcts_dict['valid'] = comcts_dict['valid']

    return pruned_comcts_dict


def modified_qwen_response(response):
    for i in range(1, 15):
        step_idx = f'Step {i}:'
        if f'### Step {i}:' not in response and step_idx in response:
            if response.count(step_idx) == 1:
                response = response.replace(step_idx, f'### Step {i}:')

    if "### Final Answer:\nThe final answer is:" in response:
        response = response.replace('### Final Answer:\nThe final answer is:', '### The final answer is:')
    elif "### Final Answer:" in response:
        response = response.replace('### Final Answer:', '### The final answer is:')
    
    if "### Rationale:" in response and "### Rationales" not in response:
        response = response.replace('### Rationale:', '### Rationales:')

    return response

def modified_llama_response(response):
    for i in range(1, 15):
        step_idx = f'Step {i}:'
        if f'### Step {i}:' not in response and step_idx in response:
            if response.count(step_idx) == 1:
                response = response.replace(step_idx, f'### Step {i}:')

    if '### The final answer' not in response and "The final answer is" in response:
        response = response.replace("The final answer is", '### The final answer is:')
    
    if "### Rationale:" in response and "### Rationales" not in response:
        response = response.replace('### Rationale:', '### Rationales:')

    return response


def reformat_reasoning_prefix(reasoning):
        if '### The final answer is:' in reasoning:
            raise ValueError()
        reasoning_list = reasoning.split('###')
        output = ''
        len_steps = len(reasoning_list)
        for i, step in enumerate(reasoning_list):
            if i == 0:
                continue
            if i == 1:
                step = '### Image Description:' + ('###' + step).replace('### Image Description:', '').strip()
                output = output + step.replace('### Image Description:', '### Image Description:\n') + '\n\n'
            elif i == 2:
                step = '### Rationales:' + ('###' + step).replace('### Rationales:', '').strip()
                output = output + step.replace('### Rationales:', '### Rationales:\n') + '\n\n'
            elif i == 3:
                output = output + '### ' + step.strip() + '\n\n'
            elif i > 3:
                step = f'### Step {i-3}:' + ('###' + step).replace(f'### Step {i-3}:', '').strip()
                output = output + step.replace(f'### Step {i-3}:', f'### Step {i-3}:\n') + '\n\n'
        return output
```
    
