# comcts_replica
```python
python infer.py \
--model 'Mulberry_llava_8b' \
--model_path 'HuanjinYao/Mulberry_llava_8b' \
--question 'Question: <Your_Question>' \
--img_path '<Your_Img_Path>'
```
infer.py
```python

import transformers
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import argparse


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


def llava_infer(model_path, question, img_path, only_output_final_answer=False):

    def output_process(answer):
        if "<s>" in answer:
            answer = answer.replace("<s>", "").strip()
        if "[/INST]" in answer:
            answer = answer.split("[/INST]")[1].strip()
        elif "ASSISTANT:" in answer:
            answer = answer.split("ASSISTANT:")[1].strip()
        elif "assistant\n" in answer:
            answer = answer.split("assistant\n")[1].strip()
        elif "<|end_header_id|>\n\n" in answer:
            answer = answer.split("<|end_header_id|>\n\n")[2].strip()

        if "</s>" in answer:
            answer = answer.split("</s>")[0].strip()
        elif "<|im_end|>" in answer:
            answer = answer.split("<|im_end|>")[0].strip()
        elif "<|eot_id|>" in answer:
            answer = answer.split("<|eot_id|>")[0].strip()
        return answer


    processor = LlavaNextProcessor.from_pretrained(model_path)

    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_flash_attention_2=True,
    )

    model = model.eval().cuda()

    kwargs = dict(
        do_sample=False, temperature=0, max_new_tokens=1024, top_p=None, num_beams=1
    )

    images = [Image.open(img_path).convert("RGB")]
    
    content = [
        {"type": 'text', "text":PROMPT.format(question=question)},
        {"type": "image"}
        ]

    conversation = [
        {
            "role": "user",
            "content": content,
        }
    ]

    prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True
    )
    inputs = processor(prompt, images, return_tensors="pt").to(
        "cuda", torch.float16
    )

    output = model.generate(**inputs, **kwargs)
    answer = processor.decode(output[0], skip_special_token=True)
    answer = output_process(answer)

    if only_output_final_answer:
        if len(answer.split('### The final answer is:')) == 2:
            answer = answer.split('### The final answer is:')[-1].strip()
            return answer
    else:
        return answer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='Mulberry_llava_8b')
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--img_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--only_output_final_answer", action='store_true')
    args = parser.parse_args()

    if args.model == 'Mulberry_llava_8b':
        answer = llava_infer(args.model_path, args.question, args.img_path, args.only_output_final_answer)
    else:
        raise NotImplementedError()

    print(answer)
```
