import math
import random

from openai import OpenAI
import json
import base64
from tqdm import tqdm
import os
import math
import argparse
from utils import *
from model import *
from comcts import *
import pdb
import time
from collections import deque
from threading import Thread


import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import MllamaForConditionalGeneration
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
class Node:
    def __init__(self, step_text, prefix_steps, step_correctness=[], parent=None):
        """
        Initializes a node in the tree.

        step_text: Current step text.
        prefix_steps: Prefix steps text.
        step_correctness: List indicating correctness of steps (default is an empty list).
        parent: Parent node (default is None).
        """
        self.step_text = step_text  # current step text
        self.parent = parent  # parent node
        self.children = []  # children nodes
        self.visits = 0  
        self.value = 0  
        self.prefix_steps = prefix_steps    # prefix steps text
        self.step_correctness = step_correctness
        self.depth = len(self.step_correctness)
        self.text = self.prefix_steps + self.step_text

    def is_leaf(self):
        """
        Checks if the current node is a leaf node (i.e., it has no children).
        
        :return: True if the node is a leaf, False otherwise.
        """
        return len(self.children) == 0

    def best_child(self, exploration_weight=0.5):
        """
        Finds the child node with the highest UCB value in the subtree.
        
        :param exploration_weight: Weight parameter for exploration in UCB formula.
        :return: The best child node based on UCB value.
        """
        if self.is_leaf():
            return self

        best_value = -math.inf
        best_nodes = []

        for child in self.children:
            # Recursively find the best leaf node in the subtree
            best_leaf = child.best_child(exploration_weight)
            if best_leaf.is_leaf():
                ucb1 = (best_leaf.value +
                        exploration_weight * math.sqrt(math.log(best_leaf.parent.visits+1) / best_leaf.visits+1))

                # print(
                #     'UCB1 Calculation:',
                #     f'value={best_leaf.value}, parent_visits={best_leaf.parent.visits}, visits={best_leaf.visits},',
                #     f'ucb1={ucb1}, text={best_leaf.text}'
                # )

                # Update the best node list
                if ucb1 > best_value:
                    best_value = ucb1
                    best_nodes = [best_leaf]
                elif ucb1 == best_value:
                    best_nodes.append(best_leaf)

        # Return a random choice among the best nodes
        return random.choice(best_nodes)

    def add_child(self, step_text, prefix_steps, step_correctness):
        """
        Adds a child node to the current node.

        :param step_text: Step text for the child node.
        :param prefix_steps: Prefix steps for the child node.
        :param step_correctness: Correctness list for the child node.
        :return: The newly added child node.
        """
        child_node = Node(step_text, prefix_steps, step_correctness, parent=self)
        self.children.append(child_node)
        return child_node

    def update_visits(self):
        """
        Increments the visit count of the current node.
        """
        self.visits += 1

    def update_value(self, parent_visits, parent_value, new_value, new_visits):
        """
        Updates the value of the node based on the parent and new observations.

        :param parent_visits: Number of visits to the parent node.
        :param parent_value: Value of the parent node.
        :param new_value: New value to integrate into the node.
        :param new_visits: Number of new visits to incorporate.
        """
        total_visits = parent_visits + new_visits + self.visits
        self.value = (parent_visits * parent_value + new_value + self.value * self.visits) / total_visits


class CoMCTS:
    def __init__(self, args, step_text, prefix_steps, max_iterations=15):
        self.root = Node(step_text, prefix_steps)
        self.max_iterations = max_iterations
        self.args = args

    def search(self, data, client, activated_models, model_dict, ans_file, failed_search_file):
        iteration = 0        
        question = question_process(data)
        gt_answer = data["conversations"][1]['value']
        img_path = find_img_path(data, self.args)
        base64_image = encode_image(img_path)
        temperature = 0.9
# def question_process(d):
#     if 'question' in d.keys():
#         question = d['question']
#     elif '\nAnswer the question with a short answer.' in d["conversations"][0]['value']:
#         question = d["conversations"][0]['value'].replace('\nAnswer the question with a short answer.', '')
#     elif "\nAnswer with the option's letter from the given choices directly." in d["conversations"][0]['value']:
#         question = d["conversations"][0]['value'].replace("\nAnswer with the option's letter from the given choices directly.", '')
#     elif "\nAnswer the question using a single word or phrase." in d["conversations"][0]['value']:
#         question = d["conversations"][0]['value'].replace("\nAnswer the question using a single word or phrase.", '')
#     elif "<image>\nFirst perform reasoning, then finally select the question from the choices in the following format: Answer: xxx.\n" in d["conversations"][0]['value']:
#         question = d["conversations"][0]['value'].replace('<image>\nFirst perform reasoning, then finally select the question from the choices in the following format: Answer: xxx.\n', '')
#     elif "<image>\nBased on the image, directly select the correct answer for the following question:\n" in d["conversations"][0]['value']:
#         question = d["conversations"][0]['value'].replace('<image>\nBased on the image, directly select the correct answer for the following question:\n', '')
#     else:
#         question = d["conversations"][0]['value']

#     if not question.startswith('Question:'):
#         question = 'Question: ' + question

#     return question

# def find_img_path(d,args):
#     if os.path.exists(os.path.join(args.image_dir_path, d['image'])):
#         img_path = os.path.join(args.image_dir_path, d['image'])
#     elif os.path.exists(d['image']):
#         img_path = d['image']
#     else:
#         raise ValueError(f"Image path not found: {d['image']}")

#     return img_path

# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode('utf-8')        
#         while True:
#             print(f'Start the {iteration} round of search')            
#             iteration += len(activated_models)

            ## Select Node
            node = self.root
            while not node.is_leaf():
                node = node.best_child(self.args.exploration_weight)
            prefix_steps = reformat_reasoning_prefix(node.text)

            # init comcts_dict
            comcts_dict = {activated_model: {'valid': 1} for activated_model in activated_models}

            ## Expansion: Generate responses for each model
            for model_name in activated_models:
                response = self._generate_model_response(
                    model_name, question, prefix_steps, base64_image, temperature, model_dict, img_path, iteration, client
                )
                comcts_dict[model_name]['response'] = response if response else ''

            # Validate responses
            for model_name in activated_models:
                response = comcts_dict[model_name]['response']
                if not check_validity(response):
                    comcts_dict[model_name]['valid'] = -1

            ## Simulation, Error Positioning
            all_correctness = self._determine_correctness(
                comcts_dict, client, question, gt_answer, activated_models
            )
            if len(all_correctness) == 0:
                continue

            expand_node = node
            if 1 in all_correctness:
                comcts_dict = self._process_correct_paths(
                    model_dict, comcts_dict, expand_node, question, gt_answer, img_path, base64_image, temperature, activated_models, client, prefix_steps
                )
                comcts_dict['image'] = data['image']
                comcts_dict['question'] = question
                comcts_dict['prefix_prompt'] = prefix_steps
                comcts_dict['conversations'] = data['conversations']
                ans_file.write(json.dumps(comcts_dict) + "\n")
                ans_file.flush()
                break
            else:
                self._process_incorrect_paths(
                    model_dict, comcts_dict, expand_node, question, gt_answer, img_path, base64_image, temperature, activated_models, client, prefix_steps
                )

            if iteration >= self.max_iterations:
                for model_name in activated_models:
                    data[model_name] = {'response': comcts_dict[model_name]['response'], 'valid': comcts_dict[model_name]['valid']}
                data['prefix_prompt'] = prefix_steps
                failed_search_file.write(json.dumps(data) + "\n")
                failed_search_file.flush()
                break

    def _generate_model_response(self, model_name, question, prefix_steps, base64_image, temperature, model_dict, img_path, iteration, client):
        """Generate model-specific responses."""
        open_source_prefix_steps = "### Image Description:" if prefix_steps == '' else ''
        try:
            if model_name == 'gpt-4o':
                if iteration == 0:
                    return gpt_forward(client, PROMPT.format(question=question), base64_image, temperature)
                return gpt_forward(client, GPT_PREFIX_PROMPT.format(question=question, reasoning_prefix=prefix_steps), base64_image, temperature)
            elif 'qwen2_vl' in model_name:
                return modified_qwen_response(
                    qwen2_vl_forward(
                        model_dict[model_name]['model'],
                        model_dict[model_name]['processor'],
                        PROMPT.format(question=question),
                        open_source_prefix_steps + prefix_steps,
                        img_path
                    )
                )
            elif 'llama_vision' in model_name:
                return modified_llama_response(
                    llama_forward(
                        model_dict[model_name]['model'],
                        model_dict[model_name]['processor'],
                        PROMPT.format(question=question),
                        open_source_prefix_steps + prefix_steps,
                        img_path
                    )
                )
        except Exception as e:
            print(f"Error generating response for {model_name}: {e}")
            time.sleep(1)
            return None


    def _determine_correctness(self, comcts_dict, client, question, gt_answer, activated_models):
        """determine correctness."""
        all_correctness = []
        for model_name in activated_models:
            if comcts_dict[model_name]['valid'] == -1:
                continue
            response = comcts_dict[model_name]['response']
            model_answer = response.split('### The final answer is:')[-1].strip()
            while True:
                try:
                    judge_output = gpt_forward(client, JUDGE_PROMPT.format(question=question, model_answer=model_answer, gt_answer=gt_answer))
                    break
                except Exception as e:
                    time.sleep(1)
                    print(e)
            
            is_correct = get_correctness(judge_output)

            all_correctness.append(is_correct)
            comcts_dict[model_name]['is_correct'] = is_correct
        
        return all_correctness
        

    def _process_correct_paths(self, model_dict, comcts_dict, expand_node, question, gt_answer, img_path, base64_image, temperature, activated_models, client, prefix_steps):
        """Handle scenarios where correct paths are found."""
        for model_name in activated_models:
            if comcts_dict[model_name]['valid'] == -1:
                continue
            depth = get_depth(comcts_dict[model_name]['response'])

            if 'gpt-4o' in self.args.eval_expert:
                is_correct = comcts_dict[model_name]['is_correct']
                while True:
                    max_try_count = 3
                    try_count = 0
                    try:
                        step_correctness_response = gpt_forward(client, LOCATE_ERROR_PROMPT.format(question=question, reasoning=comcts_dict[model_name]['response'], gt=gt_answer), base64_image, temperature)
                        step_correctness = step_correctness_to_list(step_correctness_response, depth=depth)
                        if step_correctness != [-2] or try_count > max_try_count:
                            break
                        try_count += 1
                    except Exception as e:
                        time.sleep(1)
                        print(e)

            if 'qwen2_vl_72b' in self.args.eval_expert and 'qwen2_vl_72b' in activated_models:
                qwen2_vl_step_correctness_response = qwen2_vl_forward(model_dict['qwen2_vl_72b']['model'], model_dict['qwen2_vl_7b']['processor'], \
                    LOCATE_ERROR_PROMPT.format(question=question, reasoning=comcts_dict[model_name]['response'], gt=gt_answer), '', img_path, temperature)
                qwen2_vl_step_correctness = step_correctness_to_list(qwen2_vl_step_correctness_response, depth=depth)

                if len(step_correctness) == len(qwen2_vl_step_correctness) and step_correctness != [-2] and qwen2_vl_step_correctness != [-2]:
                    for j in range(len(step_correctness)):
                        step_correctness[j] = 0.7 * step_correctness[j] + 0.3 * qwen2_vl_step_correctness[j]
                elif qwen2_vl_step_correctness != [-2] and qwen2_vl_step_correctness == [-2]:
                    step_correctness = qwen2_vl_step_correctness

            if step_correctness == [-2]:
                comcts_dict[model_name]['valid'] = -1


            prefix_steps_depth = get_depth(expand_node.text)
            suffix_steps_depth = get_depth(comcts_dict[model_name]['response']) - 1 # remove final answer
            new_step = ''
            current_node = expand_node
            new_prefix_steps = prefix_steps
            for i in range(prefix_steps_depth, suffix_steps_depth):  
                new_prefix_steps = new_prefix_steps + new_step
                new_step = get_step(comcts_dict[model_name]['response'], i+1)
                current_node = current_node.add_child(step_text=new_step, prefix_steps=new_prefix_steps, step_correctness=step_correctness[:(i+1)])
            
            ## Backpropagation
            # leaf node
            up_node = current_node
            depth_diff = suffix_steps_depth - prefix_steps_depth
            step_value = []
            for idx in range(suffix_steps_depth, 0, -1):
                if idx > prefix_steps_depth:
                    # new node
                    new_value = sum(step_correctness[prefix_steps_depth:idx])
                    up_node.update_value(parent_visits=expand_node.visits, parent_value=expand_node.value, new_value=new_value, new_visits=idx-prefix_steps_depth)
                    up_node.update_visits()
                else:
                    new_value = step_correctness[idx-1]
                    up_node.update_value(parent_visits=up_node.parent.visits, parent_value=up_node.parent.value, new_value=new_value, new_visits=1)
                    up_node.update_visits()

                step_value.insert(0, round(up_node.value,3))
                up_node = up_node.parent

            value = (current_node.value +
                    self.args.exploration_weight * math.sqrt(math.log(current_node.parent.visits+1) / current_node.visits+1))

            comcts_dict[model_name] = {'response': comcts_dict[model_name]['response'], "value": round(value,3), 'step_value': step_value, "is_correct": is_correct, 'valid': comcts_dict[model_name]['valid']}
        
        return comcts_dict
        

    def _process_incorrect_paths(self, model_dict, comcts_dict, expand_node, question, gt_answer, img_path, base64_image, temperature, activated_models, client, prefix_steps):
        """Handle scenarios where correct paths are not found."""
        for model_name in activated_models:
            if comcts_dict[model_name]['valid'] == -1:
                continue

            depth = get_depth(comcts_dict[model_name]['response'])
            if 'gpt-4o' in self.args.eval_expert:
                while True:
                    max_try_count = 3
                    try_count = 0
                    try:
                        step_correctness_response = gpt_forward(client, LOCATE_ERROR_PROMPT.format(question=question, reasoning=comcts_dict[model_name]['response'], gt=gt_answer), base64_image, temperature)
                        step_correctness = step_correctness_to_list(step_correctness_response, depth=depth)
                        if step_correctness != [-2] or try_count > max_try_count:
                            break
                        try_count += 1
                    except Exception as e:
                        time.sleep(1)
                        print(e)


            if 'qwen2_vl_72b' in self.args.eval_expert and 'qwen2_vl_72b' in activated_models:
                qwen2_vl_step_correctness_response = qwen2_vl_forward(model_dict['qwen2_vl_72b']['model'], model_dict['qwen2_vl_7b']['processor'], \
                    LOCATE_ERROR_PROMPT.format(question=question, reasoning=comcts_dict[model_name]['response'], gt=gt_answer), '', img_path, temperature)
                qwen2_vl_step_correctness = step_correctness_to_list(qwen2_vl_step_correctness_response, depth=depth)

                if len(step_correctness) == len(qwen2_vl_step_correctness) and step_correctness != [-2] and qwen2_vl_step_correctness != [-2]:
                    for j in range(len(step_correctness)):
                        step_correctness[j] = 0.7 * step_correctness[j] + 0.3 * qwen2_vl_step_correctness[j]
                elif qwen2_vl_step_correctness != [-2] and qwen2_vl_step_correctness == [-2]:
                    step_correctness = qwen2_vl_step_correctness

            if len(step_correctness) == 0:
                comcts_dict[model_name]['valid'] = -1
                continue
            if step_correctness[0] == -2:
                comcts_dict[model_name]['valid'] = -1
                continue

            comcts_dict[model_name] = {'response': comcts_dict[model_name]['response'], 'step_correctness': step_correctness, 'depth':depth, 'valid': comcts_dict[model_name]['valid']}
            

            # Prune the first node smaller than the threshold.
            comcts_dict[model_name] = prune_tree(comcts_dict[model_name], start_index=get_depth(expand_node.text), threshold=self.args.threshold)
            if comcts_dict[model_name]['valid'] == -1:
                up_node = expand_node
                while up_node.parent is not None:
                    # do not update the root node
                    up_node.update_visits()
                    up_node = up_node.parent
                continue

            pruned_response = comcts_dict[model_name]['response']
            updated_step_correctness = comcts_dict[model_name]['step_correctness']

            # add nodes
            prefix_steps_depth = get_depth(expand_node.text)
            pruned_steps_depth = get_depth(pruned_response)
            new_step = ''
            current_node = expand_node
            new_prefix_steps = prefix_steps
            # print(prefix_steps_depth, pruned_steps_depth, comcts_dict)
            for i in range(prefix_steps_depth, pruned_steps_depth):
                new_prefix_steps = new_prefix_steps + new_step
                new_step = get_step(pruned_response, i+1)
                current_node = current_node.add_child(step_text=new_step, prefix_steps=new_prefix_steps, step_correctness=updated_step_correctness[:(i+1)])

            ## Backpropagation
            # bottom-up update node
            up_node = current_node
            depth_diff = pruned_steps_depth - prefix_steps_depth
            for idx in range(pruned_steps_depth, 0, -1):
                if idx > prefix_steps_depth:
                    # new node
                    new_value = sum(updated_step_correctness[prefix_steps_depth:idx])
                    up_node.update_value(parent_visits=expand_node.visits, parent_value=expand_node.value, new_value=new_value, new_visits=idx-prefix_steps_depth)
                    up_node.update_visits()
                else:
                    new_value = updated_step_correctness[idx-1]
                    up_node.update_value(parent_visits=up_node.parent.visits, parent_value=up_node.parent.value, new_value=new_value, new_visits=1)
                    up_node.update_visits()

                up_node = up_node.parent
