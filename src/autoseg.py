from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm
import json
import segeval
from typing import List
from collections import defaultdict
from statsmodels.stats import inter_rater as irr
import numpy as np
import os
from nltk.tokenize import sent_tokenize
import anthropic
from sklearn.metrics import f1_score, cohen_kappa_score
# from vllm import LLM, SamplingParams
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
API_KEY = {
}
ANTHROPIC = ""
GPT_VERSION = "gpt-4o-2024-05-13"

def alternative_load_dataset(dataset_name, dataset_split):
    return load_dataset('data/DTS_session_datasets', data_files={'train': f'{dataset_name}_{dataset_split}.jsonl'})['train']

def init_llm(model_name):
    from vllm import LLM
    from transformers import AutoTokenizer
    model = LLM(model=model_name,
                tensor_parallel_size=4,
                gpu_memory_utilization=0.9,
                enforce_eager=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    
    return model, tok

class SegAnnotator:
    def __init__(self, key_owner='', template='no_dst', model=GPT_VERSION, change_speaker=False):
        self.key_owner = key_owner
        
        self.gpt = OpenAI(api_key = API_KEY[self.key_owner])
        self.sonnet = anthropic.Anthropic(api_key=ANTHROPIC)
        self.old = template == 'no_dst'
        self.template_name = template
        self.template = ''
        if template != 'plain':
            with open(f'prompts/{template}.prompt', 'r', encoding='utf-8') as f:
                self.template = f.read()
        
        
        self.result = {}
        self.verbose = False
        if 'gpt' in model or 'claude' in model:
            self.model = model
        else:
            self.model, self.tok = init_llm(model)
        self.domain = None
        self.speaker_map = None
        self.change_speaker = change_speaker
        self.acc_usage = {
            '$': 0.0,
            'in': 0,
            'out': 0
        }
        self.fewshot_list = []
    
    def compute_cost(self, usage):
        cost = 0
        cost += usage.completion_tokens * 15 / 1000000
        cost += usage.prompt_tokens * 5 / 1000000
        self.acc_usage['$'] += cost
        self.acc_usage['in'] += usage.prompt_tokens
        self.acc_usage['out'] += usage.completion_tokens
        return cost
    
    def create_prompt(self, data):
        dialogue = [line.split(': ')[-1].strip() for line in data.split('[NEWLINE]') if line.strip() != '[BOUNDARY]']
        # making prompt
        ## define task
        prefix = [
            "Dialogue Segmentation aims to segment a dialogue D = {U1, U2, ..., Un} into several parts according to their discussing topics.\n",
            "Please help me to segment the following dialogue: \n"
        ]
        suffix = [
            "\nOutput format: Part i: Ua-Ub\n",
            "\n=====\nOutput example:\nPart 1: U1-U4\nPart 2: U5-U6\n=====\n"
        ]
        
        utterances = []
        for i, utterance in enumerate(dialogue):
            utterances.append(f"U{i+1}: {utterance}\n")
            
        prompt = ''.join(prefix + utterances + suffix)

        return prompt
        
    def fill_prompt(self, data):
        turn_inputs = []
        if self.template == '':
            return self.create_prompt(data)
        
        if self.old:
            turn_template = "<T[turn_idx]>\n<User>[user]</User>\n<AI>[ai]</AI>\n</T[turn_idx]>"
        
            dialogue = [line.split(': ')[-1].strip() for line in data.split('[NEWLINE]') if line.strip() != '[BOUNDARY]']
        
            for idx in range(0, len(dialogue) - 1, 2):
                user, ai = dialogue[idx], dialogue[idx + 1]
                turn_idx = (int)(idx / 2)
                turn_inputs.append(turn_template.replace('[turn_idx]', str(turn_idx)).replace('[user]', user).replace('[ai]', ai))
        else:
            turn_template = "<U[uttr_idx]>\n<speaker>[speaker]</speaker>\n<utterance>[uttr]</utterance>\n</U[uttr_idx]>"
            if 'no_dst' in self.template_name:
                turn_template = "<U[uttr_idx]>\n<[speaker]>[uttr]</[speaker]>\n</U[uttr_idx]>"
            
            speakers = [line.split(': ')[0].strip() for line in data.split('[NEWLINE]') if line.strip() != '[BOUNDARY]']
            dialogues = [line.split(': ')[-1].strip() for line in data.split('[NEWLINE]') if line.strip() != '[BOUNDARY]']
            
            for idx in range(len(speakers)):
                turn_inputs.append(
                    turn_template.replace('[uttr_idx]', str(idx))\
                        .replace('[speaker]', self.speaker_map[speakers[idx]])\
                        .replace('[uttr]', dialogues[idx])
                )
            
        dialogue = '\n'.join(turn_inputs)
        prompt = self.template.replace("{XML-structured dialogue}", dialogue)
        if self.verbose: print(prompt)
        return prompt
    
    def infer(self, prompt):
        messages = []
        
        if self.template == '':
            messages += [{
                'role': 'system',
                'content': "You are a helpful assistance to segment give dialogues.\nPlease follow the output format.\nDO NOT explain."
            }]
        
        messages += self.fewshot_list
        messages += [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        
        chat_completion = self.gpt.chat.completions.create(
            messages=messages,
            temperature=0.0,
            model=self.model,
        )
        return chat_completion
    
    def infer_sonnet(self, prompt):
        message = self.sonnet.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=8000,
            temperature=0.0,
            system="Following TASK instruction, OUTPUT_FORMAT, array of valid label and INPUT dialogue",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        response_texts = [block.text for block in message.content if hasattr(block, 'text')]
        result_text = " ".join(response_texts)
        return result_text
    
    def infer_local(self, prompt, model, tok):
        from vllm import SamplingParams
        messages = []
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=4096,
        )
        if self.template == '':
            messages += [{
                'role': 'system',
                'content': "You are a helpful assistance to segment give dialogues.\nPlease follow the output format.\nDO NOT explain."
            }]
        
        messages += self.fewshot_list
        messages += [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        chat = tok.apply_chat_template(messages, tokenize=False)
        response = model.generate(chat,
                                  sampling_params=sampling_params)
        return response[0].outputs[0].text        
        
    def construct_fewshot_example(self, example_path, dialogue):
        example_prompt = self.fill_prompt(dialogue)
        with open(example_path, 'r') as f:
            example_output = f.read()
        self.fewshot_list.extend([
            {
            "role": "user",
            "content": example_prompt
        },  {
            "role": "assistant",
            "content": example_output
        }])
    
    def load_data(self, dataset_name, dataset_split, ratio=1.0, start=0):
        self.domain = dataset_name
        self.speaker_map = {
                'user' : 'user',
                'agent' : 'agent'
            }
        if self.domain == 'tiage' and self.change_speaker:
            self.speaker_map = {
                'user' : 'speaker1',
                'agent' : 'speaker2'
            }
        
            
        dataset = alternative_load_dataset(dataset_name, dataset_split)
        
        ratio = max(0, ratio)
        n_samples = ratio if ratio >= 1 else int(len(dataset['dialogue']) * ratio)
        
        end = min(len(dataset['dialogue']), start + n_samples)
        
        print(f"DATASET RANGE: [{start}:{end}]")
    
        dataset = dataset.select(range(start, end))
        return dataset     
    
    def data_process(self, dataset_name, dataset_split, ratio=1.0, token_check_only=False, start=0):
        dataset = self.load_data(dataset_name, dataset_split, ratio, start)
        if token_check_only: return
        for data in tqdm(dataset):
            prompt = self.fill_prompt(data['dialogue'])
            
            if self.model == 'gpt-4o':
                result = self.infer(prompt)
                self.result[data['id']] = result.choices[0].message.content
                cost = self.compute_cost(result.usage)
            elif self.model =='claude':
                result = self.infer_sonnet(prompt)
                self.result[data['id']] = result
            else:
                result = self.infer_local(prompt, model=self.model, tok=self.tok)
                self.result[data['id']] = result.split('|>\n\n')[-1]
                
        if self.model == 'gpt-4o': 
            with open('usage.json', 'r') as f:
                usage = json.load(f)
                
            try:
                usage[self.key_owner] += self.acc_usage['$']
            except:
                usage[self.key_owner] = self.acc_usage['$']
            
            print('[current cost]')
            for k, v in self.acc_usage.items():
                print(f'{k:<3}: {v}')
                
            print('[acc cost]')
            for k, v in usage.items():
                print(f'{k:<3}: {v}$')
            
            with open('usage.json', 'w') as f:
                json.dump(usage, f)
                        
    def save_result_json(self, save_path='result_json.json'):
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.result, f, indent=4, ensure_ascii=False)

def process(domain:str, template:str='fift', amount:str=30, start=0, save_path=None, key_owner='mh', change_speaker=False, fewshot_idxs = [], model='gpt-4o') -> None:
    if save_path is None:
        save_path=f'results/{template}_{domain}.json'
    
    agent = SegAnnotator(
        key_owner=key_owner,
        template=template, 
        model=model, 
        change_speaker=change_speaker
    )
    
    # Build Few-shot Examples
    if len(fewshot_idxs) != 0:
        fewshot_source = agent.load_data('tiage', 'train', 300, 0)
        data_ids = [f"tiage_train_{idx}" for idx in fewshot_idxs]
        for data in fewshot_source:
            if data['id'] not in data_ids: continue
            agent.construct_fewshot_example(f"few_shot_silo/{template}/{data['id']}.example", data['dialogue'])
    
    # Processing..
    agent.data_process(domain, 'test', ratio=amount, token_check_only=False, start=start)
    
    # Load Progress
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        for k, v in results.items():
            if k not in agent.result.keys():
                agent.result[k] = v
    
    agent.save_result_json(save_path=save_path)
    print(f"Saved on {save_path}")

def load_json(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def binarize_segment(seq: List[int]) -> List[int]:
    result = []
    for n in seq:
        result += [1] + [0] * (n - 1)
    result[0] = 0
    return result

def compute_metrics(preds, labels) -> dict:
    wd_score = 0
    pk_score = 0
    
    pred_binary = []
    label_binary = []
    
    mismatched_count = 0
    for pred, label in zip(preds, labels):
        if sum(pred) != sum(label):
            mismatched_count += 1
            continue
        
        wd = float(segeval.window_diff(pred, label))
        pk = float(segeval.pk(pred, label))
        
        wd_score += wd
        pk_score += pk
        
        pred_bin = binarize_segment(pred)
        label_bin = binarize_segment(label)
        if len(pred_bin) != len(label_bin):
            print(len(pred_bin))
            print(pred)
            print(pred_bin)
            print(len(label_bin))
            print(label)
            print(label_bin)
        
        pred_binary.extend(pred_bin)
        label_binary.extend(label_bin)
        
    c = len(preds)
    c -= mismatched_count
    
    wd_score /= c
    pk_score /= c
    f1 = f1_score(y_true=label_binary, y_pred=pred_binary, average='binary', zero_division=0)
    
    if mismatched_count > 0:
        print("mismatched_count: ", mismatched_count)
    return {
        'pk' : pk_score,
        'wd' : wd_score,
        'f1' : f1
    }

def extract_label(dialogue: List[str], return_output: bool = False) -> List[int]:
    boundary = [0]
    output = []
    toggle = False
    for utterance in dialogue:
        if utterance == '[BOUNDARY]':
            boundary.append(0)
            toggle = True
        else:
            boundary[-1]+=1
            output.append(toggle)
            toggle = False
    
    if return_output: return (boundary, output)
    return boundary

def extract_pred(model_output: List[bool]) -> List[int]:
    boundary = [0]
    for output in model_output:
        if output:
            boundary.append(1)
        else:
            boundary[-1] += 1
        
    return boundary

def parse_output(data_path):
    data = load_json(data_path)
    error_keys = []
    result = {}
    for k, v in data.items():
        if type(v) != str:
            print(f"Error: {k}")
            error_keys.append(k)
            continue
        # format generalization
        if v.startswith('```'):
            v = v.replace('```', '')
        while not v.startswith('<') and len(v) > 1:
            v = v[1:]

        new_key = int(k.split('_')[-1])
        result[new_key] = {
            'id' : k,
            'uttr' : [],
            'act' : []
        }
            
        # parse
        vs = v.strip().split('\n')
        for line in vs:
            if '<topic_shift' in line or '<preceding_topical_relation' in line:
                result[new_key]['uttr'].append('YES' in line.upper())
            if '<utterance_type' in line or '<dialogue_type' in line or '<intent_label' in line or '<utterance_pattern' in line or '<utterance_intent' in line:
                result[new_key]['act'].append(line.split('>')[1].split('<')[0])
                
    return result

def get_data(domain, domain_split='test'):
    dataset = alternative_load_dataset(domain, domain_split)
    
    newlined = []
    for data in dataset:
        newlined.append(data['dialogue'].split('[NEWLINE]'))
        
    return newlined

def parse_simple_output(path):
    result = load_json(path)
    slots = {}
    for k, dialogue_output in result.items():
        slot = []
        for line in dialogue_output.split('\n'):
            if 'YES' in line:
                slot.append(True)
            if 'NO' in line:
                slot.append(False)
        nk = int(k.split('_')[-1])
        slots[nk] = {
            'id' : k,
            'uttr' : slot,
            'act' : []
        }
    return slots

def parse_plain_output(domain, domain_split='test', specified_path=None):
    refs = get_data(domain, domain_split)
    result_path = specified_path if specified_path else f'results/plain_{domain}.json'

    preds = {}

    for k, v in load_json(result_path).items():
        nk = int(k.split('_')[-1])
        ref = [line.split(': ')[-1].strip() for line in refs[nk] if line.strip() != '[BOUNDARY]']
        results = v.split('\n')
        
        end_indices = []
        for line in results:
            try:
                end_index = int(line.strip().split('U')[-1]) - 1
                end_indices.append(end_index)
            except:
                continue

        predictions = [False] * len(ref)
        for end_index in end_indices:
            predictions[end_index] = True

        predictions[-1] = False
        preds[nk] = predictions
    return preds

def align_pred_label(pred, label):
    
    # Delete empty segment
    if pred[0] == 0: pred = pred[1:]
    
    # Extension for Turn-level prediction template
    if sum(pred) == sum(label) // 2:
        pred = [p * 2 for p in pred]
    
    # Fix micro parsing error
    if sum(pred) == sum(label) + 1:
        pred[0]-=1
    if sum(pred) == sum(label) - 1:
        pred[-1]+=1
        
    n_pred, n_label = [p for p in pred if p != 0], [l for l in label if l != 0]
    return n_pred, n_label

def compute_plain_performance(
                            domain, 
                            domain_split='test', 
                            length=-1, 
                            start=0, 
                            verbose=False,
                            indices=[], 
                            specified_path=None,
                            compute_metric=True):
    dataset = alternative_load_dataset(domain, domain_split)
    result = parse_plain_output(domain, domain_split, specified_path)
    indices = indices if len(indices) > 0 else list(range(start, start + length))
    
    preds = []
    labels = []
    
    for idx in indices:
        if idx not in result.keys(): continue
        
        res = result[idx]
        pred = extract_pred(res)
        label_dialogue = dataset[idx]['dialogue'].split('[NEWLINE]')
        label, label_output = extract_label(label_dialogue, True)
        
        pred, label = align_pred_label(pred, label)
        
        if verbose: print(f"{idx:2d}: [p->l] {pred} -> {label}")
        if sum(pred) != sum(label):
            print(f"ERROR in {idx:4d}: {sum(pred)} != {sum(label)}")
        
        preds.append(pred)
        labels.append(label)
        
    if compute_metric:
        return compute_metrics(preds, labels)
    else:
        return (preds, labels)

def compute_gpt_performance(domain, 
                            domain_split='test', 
                            template='deft', 
                            length=-1, 
                            start=0, 
                            verbose=False, 
                            specified_path=None, 
                            simple=False, 
                            indices=[],
                            specified_result=None,
                            compute_metric=True):
    dataset = alternative_load_dataset(domain, domain_split)
    path = specified_path if specified_path else f'results/{template}_{domain}.json'
    
    if specified_result: result = specified_result
    else: result = parse_output(path) if not simple else parse_simple_output(path)
    
    indices = indices if len(indices) > 0 else list(range(start, start + length))
    
    report = defaultdict(list)
    detail_error = defaultdict(list)
    
    preds = []
    labels = []
    
    for idx in indices:
        if idx not in result.keys(): continue
        
        res = result[idx]
        pred = extract_pred(res['uttr'])
        label_dialogue = dataset[idx]['dialogue'].split('[NEWLINE]')
        label, label_output = extract_label(label_dialogue, True)
        
        pred, label = align_pred_label(pred, label)
            
        for i, (p, l) in enumerate(zip(res['uttr'], label_output)):
            if i == 0: continue
            if len(res['act']) != 0:
                report[res['act'][i]].append(f'{str(l)}-{str(p)}')
            reference = [line for line in label_dialogue if line != "[BOUNDARY]"][i]
            
            
            if p != l:
                detail_err = f"{l:3d} | {reference}"
                if len(res['act']) > 0: detail_err = detail_err.replace('|', f'| {res["act"][i]} |')
                detail_error[idx].append(detail_err)
        
        if verbose: print(f"{idx:2d}: [p->l] {pred} -> {label}")
        if sum(pred) != sum(label):
            print(f"ERROR in {idx:4d}: {sum(pred)} != {sum(label)}")
            # pred = [sum(label)]
        
        preds.append(pred)
        labels.append(label)

    if compute_metric:
        return compute_metrics(preds, labels), report, detail_error
    else:
        return (preds, labels)

def load_dialstart(domain:str, length=-1):
    dialstart_id = [x['label'] for x in load_json(f'dialstart_reproduce/{domain}.json')]
    dialstart_pred = [x['pred'] for x in load_json(f'dialstart_reproduce/{domain}.json')]
    
    if length != -1:
        dialstart_id = dialstart_id[:length]
        dialstart_pred = dialstart_pred[:length]
        
    return dialstart_pred, dialstart_id

def compute_kappa(domain, template, length=50, start=0):
    dataset = alternative_load_dataset(domain, 'test')
    result = parse_output(f'results/{template}_{domain}.json')
    
    model_selection = []
    dataset_selection = []
    
    for idx in range(start, start + length):
        if idx not in result.keys(): continue
        
        pred = [1 if select else 0 for select in result[idx]['uttr']]
        _, label_output = extract_label(dataset[idx]['dialogue'].split('[NEWLINE]'), True)
        label = [1 if select else 0 for select in label_output]
        
        while(len(pred) != len(label)): pred.append(0)
        
        model_selection.extend(pred)
        dataset_selection.extend(label)
        
    return cohen_kappa_score(model_selection, dataset_selection)

def pretty_print(d:dict, precision:int=6):
    for k, v in d.items():
        print(f"[{k}] {round(v, precision)}", end = ' ')
    print()

def parse_report(report):
    new_report = {}
    new_report['all'] = {
            'tp' : 0,
            'fp' : 0,
            'tn' : 0,
            'fn' : 0
        }
    for k, vs in report.items():
        if k == 'all': continue
        new_report[k] = {
            'tp' : 0,
            'fp' : 0,
            'tn' : 0,
            'fn' : 0
        }
        for v in vs:
            l, p = v.split('-')
            if l == 'True' and p == 'True':
                new_report[k]['tp'] += 1
                new_report['all']['tp'] += 1
            if l == 'True' and p == 'False':
                new_report[k]['fn'] += 1
                new_report['all']['fn'] += 1
            if l == 'False' and p == 'True':
                new_report[k]['fp'] += 1
                new_report['all']['fp'] += 1
            if l == 'False' and p == 'False':
                new_report[k]['tn'] += 1
                new_report['all']['tn'] += 1
                
    return new_report

def get_label_sentlen_analysis(detail_error):
    counter = np.zeros((2, 5))
    for errors in detail_error.values():
        for error in errors:
            error_uttr = error.split('agent:')[-1].split('user:')[-1].strip()
            label = int(error.split(' | ')[0])
            cnt = len(sent_tokenize(error_uttr))
            try:
                counter[label][cnt] += 1
            except:
                pass
        
    return counter

def agg_result(domain='tiage', template='deft_test', domain_split='test', length=30, start=0, verbose=False, specified_path=None):
    agg = {}
    path = specified_path if specified_path else f"results/{template}_{domain}.json"
    performance, report, detail_error = compute_gpt_performance(domain, domain_split, template, length=length, start=start, verbose=verbose, specified_path=specified_path)
    agg['performance'] = performance
    agg['report'] = parse_report(report)
    agg['parsed'] = parse_output(path)
    agg['detail_error'] = detail_error
    agg['sentlen'] = get_label_sentlen_analysis(detail_error)
    return agg