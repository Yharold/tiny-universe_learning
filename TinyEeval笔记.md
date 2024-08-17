# TinyEval学习笔记

整个Evaluate项目没什么复杂的，大体分为四个部分：

- 模型构建：载入AI模型，并创建专门的输出方法。实现文件是LLM.py，输出方法是get_pred()
- 度量构建：构建各种评价方法，包括简单的数据处理。实现文件是metrics.py
- 推理：使用模型得到结果。实现文件是inference.py。
- 评估：使用度量方法评估模型效果。实现文件是eval.py。

## 模型构建

模型构建没什么特殊的，重点在其中的get_pred方法。这个方法其实就是一个模型的使用方法。类似与chat,generate等，不过针对输入的数据做了特殊的处理。

```python
import json, torch
from unittest import skip
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from tqdm import tqdm
from peft import PeftModel

class BaseLLM:
    def __init__(self, path: str, model_name: str, adapter_path: str) -> None:
        self.path = path
        self.model_name = model_name
        self.adapter_path = adapter_path

    def build_chat(self, tokenizer, prompt, model_name):
        pass

    def load_model_and_tokenizer(self, path, model_name, device):
        pass

    def post_process(self, response, model_name):
        pass

    def get_pred(
        self,
        data: list,
        max_length: int,
        max_gen: int,
        prompt_format: str,
        deivce,
        out_path: str,
    ):
        pass


class Internlm2Chat(BaseLLM):
    def __init__(self, path: str, model_name: str = "", adapter_path: str = "") -> None:
        super().__init__(path, model_name, adapter_path)

    def build_chat(self, prompt):
        prompt = f"<|im_start|>user\n{prompt}<|im_end>\n<|im_start|>assistant\n"
        return prompt

    def post_process(self, response):
        response = response.split("<|im_end|>")[0]
        return response
    # 载入模型
    def load_model_and_tokenizer(self, path, device, adapter_path):
        # model = AutoModelForCausalLM.from_pretrained(
        #     path, trust_remote_code=True, torch_dtype=torch.bfloat16
        # ).to(device)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).to(
            device
        )
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if adapter_path:
            model = PeftModel.from_pretrained(model, model_id=adapter_path)
        model = model.eval()
        return model, tokenizer
    # 得到预测结果
    def get_pred(self, data, max_length, max_gen, prompt_format, device, out_path):
        # 加载模型
        model, tokenizer = self.load_model_and_tokenizer(
            path=self.path,
            device=device,
            adapter_path=self.adapter_path,
        )
        for json_obj in tqdm(data):
            # 处理prompt
            prompt = prompt_format.format(**json_obj)
            tokenized_prompt = tokenizer(
                prompt, truncation=False, return_tensors="pt"
            ).input_ids[0]
            # 当prompt长度超过max_length,截断prompt，
            # 截为三份，前后加起来是max_length,中间舍弃
            if len(tokenized_prompt) > max_length:
                half = int(max_length / 2)
                prompt = tokenizer.decode(
                    tokenized_prompt[:half], skip_special_tokens=True
                ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            # 添加特殊字符
            prompt = self.build_chat(prompt)
            # 模型的输入
            model_input = tokenizer(prompt, truncation=False, return_tensors="pt").to(
                device
            )
            context_length = model_input.input_ids.shape[-1]
            eos_token_id = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids(["<|im_end|>"])[0],
            ]
            # 使用模型
            output = model.generate(
                **model_input,
                max_new_tokens=max_gen,
                do_sample=False,
                temperature=1.0,
                eos_token_id=eos_token_id,
            )[0]
            # 提取模型输出的内容
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
            pred = self.post_process(pred)
            # 全部信息写入文件
            with open(out_path, "a", encoding="utf-8") as f:
                # json.dump(
                #     {
                #         "pred": pred,
                #         "answers": json_obj["answers"],
                #         "all_classes": json_obj["all_classes"],
                #         "length": json_obj["length"],
                #     },
                #     f,
                #     ensure_ascii=False,
                # )
                json.dump(
                    {
                        "pred": pred,
                        "answers": json_obj["output"],
                        "all_classes": json_obj.get("all_classes", None),
                        "length": json_obj.get("length", None),
                    },
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")


class Qwen2Chat(BaseLLM):
    def __init__(self, path: str, model_name: str = "", adapter_path: str = "") -> None:
        super().__init__(path, model_name, adapter_path)  # 调用父类初始化函数并传入参数

    def build_chat(self, prompt, instruct=None):
        if instruct is None:
            instruct = "You are a helpful assistant."
        prompt = f"<|im_start|>system\n{instruct}<im_end>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        return prompt

    def load_model_and_tokenizer(self, path, device, adapter_path):
        # model = AutoModelForCausalLM.from_pretrained(
        #     path, trust_remote_code=True, torch_dtype=torch.bfloat16
        # ).to(device)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).to(
            device
        )
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        # adapter_path = ''
        if adapter_path:
            model = PeftModel.from_pretrained(model, model_id=adapter_path)
            print(f"adapter loaded in {adapter_path}")
        model = model.eval()
        return model, tokenizer

    def get_pred(self, data, max_length, max_gen, prompt_format, device, out_path):
        # 加载模型
        model, tokenizer = self.load_model_and_tokenizer(
            self.path, device, self.adapter_path
        )
        for json_obj in tqdm(data):
            # 将输入数据转为固定模板
            prompt = prompt_format.format(**json_obj)
            
            tokenized_prompt = tokenizer(
                prompt, truncation=False, return_tensors="pt"
            ).input_ids[0]
            # 在中间截断,因为两头有关键信息.
            if len(tokenized_prompt) > max_length:
                half = int(max_length / 2)
                prompt = tokenizer.decode(
                    tokenized_prompt[:half], skip_special_tokens=True
                ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            # 添加特殊符号，也可以归类到将输入转为固定模板
            prompts = self.build_chat(prompt, json_obj.get("instruction", None))
            inputs = tokenizer(prompts, truncation=False, return_tensors="pt").to(
                device
            )
            # 使用模型预测结果
            output = model.generate(
                inputs.input_ids,
                do_sample=True,
                temperature=1.0,
                max_new_tokens=max_gen,
                top_p=0.8,
            )

            pred = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, output)
            ]
            pred = tokenizer.batch_decode(pred, skip_special_tokens=True)[0]
            # 记录结果
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump(
                    {
                        "pred": pred,
                        "answers": json_obj["output"],
                        "all_classes": json_obj.get("all_classes", None),
                        "length": json_obj.get("length", None),
                    },
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")

```
在get_pred方法中，大体的逻辑可以分为这么几块：

- 将输入数据转为固定模板格式：这一步不仅包括将输入转为固定模板，还包括了数据超出长度的处理方法——中间截断只留两头。
- 使用模型，得到预测结果
- 记录结果


## 度量构建

这一步就是各种评价方法的具体实现了。其中还包括了一些数据的处理，比如最小化，去掉空格和换行等。

```python
import re
import string
import jieba
from rouge import Rouge
from collections import Counter

jieba.setLogLevel(jieba.logging.INFO)


def normalize_zh_aswer(s):

    def white_space_fix(text):
        return "".join(text.split())
    # 移除特殊字符
    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def normalize_en_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def classification_score(prediction, ground_truth, **kwargs):
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        # 这段有点看不懂，什么叫在答案中出现但不是答案？
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        score = 1.0 / len(em_match_list)
    else:
        score = 0.0
    return score


def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]


def rouge_zh_score(prediction, ground_truth, **kwargs):
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    score = rouge_score(prediction, ground_truth)
    return score


def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_en_answer(prediction)
    normalized_ground_truth = normalize_en_answer(ground_truth)
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens_norm = [normalize_zh_aswer(t) for t in prediction_tokens]
    ground_truth_tokens_norm = [normalize_zh_aswer(t) for t in ground_truth_tokens]
    prediction_tokens = [t for t in prediction_tokens_norm if len(t) > 0]
    ground_truth_tokens = [t for t in ground_truth_tokens_norm if len(t) > 0]
    return f1_score(prediction_tokens, ground_truth_tokens)


def GAOKAO_math(prediction, ground_truth, **kwargs):
    score = 0
    if len(ground_truth) > 1:
        pattern = r"[A-D]"
        matches = re.findall(pattern, prediction)
        prediction_answer = ""
        if matches:
            reversed_prediction = prediction[::-1]
            if len(matches) > 1:
                for i, match in enumerate(matches):
                    if i == 0:
                        prediction_answer += match
                    else:
                        distance = (
                            reversed_prediction.find(matches[i - 1])
                            - reversed_prediction.find(match)
                            - 1
                        )
                        if distance > 5:
                            break
                        prediction_answer += match
                prediction_answer = "".join(sorted(set(prediction_answer)))
            # 全选对了
            if prediction_answer == ground_truth:
                score = 1
            # 选对了一部分
            elif all(option in ground_truth for option in prediction_answer) and len(
                prediction_answer
            ) < len(ground_truth):
                score = 0.5
    else:
        pattern = r"[A-D]"
        matches = re.findall(pattern, prediction)
        if matches and matches[-1] == ground_truth:
            score = 1
    return score
```

## 推理

推理的代码逻辑很简单，没什么说的。

首先是设置了随机数种子，这样每次的随机数都是同一个，代码结果可以复现。

然后加载一些数据，比如提示词模板，模型路径，adapter路径，数据最大长度等。它将这些数据写为了json文件，放在了config文件夹下。

接着加载数据，数据在dataset文件夹下。

最后使用模型的get_pred方法的到结果。

这一步的代码如下：

```python
import argparse
from datasets import load_dataset
import json
import os, sys
import random
from LLM import Qwen2Chat, Internlm2Chat
import torch
import torch.backends.cudnn


# 设置随机种子，控制随机数能重复出现
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


# 注入参数
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, default="Qwen2")
    parser.add_argument("--model", type=str, default="internlm2")
    return parser.parse_args(args)


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    # 加载配置
    model2path = json.load(
        open("./content/TinyEval/m_data/config/model2path.json", "r")
    )
    model2maxlen = json.load(
        open("./content/TinyEval/m_data/config/model2maxlen.json", "r")
    )
    adapter2path = json.load(open("./content/TinyEval/m_data/config/adapter2path.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model
    max_length = model2maxlen[model_name]
    datasets = ["GAOKAO_math"]
    dataset2prompt = json.load(
        open(
            "./content/TinyEval/m_data/config/dataset2prompt.json",
            "r",
            encoding="utf-8",
        )
    )
    dataset2maxlen = json.load(
        open("./content/TinyEval/m_data/config/dataset2maxlen.json", "r")
    )
    # 加载模型
    # pred_model = Qwen2Chat(model2path[model_name], model_name, adapter2path[model_name])
    pred_model = Internlm2Chat(
        model2path[model_name], model_name, adapter2path[model_name]
    )
    # 创建输出文件夹
    if not os.path.exists("./content/TinyEval/m_data/pred"):
        os.makedirs("./content/TinyEval/m_data/pred")

    for dataset in datasets:
        data = load_dataset(
            "json",
            data_files=f"./content/TinyEval/m_data/dataset/{dataset}.jsonl",
            split="train",
        )

        if not os.path.exists(f"./content/TinyEval/m_data/pred/{model_name}"):
            os.makedirs(f"./content/TinyEval/m_data/pred/{model_name}")
        out_path = f"./content/TinyEval/m_data/pred/{model_name}/{dataset}.jsonl"
        if os.path.isfile(out_path):
            os.remove(out_path)
        prompt_format = dataset2prompt.get(dataset, dataset2prompt.get("custom_zh"))
        max_gen = dataset2maxlen.get(dataset, dataset2maxlen.get("custom_zh"))
        data_all = [data_sample for data_sample in data]
        pred_model.get_pred(data, max_length, max_gen, prompt_format, device, out_path)
    print("")

```

## 评估

评估就更不用说了，逻辑还是一样的，加载数据。然后调用metrics中的评估方法。最后得到评估分数。

```python
import os
import json
import argparse
import numpy as np

from metrics import (
    classification_score,
    qa_f1_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    rouge_zh_score,
    GAOKAO_math,
)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen2")
    return parser.parse_args(args)


dataset2metric = {
    "multifieldqa_zh": qa_f1_zh_score,
    "multi_news": rouge_score,
    "trec": classification_score,
    "custom_zh": rouge_zh_score,
    "GAOKAO_math": GAOKAO_math,
}


def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        if dataset in ["trec"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        if dataset in ["custom_zh", "custom_en"]:
            score = max(
                score,
                dataset2metric[dataset](
                    prediction, ground_truths, all_classes=all_classes
                ),
            )
        else:
            score = max(
                score,
                dataset2metric.get(dataset, dataset2metric[dataset])(
                    prediction, ground_truths, all_classes=all_classes
                ),
            )
        total_score += score
    return round(100 * total_score / len(predictions), 2)


if __name__ == "__main__":
    scores = dict()
    args = parse_args()
    path = f"./content/TinyEval/m_data/pred/{args.model}/"
    all_files = os.listdir(path)
    print("Evalutating on:", all_files)
    for file in all_files:
        if not file.endswith(".jsonl") or file == "result.json":
            continue
        predictions, answers, lengths = [], [], []
        dataset = file.split(".")[0]
        with open(f"{path}{file}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
            score = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score
    out_path = f"./content/TinyEval/m_data/pred/{args.model}/result.json"
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
```

## 总结

整个模型的设计还是很简单的。复杂的反倒是各个标准的指定。比如所有的数据如何处理？保存为什么样的格式？不同的评价方法的输出不同，是否可以创建同样的格式？如果可以那么这个格式如何设计？不可以的话怎么解决？是每个方法专门设计一套规则吗？

技术的实现其实比较统一，标准的实现有点难以抉择。







