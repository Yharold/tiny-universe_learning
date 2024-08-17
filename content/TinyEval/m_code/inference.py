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
    sys.path.append("E:/Code/tiny-universe/")
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
