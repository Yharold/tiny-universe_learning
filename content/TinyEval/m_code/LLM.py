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
        model, tokenizer = self.load_model_and_tokenizer(
            self.path, device, self.adapter_path
        )
        for json_obj in tqdm(data):
            prompt = prompt_format.format(**json_obj)
            # 在中间截断,因为两头有关键信息.
            tokenized_prompt = tokenizer(
                prompt, truncation=False, return_tensors="pt"
            ).input_ids[0]
            if len(tokenized_prompt) > max_length:
                half = int(max_length / 2)
                prompt = tokenizer.decode(
                    tokenized_prompt[:half], skip_special_tokens=True
                ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

            prompts = self.build_chat(prompt, json_obj.get("instruction", None))
            inputs = tokenizer(prompts, truncation=False, return_tensors="pt").to(
                device
            )

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
