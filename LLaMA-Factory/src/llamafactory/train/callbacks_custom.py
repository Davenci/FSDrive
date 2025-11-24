# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from transformers import TrainerCallback
from typing_extensions import override
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments


class GenerationCallback(TrainerCallback):
    """在训练过程中定期打印模型生成的预测样例"""

    def __init__(self, tokenizer, generate_every_n_steps=100, num_samples=3):
        """
        Args:
            tokenizer: 用于解码的tokenizer
            generate_every_n_steps: 每隔多少步生成一次样例
            num_samples: 每次生成多少个样例
        """
        self.tokenizer = tokenizer
        self.generate_every_n_steps = generate_every_n_steps
        self.num_samples = num_samples

    @override
    def on_log(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        # 每隔指定步数生成样例
        if state.global_step % self.generate_every_n_steps == 0 and state.global_step > 0:
            model = kwargs.get("model")
            train_dataloader = kwargs.get("train_dataloader")

            if model is None or train_dataloader is None:
                return

            # 获取一个batch的数据
            try:
                batch = next(iter(train_dataloader))
                # 移动到正确的设备
                device = model.device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # 只取前几个样例
                num_samples = min(self.num_samples, batch["input_ids"].size(0))
                input_ids = batch["input_ids"][:num_samples]

                # 打印分隔线
                print("\n" + "="*80)
                print(f"🔍 Generation Samples at Step {state.global_step}")
                print("="*80)

                model.eval()
                with torch.no_grad():
                    # 生成预测
                    generated = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=100,  # 最多生成100个新token
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                # 解码并打印
                for i in range(num_samples):
                    # 输入
                    input_text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    # 完整生成（包括输入）
                    full_text = self.tokenizer.decode(generated[i], skip_special_tokens=True)
                    # 只提取新生成的部分
                    generated_text = full_text[len(input_text):].strip()

                    # 如果有标签，也打印出来
                    if "labels" in batch:
                        labels = batch["labels"][i]
                        # 过滤掉 IGNORE_INDEX (-100)
                        labels = labels[labels != -100]
                        label_text = self.tokenizer.decode(labels, skip_special_tokens=True)

                        print(f"\n📝 Sample {i+1}:")
                        print(f"  Input:     {input_text[:200]}...")  # 截断显示
                        print(f"  Predicted: {generated_text[:200]}...")
                        print(f"  Label:     {label_text[:200]}...")
                    else:
                        print(f"\n📝 Sample {i+1}:")
                        print(f"  Input:     {input_text[:200]}...")
                        print(f"  Generated: {generated_text[:200]}...")

                print("="*80 + "\n")
                model.train()

            except Exception as e:
                print(f"⚠️ Error during generation: {e}")
                return
