

---
# Medical Question Answering using Fine-Tuned DeepSeek R1

This project leverages a fine-tuned version of **DeepSeek R1 (8B)** for clinical reasoning and medical question answering. Fine-tuning was performed using **Unsloth** with **LoRA adapters**, allowing for fast, memory-efficient training on domain-specific data.

---

## 📌 Project Overview

* **Base Model**: [`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)
* **Fine-Tuning Framework**: [Unsloth](https://github.com/unslothai/unsloth) (LoRA + PEFT + TRL)
* **Dataset Used**: [`FreedomIntelligence/medical-o1-reasoning-SFT`](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
* **Training Environment**: Google Colab (CUDA-enabled)
* **Model Checkpoints**: [Google Drive](https://drive.google.com/file/d/1Xvmi_sRtL7cMeZojqxtrpfts2mWjZdqF/view?usp=drive_link)

---

##  Fine-Tuning Summary

* **Prompt Template**: Chain-of-Thought enabled system prompts tailored for expert-level diagnostic reasoning.
* **Precision**: Mixed FP16/bfloat16 with memory-efficient 4-bit quantization.
* **LoRA Configuration**:

  * Target Modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
  * Rank: `16`, Alpha: `16`, Dropout: `0`
* **Trainer**: `SFTTrainer` from HuggingFace TRL
* **Max Steps**: 60 (early-stop for controlled training), Batch Size: 2 × 4 accumulation
* **Evaluation**: Manual inference + Prompt-based generation (example below)

---

##  Installation

```bash
pip install -r requirements.txt
```

Key dependencies include:

* `transformers==4.54.1`
* `torch==2.7.1`
* `streamlit==1.47.1`
* `trl==0.14.0`, `peft==0.14.0`, `unsloth`
* `xformers`, `accelerate==1.9.0`

---

##  Inference Example (Script Mode)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, os

model_path = os.path.abspath("./model")  # local model path

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

question = "What are common symptoms of pneumonia?"
prompt = f"<|user|>\n{question}\n<|assistant|>\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

##  Folder Structure

```
.
├── model/                     # Fine-tuned DeepSeek R1 model
├── notebook.ipynb            # Colab notebook with training pipeline
├── requirements.txt          # Python package dependencies
└── README.md                 # Project documentation
```

---

##  Sample Prompt Template (Used for Fine-Tuning)

```
Below is a task description along with additional context provided in the input section. Your goal is to provide a well-reasoned response that effectively addresses the request.

Before crafting your answer, take a moment to carefully analyze the question. Develop a clear, step-by-step thought process to ensure your response is both logical and accurate.

### Instruction:
You are a medical expert specializing in clinical reasoning, diagnostics, and treatment planning. Answer the medical question below using your advanced knowledge.

### Question:
{Medical Question}

### Response:
<think>
{Chain-of-Thought Reasoning}
<think>
```

---

##  Future Work

* [ ] Add automatic evaluation pipeline with accuracy metrics.
* [ ] Build a **Streamlit UI** for interactive Q\&A.
* [ ] Deploy via **Hugging Face Spaces** or FastAPI backend.
* [ ] Integrate feedback loop for continuous fine-tuning.

---

##  License

This project is released under the MIT License.

---

