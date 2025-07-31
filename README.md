
#  Medical Question Answering using Fine-Tuned DeepSeek R1

This project focuses on using a fine-tuned version of the DeepSeek R1 language model for answering medical-related queries. The model is designed to interpret medical questions and provide informed responses based on fine-tuned domain-specific data.

##  Project Overview

* **Model Used**: [`DeepSeek R1`](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat)
* **Fine-Tuning Objective**: Customize DeepSeek R1 to better respond to healthcare-specific queries.
* **Training Platform**: Google Colab
* **Model Storage**: Saved and stored on Google Drive for portability and future use.

##  Notebook and Results

* The full notebook including preprocessing, fine-tuning, and inference can be viewed here:
   [Google Colab Notebook](https://colab.research.google.com/drive/19dm0b5fTdzA5FugzdRNO0GC_ezWg19a3#scrollTo=p3p-Cgzk8-te)

* Download the fine-tuned model:
   [Google Drive Link](https://drive.google.com/file/d/1Xvmi_sRtL7cMeZojqxtrpfts2mWjZdqF/view?usp=drive_link)

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Sample of key dependencies:

* `transformers==4.54.1`
* `torch==2.7.1`
* `streamlit==1.47.1`
* `accelerate==1.9.0`

*(Full list available in `requirements.txt`)*

##  Inference Example (No UI)

Here's a minimal example of how to load and query the fine-tuned model:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

model_path = os.path.abspath("./model")  # point to your local model folder

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

question = "What are common symptoms of pneumonia?"
prompt = f"<|user|>\n{question}\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

##  Future Work

* [ ] Build a **Streamlit-based UI** to allow interactive querying.
* [ ] Deploy the app via Hugging Face Spaces or a similar platform.
* [ ] Add evaluation scripts and benchmarks.

##  Folder Structure

```
.
├── model/                     # Fine-tuned DeepSeek R1 model
├── notebook.ipynb            # Colab notebook for training/inference
├── requirements.txt          # Required Python packages
└── README.md                 # Project documentation
```

##  License

This project is distributed under the MIT License.

---

