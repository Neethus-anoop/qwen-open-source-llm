# Qwen Open-Source LLM Project

A local AI project using **Qwen 2.5 0.5B Instruct**, Ollama, FastAPI, and QLoRA fine-tuning.

## Features

- Local Qwen 2.5 0.5B inference using Ollama
- FastAPI inference API
- `/generate` endpoint
- QLoRA fine-tuning with PEFT and TRL
- 4-bit NF4 quantization
- LoRA adapter training
- GPU-based training on NVIDIA Tesla T4

## Project Structure

```text
qwen-open-source-llm/
├── app/
│   └── main.py
├── data/
│   └── qlora_dataset.json
├── src/
│   └── train_qlora.py
├── docs/
├── notebooks/
├── requirements.txt
├── README.md
└── .gitignore

&#x20; v

Generated Response

