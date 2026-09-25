\# Qwen Open-Source LLM Project



A hands-on open-source LLM project using \*\*Qwen 2.5, Ollama, FastAPI, Hugging Face Transformers, PEFT, and QLoRA\*\*.



\## Project Overview



This project demonstrates:



\- Local Qwen 2.5 inference using Ollama

\- FastAPI-based inference API

\- Swagger API documentation

\- Hugging Face Transformers integration

\- PEFT/LoRA configuration

\- QLoRA 4-bit quantization training setup

\- GPU-ready fine-tuning workflow



\## Architecture



```text

Client

&#x20; |

&#x20; v

FastAPI

&#x20; |

&#x20; v

Ollama

&#x20; |

&#x20; v

Qwen 2.5 0.5B

&#x20; |

&#x20; v

Generated Response

