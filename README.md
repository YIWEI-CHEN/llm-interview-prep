# LLM Interview Prep

A deep-dive, hands-on repository for preparing technical interviews and advancing expertise in **LLM infrastructure, fine-tuning, evaluation, and reasoning systems**.

This repo curates **notebooks, markdown explainers, and architecture diagrams** that go beyond surface-level tutorials — focusing on **how modern large-scale language model systems are built, optimized, and evaluated** in real production environments.

---

## Core Focus Areas

### 1. Design and Build Scalable LLM Infrastructure
- **Inference frameworks**: [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang)
- **Serving optimization**: continuous batching, paged attention, KV cache sharing, CUDA graph optimization  
- **Runtime parallelism**: tensor, pipeline, and model parallelism with DeepSpeed / Accelerate  
- **Memory and throughput trade-offs** in multi-GPU deployments  

### 2. Fine-Tune and Adapt Models Efficiently
- **Parameter-efficient methods**: LoRA, QLoRA, PEFT  
- **Frameworks**: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [Hugging Face Accelerate](https://github.com/huggingface/accelerate), [Transformers](https://github.com/huggingface/transformers)  
- **Fine-tuning scenarios**: instruction tuning, reward modeling, preference optimization (PPO, DPO, GRPO)  
- **Evaluation-aware training** — connecting loss metrics to downstream reasoning and retrieval performance  

### 3. Develop Experimentation and Evaluation Frameworks
- **OpenAI’s [simple-eval](https://github.com/openai/simple-eval)** for lightweight, reproducible LLM evaluation  
- Pairwise and listwise judgment pipelines for model comparison  
- Benchmarking via MMLU, GSM8K, BBH, and custom domain datasets  
- Statistical analysis for few-shot and reasoning evaluations  

### 4. Curate and Analyze Datasets
- Data acquisition and schema validation for fine-tuning  
- Synthetic data generation and filtering with LLMs  
- Dataset lineage and documentation best practices  

### 5. Innovate in Reasoning, Retrieval, and Ranking
- **LLM reasoning paradigms**: Chain-of-Thought (CoT), Tree-of-Thought (ToT), ReAct, and self-consistency  
- **Retrieval & ranking systems**: vector search (FAISS, Milvus), dense vs hybrid retrieval  
- **Evaluation metrics**: relevance, coherence, factuality, and reasoning depth  

---

## Example Topics & Questions

| Category | Example Interview Question |
|-----------|-----------------------------|
| **Inference Infra** | How does vLLM’s paged attention improve memory utilization compared to standard transformer caching? |
| **Serving Systems** | What’s the difference between SGLang’s dynamic batching and vLLM’s continuous batching? |
| **Fine-Tuning** | Explain how LoRA adapts model weights without updating full parameters. |
| **Evaluation** | How can OpenAI’s simple-eval help ensure model-to-model comparison fairness? |
| **Reasoning** | Contrast Chain-of-Thought vs Tree-of-Thought for multi-step reasoning tasks. |

---

## Tools & Frameworks

- **Training / Fine-Tuning:** PyTorch, LLaMA-Factory, Accelerate, PEFT, Transformers  
- **Inference / Serving:** vLLM, SGLang, Triton  
- **Evaluation:** simple-eval, OpenAI Evals, NumPy, Pandas, SciPy  
- **Data:** Hugging Face Datasets, JSONL, Parquet  

---

## References

- **Frameworks:** vLLM, SGLang, LLaMA-Factory, Accelerate, Transformers  
- **Evaluations:** OpenAI *simple-eval*, DeepSeek reasoning tests, MMLU  
- **Key Papers:** LoRA, QLoRA, PEFT, DPO, GRPO, Chain-of-Thought, ReAct  

---

## License
MIT License — for educational and interview-preparation use.

---

## Author
**Yi-Wei Chen**  
Senior Software Engineer | AI Infrastructure & Applied ML  
🚀 Focused on building scalable and efficient LLM systems for agents and evaluation.
