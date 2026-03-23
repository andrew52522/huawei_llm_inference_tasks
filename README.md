### Huawei LLM Inference Tasks
This repository contains the solutions for the Huawei LMM Inference Optimization technical assignment. The project is divided into two main parts: a framework analysis and a custom discrete-event simulator for multi-modal inference.

#### Task 1: SGLang Framework Analysis
I have thoroughly read the SGLang paper (RadixAttention and frontend language optimizations) and prepared a structured summary.

To go beyond just theory, I have installed the SGLang framework locally and successfully ran several inference examples. I've explored how it handles structured generation and state sharing. I am ready to discuss the paper's core concepts, my hands-on experience, and show the code examples during the technical interview.

#### Task 2: Multi-Modal LLM Inference Simulator
The core task was to simulate a server handling multi-modal requests (text + images) based on Azure traces, and find the minimum number of GPUs (N) required without violating strict SLA limits (TTFT <= 10s, Decode <= 1s).

The Evolution (How it started vs. How it's going)
As you can see in the commit history, the solution evolved significantly:

Early Stages: I started with a naive "greedy" MVP approach, trying to process requests on a single massive GPU without proper batching logic or memory constraints.

Final Architecture: The project was entirely refactored into a robust Discrete-Event Simulator that mimics the physics of modern inference engines (like vLLM or SGlang).

Implemented Technologies
To meet the SLA under heavy load, the simulator now implements the "Big 4" LLM serving patterns:

Continuous Batching & Admission Control: Dynamically injecting new requests into the active batch as soon as VRAM frees up.

Chunked Prefill & Iteration-Level Scheduling: Splitting heavy context/image preprocessing into chunks to prevent new requests from blocking the fast token generation (decode) phase.

Dynamic KV-Cache Management: Strict tracking of MEM_X, MEM_Y, and MEM_Z to allocate and release memory instantly upon generation as soon as finished decoding

Eviction Policy: A safety mechanism against OOM (Out-of-Memory). If VRAM runs out, the scheduler evicts the request with the Longest Remaining Time First back to the beggining the deque, saving the rest of the batch.

#### Repository Structure

sglang_summary_task1.pdf: my summary of paper Sglang

Sglang_key_features_and_quantization_qwens.ipynb: Showing how fast SGlang, demostrated FSM, RadixTree + KV_cache, and afterall I quantizaied qwen2.5-36B-instruct and qwen2.5-7b-instruct both to float8

Sglang_key_features_and_quantization_qwens.pdf : pdf file, because .ipynd isn`t show in github web 

task_huawei.py: The core simulation engine. It contains the Request, Accelerator, and Event classes. The scheduler_step has been refactored into clean, typed, and fully documented private functions.

min_gpu_for_azure_logs2.ipynb: The analytics and research pipeline. For algorithm Chunked Prefill and Continious Batching I get by GridSearch optimum parametrs.

reluts.pdf: only results from .ipynd file

Extracts the peak load window (e.g., Day 2, 16:00-17:30) to stress-test the system.

Runs a 3D Grid Search (N GPUs x Chunk Size x Batch Size) to empirically find the minimal cluster size.

Calculates the Theoretical Bounds (Best Case vs. Worst Case processing times) to validate the efficiency of the Continuous Batching algorithm.
