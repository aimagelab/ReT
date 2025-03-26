<div align="center">
  <h1>ReT: Recurrence-Enhanced Vision-and-Language Transformers for Robust Multimodal Document Retrieval (CVPR 2025) </h1>
</div>

<div align="center">
  
  [![Paper](https://img.shields.io/badge/Paper-arxiv.2503.15621-B31B1B.svg)](https://www.arxiv.org/abs/2503.01980)
  [![ReT](https://img.shields.io/badge/Checkpoints-🤗%20ReT-blue)](https://huggingface.co/collections/aimagelab/ret-67e15d4f9c60664d08ff8747)
  [![Dataset](https://img.shields.io/badge/Dataset-🤗%20ReT--M2KR-blue)](https://huggingface.co/datasets/aimagelab/ReT-M2KR)


</div>

<p align="center">
  <img src="assets/model.png" alt="ReT" width="840" />
</p> 

Please cite with the following BibTeX:
```
@inproceedings{caffagni2025recurrence,
  title={{Recurrence-Enhanced Vision-and-Language Transformers for Robust Multimodal Document Retrieval}},
  author={Caffagni, Davide and Sarto, Sara and Cornia, Marcella and Baraldi, Lorenzo and Cucchiara, Rita},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```
## Installation
1. Create the Python environment.
```
conda create -n ret -y --no-default-packages python==3.10.16
conda activate ret
```
2. Install Pytorch.
```
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
```
3. Install faiss-gpu.
```
conda install -n ret -y -c conda-forge faiss-gpu==1.7.4
```
4. Clone the repo and install other dependencies.
```
git clone https://github.com/aimagelab/ReT.git
cd ReT
pip install -r requirements.txt
```


## Pre-trained models 🤗
ReT model checkpoints are available on [**Hugging Face**](https://huggingface.co/collections/aimagelab/ret-67e15d4f9c60664d08ff8747).
You can use these checkpoints directly for retrieval tasks or fine-tune them to suit your specific retrieval needs.

### Available Checkpoints and Benchmark Results
| Model         | WIT Recall@10 | IGLUE Recall@1 | KVQA Recall@5 | OVEN Recall@5 | LLaVA Recall@1 | InfoSeek Recall@5 | InfoSeek Pseudo Recall@5 | EVQA Recall@5 | EVQA Pseudo Recall@5 | OKVQA Recall@5 | OKVQA Pseudo Recall@5 |
|---------------|---------------|----------------|------------------|---------------|----------------|---------------|----------------------|----------------|-----------------------|-------------------|--------------------------|
| [ReT-CLIP-ViT-L-14🤗](https://huggingface.co/aimagelab/ReT-CLIP-ViT-L-14) | 0.734         | 0.818          | 0.635         | 0.820            | 0.799         | 0.470          | 0.605         | 0.445                | 0.579          | 0.202                 | 0.662             |
| [ReT-OpenCLIP-ViT-H-14🤗](https://huggingface.co/aimagelab/ReT-OpenCLIP-ViT-H-14) | 0.714         | 0.800          | 0.593         | 0.830            | 0.798         | 0.473          | 0.607         | 0.448                | 0.578          | 0.182                 | 0.634             |
| [ReT-OpenCLIP-ViT-G-14🤗](https://huggingface.co/aimagelab/ReT-OpenCLIP-ViT-G-14) | 0.751         | 0.822          | 0.606         | 0.840            | 0.792         | 0.520          | 0.625         | 0.486                | 0.602          | 0.190                 | 0.638             |


## ReT-M2KR Dataset 🤗

You can download the ReT-M2KR benchmark by following the instructions provided [here](https://huggingface.co/datasets/aimagelab/ReT-M2KR). 
This dataset is used for training and evaluating ReT in multimodal information retrieval and includes images (coming soon) and `JSONL` files.

ReT-M2KR benchmark is an extended version of the [M2KR dataset](https://huggingface.co/datasets/BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR), with the following modifications:

- MSMARCO data is excluded, as it does not contain query images
- Passage images have been added to the OVEN, InfoSeek, E-VQA, and OKVQA datasets

For further details, please refer to the associated research paper.



## Use with Transformers
```python
from src.models import RetrieverModel, RetModel
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
retriever = RetrieverModel.from_pretrained('aimagelab/ReT-CLIP-ViT-L-14', device_map=device)

# QUERY
ret: RetModel = retriever.get_query_model()
ret.init_tokenizer_and_image_processor()
q_txt = "Retrieve documents that provide an answer to the question alongside the image: What is the content of the image?"
q_img = 'assets/model.png'

ret_feats = ret.get_ret_features([[q_txt, q_img]])
print(ret_feats.shape)  # torch.Size([1, 32, 128])


# PASSAGE
ret: RetModel = retriever.get_passage_model()
ret.init_tokenizer_and_image_processor()

p_txt = """The image shows a diagram of what appears to be a neural network architecture using a fine-grained loss approach for multimodal learning.
The architecture has two parallel processing streams labeled "ReTQ" (left side, in purple) and "ReTD" (right side, in blue).
Each side has: ..."""
p_img = ''

ret_feats = ret.get_ret_features([[p_txt, p_img]])
print(ret_feats.shape)  # torch.Size([1, 32, 128])
```

## Indexing and Searching
To evaluate ReT on the on M2KR benchmark, we provide SLURM script examples [here](./scripts). These scripts handle both indexing and searching processes.

Make sure to set [`JSONL_ROOT_PATH`](https://github.com/aimagelab/ReT/blob/88abe2461106b07a047d57ccba32b7d2af52e3e1/scripts/inference_m2kr_large.sh#L37) and [`IMAGE_ROOT_PATH`](https://github.com/aimagelab/ReT/blob/88abe2461106b07a047d57ccba32b7d2af52e3e1/scripts/inference_m2kr_large.sh#L60) accordingly to the directories where the JSONL files and images have been downloaded.

#### Known issue
If the inference script got stuck while indexing, try to clear the Pytorch cache and re-run
```
rm -rf ~/.cache/torch_extensions  
```





## Acknowledgments
We thank the teams behind [ColBERT](https://github.com/stanford-futuredata/ColBERT), [PreFLMR](https://github.com/LinWeizheDragon/FLMR), and [UniIR](https://github.com/TIGER-AI-Lab/UniIR) for open-sourcing their models, datasets, and code.
