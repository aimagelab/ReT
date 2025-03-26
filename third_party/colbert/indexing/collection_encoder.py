import torch

from third_party.colbert.infra.run import Run
from third_party.colbert.utils.utils import print_message, batch


class CollectionEncoder:
    def __init__(self, config, checkpoint):
        self.config = config
        self.checkpoint = checkpoint
        self.use_gpu = self.config.total_visible_gpus > 0

    def encode_passages(self, passages):
        Run().print(f"#> Encoding {len(passages)} passages..")

        if len(passages) == 0:
            return None, None

        with torch.inference_mode():
            embs, doclens = [], []

            # Batch here to avoid OOM from storing intermediate embeddings on GPU.
            # Storing on the GPU helps with speed of masking, etc.
            # But ideally this batching happens internally inside docFromText.
            for passage_batch in batch(passages, self.config.index_bsize):
                embs_, doclens_ = self.checkpoint.docFromText(passage_batch)
                embs.append(embs_)
                doclens.extend(doclens_)

            embs = torch.cat(embs)

        return embs, doclens
