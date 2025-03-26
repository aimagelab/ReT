from collections import defaultdict, OrderedDict
import os
import json
import numpy as np
from tqdm import tqdm
import torch

from third_party.colbert.infra import Run, RunConfig, ColBERTConfig
from third_party.colbert import Indexer
from third_party.colbert.data import Queries
from third_party.colbert import Searcher

from transformers import set_seed
from argparse import ArgumentParser
import pandas as pd
from pathlib import Path
import src.utils as utils

from third_party.colbert.infra.config.core_config import DefaultVal
from third_party.colbert.utils.utils import timestamp

logger = utils.get_logger()


def parse_command_line():
    parser = ArgumentParser()
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--dataset_passages_path', type=str)
    parser.add_argument('--image_root_path', type=str)
    parser.add_argument('--n_samples', type=int)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--recall_results_filename',
                        type=str, default='recall_results')

    parser.add_argument('--action', type=str, default='index')
    parser.add_argument('--index_name', type=str, default='latent')
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--index_bsize', type=int, default=4)
    parser.add_argument('--num_docs_to_retrieve', type=int, default=100)
    parser.add_argument('--root_path', type=str,
                        default='scratch/latent_index')
    parser.add_argument('--ranking_name', type=str, default='ranking')

    return parser.parse_args()


if __name__ == '__main__':
    set_seed(42)
    args = parse_command_line()

    tqdm.pandas()

    data = pd.read_json(args.dataset_path, lines=True)

    logger.info(
        f"Loaded dataset with {len(data)} samples from {args.dataset_path}")
    if args.n_samples:
        data = data.head(args.n_samples)
        logger.info(f"Using {args.n_samples} samples for debugging")

    IMG_ROOT = Path(args.image_root_path)

    search_run_name = DefaultVal(timestamp(daydir=True))
    search_run_name = os.path.join(
        *[f"{args.index_name}__{x}" for x in search_run_name.val.split(os.sep)])

    run_config = RunConfig(
        root=args.root_path,
        experiment=args.experiment_name,
        name=search_run_name,
        nranks=torch.cuda.device_count()
    )

    with Run().context(run_config):
        checks = "colbert-ir/colbertv2.0"

        config = ColBERTConfig(
            nbits=8,
            checkpoint_path=args.checkpoint_path,
            index_bsize=args.index_bsize,
            root=args.root_path,
            index_name=args.index_name
        )

        if args.action == 'index':
            indexer = Indexer(checkpoint=checks, config=config)

            def get_ret_inputs(row):
                txt = '' if pd.isna(row.passage_text) else row.passage_text
                img = None if pd.isna(row.passage_image_path) else str(IMG_ROOT.joinpath(Path(row.passage_image_path)))
                return txt, img

            logger.info('Preprocessing data ...')
            collection = data.progress_apply(
                get_ret_inputs, axis=1).tolist()
            logger.info('Preprocessing data ... Done')

            indexer.index(name=args.index_name, collection=collection,  overwrite=True)

        elif args.action == 'search':
            searcher = Searcher(index=args.index_name, config=config)

            compute_pseudo_recall = False if 'answer' not in data else any(
                ~ data.answer.isna())
            
            def get_ret_inputs(row):
                txt = row.instruction if pd.isna(row.question) else f"{row.instruction} {row.question}"
                img = None if pd.isna(row.image_path) else str(IMG_ROOT.joinpath(Path(row.image_path)))
                return row.data_id, txt, img            

            logger.info('Preprocessing queries ...')
            queries = OrderedDict((x[0], x[1:]) for x in data.progress_apply(get_ret_inputs, axis=1))
            queries = Queries(data=queries)
            logger.info('Preprocessing queries ... Done')

            ranking = searcher.search_all(
                queries, k=args.num_docs_to_retrieve, bsize=config.index_bsize)
            ranking.save(f"{args.ranking_name}.tsv")

            ranking_dict = ranking.todict()

            passages_data = pd.read_json(
                args.dataset_passages_path, lines=True)
            logger.info(
                f"Loaded passages dataset with {len(passages_data)} samples from {args.dataset_passages_path}")

            Ks = [1, 2, 3, 5, 10, 50, 100]
            if args.num_docs_to_retrieve not in Ks and args.num_docs_to_retrieve < Ks[-1]:
                Ks.append(args.num_docs_to_retrieve)
                Ks.sort()

            # Process ranking data and obtain recall scores
            # Psuedo Recall@K to be computed by matching the answer in the retrieved documents
            # Positive ids Recall@K to be computed by matching the sample positive id with the retrieved documents ids
            recall_dict = defaultdict(list)
            result_dict = defaultdict(list)
            for i, (question_id, pos_ids, answers) in enumerate(zip(data.data_id, data.passage_id, data.answer)):
                retrieved_docs = ranking_dict[question_id]
                retrieved_doc_scores = [doc[2] for doc in retrieved_docs]
                retrieved_docs = [doc[0] for doc in retrieved_docs]
                retrieved_doc_texts = [
                    passages_data.iloc[doc_idx].passage_text for doc_idx in retrieved_docs]
                retrieved_doc_ids = [
                    passages_data.iloc[doc_idx].passage_id for doc_idx in retrieved_docs]
                retrieved_doc_list = [
                    {
                        "passage_id": doc_id,
                        "score": score,
                    } for doc_id, score in zip(retrieved_doc_ids, retrieved_doc_scores)
                ]
                result_dict["retrieved_passage"].append(retrieved_doc_list)

                if compute_pseudo_recall:
                    # Psuedo Recall@K
                    hit_list = []
                    # Get answers
                    for retrieved_doc_text in retrieved_doc_texts:
                        found = False
                        for answer in answers:
                            safe_answer = answer
                            if isinstance(safe_answer, dict):
                                safe_answer = str(safe_answer['wikidata'])
                            if safe_answer.strip().lower() in retrieved_doc_text.lower():
                                found = True
                        if found:
                            hit_list.append(1)
                        else:
                            hit_list.append(0)

                    # print(hit_list)
                    # input()
                    for K in Ks:
                        recall = float(np.max(np.array(hit_list[:K])))
                        recall_dict[f"Pseudo Recall@{K}"].append(recall)

                # Positive ids Recall@K
                # retrieved_doc_ids = [passage_ids[doc_idx] for doc_idx in retrieved_docs]
                hit_list = []
                for retrieved_doc_id in retrieved_doc_ids:
                    if not isinstance(pos_ids, list):
                        pos_ids = [pos_ids]
                    found = False
                    for pos_id in pos_ids:
                        if str(pos_id) == str(retrieved_doc_id):
                            found = True
                    if found:
                        hit_list.append(1)
                    else:
                        hit_list.append(0)
                for K in Ks:
                    recall = float(np.max(np.array(hit_list[:K])))
                    recall_dict[f"Recall@{K}"].append(recall)

            recall_dict = {k: np.mean(v) for k, v in recall_dict.items()}

            # Get the maximum length for each column (header or value) for proper alignment
            column_widths = {key: max(len(key), len(f"{value:.8f}"))
                             for key, value in recall_dict.items()}

            # Print the headers (the keys as columns)
            header_row = "| " + \
                " | ".join(
                    [f"{key:<{column_widths[key]}}" for key in recall_dict.keys()]) + " |"
            print(header_row)

            # Print the separator line for the table
            separator_row = "| " + \
                " | ".join(["-" * column_widths[key]
                           for key in recall_dict.keys()]) + " |"
            print(separator_row)

            # Print the values (the data in a single row)
            value_row = "| " + " | ".join([f"{round(value, 5):<{column_widths[key]}.5f}".replace(
                '.', ',') for key, value in recall_dict.items()]) + " |"
            print(value_row)

            with Run().open(f"{args.recall_results_filename}.json", 'w') as f:
                json.dump(recall_dict, f, indent=4)
            with Run().open(f"{args.recall_results_filename}.txt", 'w') as f:
                f.writelines(
                    '\n'.join([header_row, separator_row, value_row, '']))
        else:
            raise ValueError(f"Unknown action: {args.action}")

        print("done")
