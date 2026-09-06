import arxiv
import json
from vdb.store import VectorStore
from vdb.utils.metrics import MetricType
from sentence_transformers import SentenceTransformer
import random
import time
import numpy as np

def create_corpus():
    client = arxiv.Client(delay_seconds=10)
    searches = []

    searches.append(arxiv.Search(
        query='cs.LG',
        max_results=500,
        sort_by=arxiv.SortCriterion.SubmittedDate
    ))

    searches.append(arxiv.Search(
        query='math.FA',
        max_results=500,
        sort_by=arxiv.SortCriterion.SubmittedDate
    ))

    searches.append(arxiv.Search(
        query='math.PR',
        max_results=500,
        sort_by=arxiv.SortCriterion.SubmittedDate
    ))

    searches.append(arxiv.Search(
        query='cs.NA',
        max_results=500,
        sort_by=arxiv.SortCriterion.SubmittedDate
    ))

    searches.append(arxiv.Search(
        query='cs.DM',
        max_results=500,
        sort_by=arxiv.SortCriterion.SubmittedDate
    ))

    searches.append(arxiv.Search(
        query='physics.optics',
        max_results=500,
        sort_by=arxiv.SortCriterion.SubmittedDate
    ))

    searches.append(arxiv.Search(
        query='quant-ph',
        max_results=500,
        sort_by=arxiv.SortCriterion.SubmittedDate
    ))

    searches.append(arxiv.Search(
            query='q-fin.ST',
            max_results=500,
            sort_by=arxiv.SortCriterion.SubmittedDate
    ))

    entries = []
    for search in searches:
        entries.extend([{'title': result.title, 'categories': result.categories, 'abstract': result.summary.replace('\n', ' ')}
                        for result in client.results(search)])

    with open('corpus.json', 'w') as f:
        json.dump(entries, f, indent=2)

def load_database_fresh(vdb: VectorStore, model: SentenceTransformer, corpus_file: str, embed_file: str, meta_file: str, count=4_000):
    with open(corpus_file, 'r') as f:
        content = json.load(f)

    sampled_content = random.sample(content, min(len(content), count))

    abstracts = []
    metadata_list = []
    for block in sampled_content:
        abstracts.append(block['abstract'])
        metadata_list.append({
            'abstract': block['abstract'],
            'title': block['title'],
            'categories': block['categories']
        })

    embeddings_array = model.encode(abstracts, convert_to_numpy=True, show_progress_bar=True)
    vdb.add_bulk(embeddings_array, metadata_list)

    np.save(embed_file, embeddings_array)
    with open(meta_file, 'w') as f:
        json.dump(metadata_list, f, indent=2)

def load_database_cached(vdb: VectorStore, embed_file: str, meta_file: str):
    with open(meta_file, 'r') as f:
        metadata_list = json.load(f)

    embeddings_array = np.load(embed_file)
    vdb.add_bulk(embeddings_array, metadata_list)

def retrieve(vdb, model, query, k):
    query_vector = model.encode(query, convert_to_numpy=True)
    top_k = vdb.flat_search(query_vector, k=k, metric=MetricType.COSINE) # (score, id)
    ids = [res[1] for res in top_k]
    return [vdb.get_metadata(id) for id in ids]

if __name__ == '__main__':
    vdb = VectorStore(dim=384)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    start = time.perf_counter()
    load_database_fresh(vdb, model, 'corpus.json', 'embed.npy', 'metadata.json')
    end = time.perf_counter()
    print(f"TIME: {end - start}")