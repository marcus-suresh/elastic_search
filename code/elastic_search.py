# -*- coding: utf-8 -*-
"""
Text Similarity - Semantic Elastic Search

# Semantic Elastic Search

The Semantic Elastic Search Tool is designed to estimate the semantic similarity of a target text within a given knowledge base (corpus).
It's core function is harness transformer based large language models to generate embeddings which serve as the semantic search target. 
The output is a list of documents that contain the semantic search target within the identified document along with a similarity metric.

"""

# =============================================================================
# 1.   Installation & ENV set-up
# =============================================================================

#!pip install sentence-transformers #Installation of sentence-transformers pipeline from HuggingFace
"""Run this pip install for first time installation"""

# =============================================================================
# 2.   Import and Ingestion"""
# =============================================================================

import os
import sys
import time
import json
import pandas
import torch
import gc
gc.collect()

from sentence_transformers import SentenceTransformer, util
"""Standard DS toolkit imports"""

tic = time.perf_counter()

# =============================================================================
#CUDA Core Settings
# =============================================================================
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

torch.cuda.is_available()
if torch.cuda.is_available(): 
 dev = "cuda:0" 
else: 
 dev = "cpu" 
device = torch.device(dev) 

# =============================================================================
#EXCEL2JSON Conversion
# =============================================================================

# #Data Ingestion (Excel2JSON)
# data_excel = pandas.read_excel("")
# data_json = data_excel.to_json(orient='records')
#   
# with open('data.json', 'w') as f:
#     json.dump(data_json, f)
# 
# # Print out the result
# print('Excel Sheet to JSON:\n', data_json)
# 
# # Make the string into a list to be able to input in to a JSON-file
# data_json_dict = json.loads(data_json)
# 
# # Define file to write to and 'w' for write option -> json.dump() 
# # defining the list to write from and file to write to
# with open('', 'w') as json_file:
#     json.dump(data_json_dict, json_file)
# print('Excel Sheet to JSON:\n', data_json)
# =============================================================================
"""Above code is for Excel2JSON conversion if input data is not allready in JSON"""

# =============================================================================
#Data Ingestion (JSON)
# =============================================================================

DATA = "F://GitHub//__data_science_projects__//elastic_search//data//emnlp2016-2018.json"
AI_LLM_path = "F://GitHub//__data_science_projects__//elastic_search//_LM_//googleflan-t5-xxl"

#t5-base
#t5-large
#google-flan-t5-xl
#google-flan-t5-xxl
#gpt2
"""Current available LLM in our directory D:\\_LM_\\"""

if not os.path.exists(DATA):
  util.http_get("https://sbert.net/datasets/emnlp2016-2018.json", DATA)

with open(DATA) as fIn:
  papers = json.load(fIn)

print(len(papers), ": Total number of documents in knowledge base")

# =============================================================================
# 3.   Transformer Model & Embeddings"""
# =============================================================================

#Select Transformer - Use the copy function in Huggingface to paste the model handle
transformer_model = SentenceTransformer(AI_LLM_path)

#Combine title and abstract into continous string
document_texts = [paper['title'] + '[SEP]' + paper['abstract'] + '[SEP]' + paper['url'] + '[SEP]' + paper['venue'] + '[SEP]' + paper['year'] for paper in papers]

#Compute embeddings for all text
corpus_embeddings = transformer_model.encode(document_texts, convert_to_tensor=True)
torch.save(corpus_embeddings, "pytorch_embeddings//computed_embeddings//computed_embeddings.pt")


# =============================================================================
#Input: title & abstract. Output: Search knowledgebase (corpus) for highly relevant matches by similarity metrics
def semantic_scanner(title, abstract):
  query_embedding = transformer_model.encode(title+'[SEP]'+abstract, convert_to_tensor=True)

  search_hits = util.semantic_search(query_embedding, corpus_embeddings,top_k=len(corpus_embeddings))
  search_hits = search_hits[0]

  print("***Title:***", title)
  print("Most Similar Papers:")
  for hit in search_hits:
    related_paper = papers[hit['corpus_id']]
    print("{:.5f}\t{}\t{} {}".format(hit['score'], related_paper['title'], related_paper['venue'], related_paper['year']))
  return search_hits

# =============================================================================
# 4.   Elastic Search
# =============================================================================

#Input title and abstract to find similar documents
papers_results = semantic_scanner(title='Attention-based Hierarchical Neural Networks', 
              abstract='Attention-based Hierarchical Neural Networks')

toc = time.perf_counter()
print(f"Semantic Elastic Search executed in {toc - tic:0.4f} seconds")

###TODO###
#Filepaths
#Create new py file that loads the pt files
#Investigate why GPU capping at 5GB
#Visualisation of embeddings
