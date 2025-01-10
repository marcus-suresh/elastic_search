import elastic_search

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



# 4.   Elastic Search


#Input title and abstract to find similar documents
papers_results = semantic_scanner(title='gender studies', 
              abstract='')

toc = time.perf_counter()
print(f"Semantic Elastic Search executed in {toc - tic:0.4f} seconds")
