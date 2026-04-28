from langchain_classic.retrievers import EnsembleRetriever
from services.embedding_service import vectordb, bm25_retriever, reranker_model

def get_retrieved_context(search_query, query_filter, top_k=3):
    """Filtreleri alıp dinamik hibrit arama ve yeniden sıralama yapar."""
    
    search_kwargs = {"k": 15}
    if query_filter:
        search_kwargs["filter"] = query_filter
        print(f"🔍 Aktif Metadata Filtresi: {query_filter}")
        
    dynamic_vector_retriever = vectordb.as_retriever(search_kwargs=search_kwargs)
    
    # BM25 boşsa (veri yoksa) sadece vektör araması yapmayı dene
    if bm25_retriever:
        dynamic_ensemble = EnsembleRetriever(
            retrievers=[bm25_retriever, dynamic_vector_retriever],
            weights=[0.5, 0.5]
        )
        results = dynamic_ensemble.invoke(search_query)
    else:
        results = dynamic_vector_retriever.invoke(search_query)
    
    if not results: 
        return "İlgili bilgi bulunamadı."

    cross_inp = [[search_query, doc.page_content] for doc in results]
    scores = reranker_model.predict(cross_inp)
    
    scored_docs = list(zip(results, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    best_docs = scored_docs[:top_k]

# En iyi ilk 3 belgeyi yazdır
    for i, doc in enumerate(best_docs[:3], start=1):
        print("skor: " + str(scores[i-1]) + "\n" + "-"*50,f"skor {i}. belge:\n{doc}\n")

    print("Bu da sıralanmış en iyi belgeler ve skorlarıdır. En yüksek skorlu ilk 3 belgeyi alıyoruz.")
    
    return "\n\n".join([f"[KAYNAK {i+1}]: {doc.page_content}" for i, (doc, score) in enumerate(best_docs)])