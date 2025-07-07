from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class SimpleRAGRetriever:
    def __init__(self,corpus, embedding_model="all-MiniLM-l6-V2") -> None:
        # load quantized model
        self.embedder = SentenceTransformer(embedding_model)
        # quantize the document
        self.documents = corpus
        self.document_embeddings = self.embedder.encode(corpus,convert_to_numpy=True)
        # use FAISS to generate index
        dim = self.document_embeddings.shape[1]
        self.index = faiss.IndexFlat2(dim)
        self.index.add(self.document_embeddings)

    def retrieve(self,query,top_k=3):
        query_vec = self.embedder.encode([query],convert_to_numpy=True)
        distances, indices = self.index.search(query_vec, top_k)
        return [self.documents[i] for i in indices[0]]
        