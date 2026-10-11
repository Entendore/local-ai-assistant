"""
SQLite-backed RAG vector store for instant loading and scalable storage.
"""
import os
import sqlite3
import numpy as np
from llama_cpp import Llama

class LocalRAG:
    def __init__(self, embed_model_path: str, db_path: str = "rag_db.sqlite", n_gpu_layers: int = -1, flash_attn: bool = True, num_threads: int = 8):
        self.db_path = db_path
        print(f"Loading embedding model: {embed_model_path}...")
        self.embed_model = Llama(
            model_path=embed_model_path, embedding=True, n_ctx=512,
            n_gpu_layers=n_gpu_layers, n_batch=512, flash_attn=flash_attn,
            num_threads=num_threads, verbose=False
        )
        
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, source TEXT, text TEXT, embedding BLOB)")
        self.conn.commit()
        
        self.cursor.execute("SELECT source, text, embedding FROM chunks")
        rows = self.cursor.fetchall()
        self.documents = [{"source": r[0], "text": r[1]} for r in rows]
        self.embeddings = [np.frombuffer(r[2], dtype=np.float32) for r in rows]
        print(f"Loaded RAG database ({len(self.documents)} chunks).")

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start + chunk_size])
            start += chunk_size - overlap
        return [c for c in chunks if c.strip()]

    def _normalize(self, vec: list[float]) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return (np.array(vec, dtype=np.float32) / norm) if norm > 0 else np.array(vec, dtype=np.float32)

    def add_text(self, text: str, source: str) -> str:
        chunks = self._chunk_text(text)
        if not chunks: return "No text to index."

        emb_response = self.embed_model.create_embedding(chunks)
        for i, chunk in enumerate(chunks):
            emb = self._normalize(emb_response["data"][i]["embedding"])
            self.cursor.execute("INSERT INTO chunks (source, text, embedding) VALUES (?, ?, ?)",
                                (source, chunk, emb.tobytes()))
            self.documents.append({"source": source, "text": chunk})
            self.embeddings.append(emb)
            
        self.conn.commit()
        return f"Successfully indexed {len(chunks)} chunks from {source}."

    def search(self, query: str, top_k: int = 3) -> str:
        if not self.documents: return "No documents indexed."
        query_emb = self._normalize(self.embed_model.create_embedding([query])["data"][0]["embedding"])
        matrix = np.array(self.embeddings, dtype=np.float32)
        
        scores = np.dot(matrix, query_emb)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0.35:
                doc = self.documents[idx]
                results.append(f"Source: {doc['source']} (Score: {scores[idx]:.2f})\nContent: {doc['text']}")
        return "\n---\n".join(results) if results else "No relevant information found."

    def list_files(self) -> str:
        if not self.documents: return "No files indexed."
        files = list(set([d['source'] for d in self.documents]))
        return "Indexed Files:\n" + "\n".join(f"- {f}" for f in files)

    def clear_db(self):
        self.cursor.execute("DROP TABLE IF EXISTS chunks")
        self.conn.commit()
        self.documents = []
        self.embeddings = []
