from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from pydantic import BaseModel


def build_ensemble_retriever_from_markdown(
    markdown_path: str,
    save_path: str = "faiss",
    score_threshold: float = 0.6,
    k: int = 7,
    faiss_weight: float = 0.75,
    bm25_weight: float = 0.25
):
    """
    Builds an ensemble retriever combining FAISS (vector) and BM25 from a markdown file.

    Parameters:
    -----------
    markdown_path : str
        Path to the markdown file.
    save_path : str
        Directory to save and load FAISS index.
    score_threshold : float
        Similarity threshold for FAISS retriever.
    k : int
        Number of top documents to retrieve.
    faiss_weight : float
        Weight for the FAISS retriever.
    bm25_weight : float
        Weight for the BM25 retriever.

    Returns:
    --------
    EnsembleRetriever
        A combined retriever for use in RAG.
    """
    # 1. Load markdown file
    with open(markdown_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 2. Split into header-based chunks
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    docs = markdown_splitter.split_text(text)

    # 3. Create embeddings and FAISS index
    hf_embedding = HuggingFaceInstructEmbeddings()
    db = FAISS.from_documents(docs, hf_embedding)
    db.save_local(save_path)

    # 4. Reload FAISS index
    db = FAISS.load_local(save_path, embeddings=hf_embedding, allow_dangerous_deserialization=True)

    # 5. Create vector-based retriever
    vector_retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": score_threshold, "k": k}
    )

    # 6. Create BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k

    # 7. Combine with weights
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[faiss_weight, bm25_weight]
    )

    return ensemble_retriever