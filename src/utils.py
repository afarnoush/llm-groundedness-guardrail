
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from collections import Counter
import pandas as pd

#-------------------------------------------------------------------------------
# Load embedding model once globally
embed_model = SentenceTransformer('distilbert-base-nli-mean-tokens')

def pretty_print_docs(docs):
    """
    Formats a list of documents into a human-readable string.

    Parameters:
    -----------
    docs : list
        List of documents returned by a retriever.

    Returns:
    --------
    str
        A formatted string showing document contents.
    """
    return "\n" + ("\n" + "-" * 100 + "\n").join(
        [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
    )
#-------------------------------------------------------------------------------
def calculate_cosine_similarity(row):
    """
    Calculate cosine similarity between expected and generated chatbot responses.

    Parameters:
    -----------
    row : pd.Series
        A row containing 'expected_answer' and 'generated_answer' fields.

    Returns:
    --------
    float
        Cosine similarity score between the expected and generated response.
    """
    emb_expected = embed_model.encode(row['expected_answer'], convert_to_tensor=True)
    emb_generated = embed_model.encode(row['generated_answer'], convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(emb_expected, emb_generated)
    return cosine_sim.item()

#-------------------------------------------------------------------------------
rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

def simple_bleu(reference, candidate):
    ref_words = reference.split()
    cand_words = candidate.split()
    ref_count = Counter(ref_words)
    cand_count = Counter(cand_words)

    match_count = sum(min(cand_count[word], ref_count[word]) for word in cand_count)
    total_count = len(cand_words)

    if total_count == 0:
        return 0.0
    return match_count / total_count

def calculate_bleu_rouge(row):
    bleu_score = simple_bleu(row['expected_answer'], row['generated_answer'])
    rouge_scores = rouge.score(row['expected_answer'], row['generated_answer'])

    rouge_1 = rouge_scores["rouge1"].fmeasure
    rouge_l = rouge_scores["rougeL"].fmeasure

    return pd.Series({
        "bleu": bleu_score,
        "rouge1": rouge_1,
        "rougeL": rouge_l
    })

