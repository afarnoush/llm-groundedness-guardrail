import pandas as pd
import time
from tqdm import tqdm
from langchain.prompts import PromptTemplate
# from langchain.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
import gc
from src.utils import pretty_print_docs,calculate_cosine_similarity
from src.retriever_builder import build_ensemble_retriever_from_markdown
from pydantic import BaseModel



def evaluate_rag(
    evaluation_question_path,
    template,
    markdown_path,
    search_kwargs=7,
    model="gemma3:4b",
       
):
    """
    Evaluate a RAG-based chatbot using a list of input questions and expected answers.

    Parameters:
    -----------
    evaluation_question_path : str
        Path to the CSV file with evaluation questions and expected answers.
    template : str
        Prompt template for the LLM.
    ensemble_retriever : retriever
        A retriever instance that supports .invoke(query) and returns relevant documents.
    search_kwargs : int
        Number of documents to retrieve.
    model : str
        Name of the local Ollama model to use (e.g., "gemma3").
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing evaluation results including time, similarity, and retrieved docs.
    """

    
    # Build ensemble retriever from markdown doc
    ensemble_retriever = build_ensemble_retriever_from_markdown(markdown_path)
    
     # Initialize prompt and model once outside the loop
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    llm = Ollama(model=model)
    llm_chain = prompt | llm 
    # llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Define response generator
    def get_response(q):
        context = ensemble_retriever.invoke(q)[:search_kwargs]
        result = llm_chain.invoke({"question": q, "context": context}).strip()
        return result,context


    # Load evaluation questions
    df = pd.read_csv(evaluation_question_path, encoding='utf-8-sig')
    questions = df['question']

    generated_answers = []
    time_taken = []
    relevant_docs = []

    # Generate answers and track evaluation
    for q in tqdm(questions):
            start_time = time.time()
            response,context=get_response(q)
            generated_answers.append(str(response))
            end_time = time.time()

            relevant_documents = ensemble_retriever.invoke((q))
            relevant_docs.append({'query': q, 'relevant_documents': pretty_print_docs(relevant_documents)})
            time_taken.append(end_time - start_time)	
            
    print("   Preparing the results...")
    
    df_results = df.copy()
    df_results['question_length'] = df_results['question'].apply(len)
    df_results['expected_answer_length'] = df_results['expected_answer'].apply(len)
    df_results['generated_answer'] = generated_answers
    df_results['generated_answer_length'] = df_results['generated_answer'].apply(len)
    df_results['cosine_similarity'] = df_results.apply(calculate_cosine_similarity, axis=1)
    df_results = pd.concat([df_results, pd.DataFrame(relevant_docs)['relevant_documents']], axis=1)
    
    df_results['time_taken'] = time_taken
     
    
    print("Evaluation finished...")

    del llm
    gc.collect()
    return df_results 