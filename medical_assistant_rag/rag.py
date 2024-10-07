import os
import json
from time import time
from dotenv import load_dotenv
from openai import OpenAI
from ingest import hybrid_query_rrf  # Import the search function from ingest.py

# Load environment variables
load_dotenv('/home/ubuntu/medical_assistant_rag/.envrc')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI client
client = OpenAI()

def search(query):
    return hybrid_query_rrf(query)

# Prompt templates
prompt_template = """
You are a knowledgeable medical assistant. Answer the QUESTION based solely on the information provided in the CONTEXT from the medical database.

Use only the facts from the CONTEXT when formulating your answer.

QUESTION: {question}

CONTEXT:
{context}
""".strip()

entry_template = """
Medical Department: {medical_department}
Condition Type: {condition_type}
Patient Demographics: {patient_demographics}
Common Symptoms: {common_symptoms}
Treatment or Management: {treatment_or_management}
Severity: {severity}
""".strip()

def build_prompt(query, search_results):
    context = ""
    for doc in search_results:
        context += entry_template.format(
            medical_department=doc.get('medical_department', 'N/A'),
            condition_type=doc.get('condition_type', 'N/A'),
            patient_demographics=doc.get('patient_demographics', 'N/A'),
            common_symptoms=doc.get('common_symptoms', 'N/A'),
            treatment_or_management=doc.get('treatment_or_management', 'N/A'),
            severity=doc.get('severity', 'N/A')
        ) + "\n\n"
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

def llm(prompt, model='gpt-4o-mini'):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    token_stats = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    return answer, token_stats

evaluation_prompt_template = """
You are an expert evaluator for a RAG system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()

def evaluate_relevance(question, answer):
    prompt = evaluation_prompt_template.format(question=question, answer=answer)
    evaluation, tokens = llm(prompt, model="gpt-4o-mini")

    try:
        json_eval = json.loads(evaluation)
        return json_eval, tokens
    except json.JSONDecodeError:
        result = {"Relevance": "UNKNOWN", "Explanation": "Failed to parse evaluation"}
        return result, tokens

def calculate_openai_cost(model, tokens):
    openai_cost = 0

    if model == "gpt-4o-mini":
        openai_cost = (
            tokens["prompt_tokens"] * 0.00015 + tokens["completion_tokens"] * 0.0006
        ) / 1000
    else:
        print("Model not recognized. OpenAI cost calculation failed.")

    return openai_cost

def rag(query, model='gpt-4o-mini'):
    t0 = time()

    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer, token_stats = llm(prompt, model=model)

    relevance, rel_token_stats = evaluate_relevance(query, answer)

    t1 = time()
    took = t1 - t0

    openai_cost_rag = calculate_openai_cost(model, token_stats)
    openai_cost_eval = calculate_openai_cost(model, rel_token_stats)

    openai_cost = openai_cost_rag + openai_cost_eval

    answer_data = {
        "answer": answer,
        "model_used": model,
        "response_time": took,
        "relevance": relevance.get("Relevance", "UNKNOWN"),
        "relevance_explanation": relevance.get(
            "Explanation", "Failed to parse evaluation"
        ),
        "prompt_tokens": token_stats["prompt_tokens"],
        "completion_tokens": token_stats["completion_tokens"],
        "total_tokens": token_stats["total_tokens"],
        "eval_prompt_tokens": rel_token_stats["prompt_tokens"],
        "eval_completion_tokens": rel_token_stats["completion_tokens"],
        "eval_total_tokens": rel_token_stats["total_tokens"],
        "openai_cost": openai_cost,
    }

    return answer_data

# # Example usage
# if __name__ == "__main__":
#     query = "A 30-year-old woman in her second trimester of pregnancy presents with symptoms of dysuria and urinary urgency. She has no significant medical history and is not allergic to any medications. Physical examination and vital signs are within normal limits. Which antibiotic is considered safe and effective for treating her urinary tract infection during pregnancy?"
#     answer_data = rag(query)
#     print("Answer:", answer_data["answer"])
