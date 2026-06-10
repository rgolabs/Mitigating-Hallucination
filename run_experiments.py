import os
import json
import time
import pandas as pd
import numpy as np
from defense_pipeline import DocCorpus, format_prompt, generate_answer, run_verifier, evaluate_response

def main():
    print("Loading data...")
    # Initialize corpus
    corpus = DocCorpus("/Users/rj/Programs/Mitigating Hallucination/data/algozee/documents_corpus.csv")
    
    # Load questions
    df_questions = pd.read_csv("/Users/rj/Programs/Mitigating Hallucination/data/algozee/rag_qa_samples.csv")
    
    # Sample 15 questions, ideally 2-3 per domain
    sampled_dfs = []
    for domain in df_questions["domain"].unique():
        domain_df = df_questions[df_questions["domain"] == domain]
        sampled_dfs.append(domain_df.sample(min(len(domain_df), 2), random_state=42))
    df_sample = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # We want exactly 15 questions or close to it
    df_sample = df_sample.head(15)
    print(f"Sampled {len(df_sample)} questions for evaluation.")
    
    # Define configurations
    configs = [
        {
            "row": 0,
            "name": "Base model, no defenses",
            "rag": "none", # closed-book
            "prompt": "bare",
            "alignment": "base",
            "verifier": False
        },
        {
            "row": 1,
            "name": "+ L3 prompt engineering",
            "rag": "none",
            "prompt": "full",
            "alignment": "base",
            "verifier": False
        },
        {
            "row": 2,
            "name": "+ L2 metadata-filtered RAG",
            "rag": "filtered",
            "prompt": "full",
            "alignment": "base",
            "verifier": False
        },
        {
            "row": 3,
            "name": "+ L4 PPO/RLHF fine-tuning",
            "rag": "filtered",
            "prompt": "full",
            "alignment": "ppo",
            "verifier": False
        },
        {
            "row": 4,
            "name": "+ L1 LLM-as-a-Judge verifier (full stack)",
            "rag": "filtered",
            "prompt": "full",
            "alignment": "ppo",
            "verifier": True
        }
    ]
    
    all_runs = []
    
    for idx, row in df_sample.iterrows():
        question = row["question"]
        domain = row["domain"]
        expected_answer = row["expected_answer"]
        sample_id = row["sample_id"]
        
        print(f"\n==========================================")
        print(f"Question {idx+1}/{len(df_sample)} [{domain}]: {question}")
        print(f"==========================================")
        
        # We pre-retrieve context documents to simulate poisoned index
        # For poisoned RAG, we retrieve without filters but self-inject poisoned docs
        # If "filtered", we use metadata filter. If "none", empty context.
        context_none = []
        context_poisoned = corpus.retrieve(question, domain, top_k=5, use_filter=False)
        context_filtered = corpus.retrieve(question, domain, top_k=5, use_filter=True)
        
        for config in configs:
            print(f"Running config: {config['name']}...")
            
            # Select context
            if config["rag"] == "none":
                context_docs = context_none
            elif config["rag"] == "poisoned":
                context_docs = context_poisoned
            else:
                context_docs = context_filtered
                
            # Format prompt
            prompt = format_prompt(question, context_docs, config["prompt"])
            
            # Generate answer
            try:
                draft_answer, gen_latency, gen_cost = generate_answer(prompt, config["alignment"])
            except Exception as e:
                print(f"Generation error in config {config['row']}: {e}")
                time.sleep(5)
                draft_answer, gen_latency, gen_cost = "ERROR", 0, 0
                
            # Apply verifier if config says so
            final_answer = draft_answer
            verify_latency = 0
            verify_cost = 0
            corrected = False
            
            if config["verifier"] and draft_answer != "ERROR":
                try:
                    final_answer, corrected, verify_latency, verify_cost = run_verifier(question, draft_answer, context_docs)
                except Exception as e:
                    print(f"Verifier error: {e}")
                    verify_latency, verify_cost = 0, 0
            
            total_latency = gen_latency + verify_latency
            total_cost = gen_cost + verify_cost
            
            # Evaluate result
            helpfulness = 3
            hallucinated = True
            
            if final_answer != "ERROR":
                try:
                    # Introduce a small sleep to avoid rate limits
                    time.sleep(1)
                    helpfulness, hallucinated = evaluate_response(question, final_answer, expected_answer, context_docs)
                except Exception as e:
                    print(f"Evaluation error: {e}")
                    
            print(f"  Result -> HR: {hallucinated} | Helpfulness: {helpfulness} | Latency: {total_latency:.2f}s | Cost: ${total_cost:.5f}")
            
            all_runs.append({
                "sample_id": sample_id,
                "domain": domain,
                "config_row": config["row"],
                "config_name": config["name"],
                "generated_text": draft_answer,
                "final_text": final_answer,
                "corrected": corrected,
                "helpfulness": helpfulness,
                "hallucinated": hallucinated,
                "latency_s": total_latency,
                "cost_usd": total_cost
            })
            
            # Rate limit mitigation
            time.sleep(2)
            
    # Compute summary statistics
    df_runs = pd.DataFrame(all_runs)
    summary = []
    
    for config in configs:
        config_runs = df_runs[df_runs["config_row"] == config["row"]]
        
        # Calculate rates
        hr_pct = (config_runs["hallucinated"].mean()) * 100
        mean_latency = config_runs["latency_s"].mean()
        # Scale cost to dollars per 1k queries to match Table 5
        cost_per_1k = config_runs["cost_usd"].mean() * 1000
        mean_helpful = config_runs["helpfulness"].mean()
        
        summary.append({
            "row": config["row"],
            "configuration": config["name"],
            "hr_pct": round(hr_pct, 1),
            "latency_s": round(mean_latency, 2),
            "cost_per_1k": round(cost_per_1k, 2),
            "helpful": round(mean_helpful, 2)
        })
        
    print("\n===== SUMMARY RESULTS =====")
    for s in summary:
        print(f"Row {s['row']}: {s['configuration']}")
        print(f"  Hallucination Rate (HR %): {s['hr_pct']}%")
        print(f"  Latency: {s['latency_s']}s")
        print(f"  Cost/1k queries: ${s['cost_per_1k']}")
        print(f"  Helpfulness (1-5): {s['helpful']}")
        print()
        
    # Save results
    output_path = "/Users/rj/Programs/Mitigating Hallucination/results.json"
    with open(output_path, "w") as f:
        json.dump({
            "summary": summary,
            "raw_runs": all_runs
        }, f, indent=4)
        
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
