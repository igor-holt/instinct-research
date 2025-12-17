"""
Instinct Platform - TAO Chat Evaluation Framework
Evaluates response relevance and groundedness for the TAO assistant.

Run: python evaluation/run_evaluation.py
"""

import os
import json
from typing import Any

# Custom Code-Based Evaluators (no LLM required)

class RelevanceEvaluator:
    """
    Evaluates how well the response addresses the query by checking
    if expected topics are present in the response.
    """

    def __init__(self):
        pass

    def __call__(self, *, query: str, response: str, expected_topics: list = None, **kwargs) -> dict:
        if not response:
            return {"relevance_score": 0.0, "relevance_reason": "Empty response"}

        response_lower = response.lower()
        query_lower = query.lower()

        # Check if response echoes query context
        query_words = set(query_lower.split())
        response_words = set(response_lower.split())
        overlap = len(query_words & response_words) / max(len(query_words), 1)

        # Check expected topics if provided
        topic_score = 0.0
        if expected_topics:
            topics_found = sum(1 for t in expected_topics if t.lower() in response_lower)
            topic_score = topics_found / len(expected_topics)

        # Combined score
        score = (overlap * 0.3) + (topic_score * 0.7) if expected_topics else overlap
        score = min(1.0, score)

        return {
            "relevance_score": round(score, 2),
            "relevance_reason": f"Query overlap: {overlap:.0%}, Topic coverage: {topic_score:.0%}" if expected_topics else f"Query overlap: {overlap:.0%}"
        }


class GroundednessEvaluator:
    """
    Evaluates if the response stays consistent with the TAO telemetry context
    (energy, penalty, scores) provided in the context.
    """

    def __init__(self):
        pass

    def __call__(self, *, response: str, context: str, **kwargs) -> dict:
        if not response or not context:
            return {"groundedness_score": 0.0, "groundedness_reason": "Missing response or context"}

        response_lower = response.lower()
        context_lower = context.lower()

        # Extract key values from context
        context_values = []
        import re

        # Look for numeric values and key terms
        numbers = re.findall(r'\d+\.?\d*', context)
        context_values.extend(numbers[:4])  # First 4 numbers (energy, penalty, patch, ignore)

        # Key regime indicators
        if 'standard' in context_lower:
            context_values.append('standard')
        if 'inversion' in context_lower:
            context_values.append('inversion')
        if 'transition' in context_lower:
            context_values.append('transition')

        # Check how many context values appear in response
        values_found = sum(1 for v in context_values if str(v).lower() in response_lower)
        score = values_found / max(len(context_values), 1)

        # Check for contradictions (saying wrong regime)
        contradictions = 0
        if 'standard' in context_lower and 'inversion regime' in response_lower:
            contradictions += 1
        if 'inversion' in context_lower and 'standard regime' in response_lower:
            contradictions += 1

        score = max(0, score - (contradictions * 0.3))

        return {
            "groundedness_score": round(score, 2),
            "groundedness_reason": f"Context values found: {values_found}/{len(context_values)}, Contradictions: {contradictions}"
        }


class LoopDetectionEvaluator:
    """
    Detects infinite loops or repetitive patterns in responses
    that might indicate tensorized fraction rotation issues.
    """

    def __init__(self, max_repetition_ratio: float = 0.5):
        self.max_repetition_ratio = max_repetition_ratio

    def __call__(self, *, response: str, **kwargs) -> dict:
        if not response:
            return {"loop_detected": False, "repetition_ratio": 0.0, "loop_reason": "Empty response"}

        words = response.split()
        if len(words) < 10:
            return {"loop_detected": False, "repetition_ratio": 0.0, "loop_reason": "Response too short to analyze"}

        # Check for repeated phrases (3-gram analysis)
        ngrams = []
        for i in range(len(words) - 2):
            ngram = ' '.join(words[i:i+3])
            ngrams.append(ngram)

        unique_ngrams = set(ngrams)
        repetition_ratio = 1 - (len(unique_ngrams) / max(len(ngrams), 1))

        # Check for exact sentence repetition
        sentences = response.split('.')
        unique_sentences = set(s.strip().lower() for s in sentences if s.strip())
        sentence_repetition = 1 - (len(unique_sentences) / max(len(sentences), 1))

        loop_detected = repetition_ratio > self.max_repetition_ratio or sentence_repetition > 0.5

        return {
            "loop_detected": loop_detected,
            "repetition_ratio": round(max(repetition_ratio, sentence_repetition), 2),
            "loop_reason": f"3-gram repetition: {repetition_ratio:.0%}, Sentence repetition: {sentence_repetition:.0%}"
        }


class ResponseLatencyEvaluator:
    """
    Evaluates response generation time against acceptable thresholds.
    """

    def __init__(self, max_latency_ms: int = 1000):
        self.max_latency_ms = max_latency_ms

    def __call__(self, *, latency_ms: float = 0, **kwargs) -> dict:
        is_acceptable = latency_ms <= self.max_latency_ms
        score = max(0, 1 - (latency_ms / (self.max_latency_ms * 2)))

        return {
            "latency_acceptable": is_acceptable,
            "latency_score": round(score, 2),
            "latency_reason": f"{latency_ms}ms vs {self.max_latency_ms}ms threshold"
        }


def run_evaluation(data_path: str, output_path: str = None) -> dict:
    """
    Run evaluation on test dataset and return aggregated results.
    """
    # Initialize evaluators
    relevance_eval = RelevanceEvaluator()
    groundedness_eval = GroundednessEvaluator()
    loop_eval = LoopDetectionEvaluator()
    latency_eval = ResponseLatencyEvaluator()

    results = []

    # Load test data
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)

                # Simulate response generation (in real scenario, call your agent)
                simulated_response = generate_simulated_response(data)

                # Run evaluators
                relevance_result = relevance_eval(
                    query=data['query'],
                    response=simulated_response['response'],
                    expected_topics=data.get('expected_topics', [])
                )

                groundedness_result = groundedness_eval(
                    response=simulated_response['response'],
                    context=data['context']
                )

                loop_result = loop_eval(response=simulated_response['response'])

                latency_result = latency_eval(latency_ms=simulated_response.get('latency_ms', 0))

                # Combine results
                row_result = {
                    "id": data['id'],
                    "query": data['query'],
                    "response": simulated_response['response'],
                    "context": data['context'],
                    **relevance_result,
                    **groundedness_result,
                    **loop_result,
                    **latency_result
                }
                results.append(row_result)

    # Calculate aggregates
    aggregates = {
        "total_samples": len(results),
        "avg_relevance_score": sum(r['relevance_score'] for r in results) / len(results),
        "avg_groundedness_score": sum(r['groundedness_score'] for r in results) / len(results),
        "loops_detected": sum(1 for r in results if r['loop_detected']),
        "avg_latency_score": sum(r['latency_score'] for r in results) / len(results),
        "pass_rate": sum(1 for r in results if r['relevance_score'] >= 0.5 and r['groundedness_score'] >= 0.3 and not r['loop_detected']) / len(results)
    }

    evaluation_output = {
        "aggregates": aggregates,
        "rows": results
    }

    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(evaluation_output, f, indent=2)
        print(f"Results saved to {output_path}")

    return evaluation_output


def generate_simulated_response(data: dict) -> dict:
    """
    Simulates the TAO chat bot response for evaluation purposes.
    In production, replace this with actual agent invocation.
    """
    import time
    import random

    start = time.time()

    query = data['query'].lower()
    context = data['context']

    # Extract values from context
    import re
    numbers = re.findall(r'[\d.]+', context)
    energy = numbers[0] if len(numbers) > 0 else "80"
    penalty = numbers[1] if len(numbers) > 1 else "1.12"
    patch = numbers[2] if len(numbers) > 2 else "0.74"
    ignore = numbers[3] if len(numbers) > 3 else "0.46"

    regime = "Standard"
    if "inversion" in context.lower():
        regime = "Inversion"
    elif "transition" in context.lower():
        regime = "Transition"

    # Generate contextual response
    response = f"**TAO Snapshot**\n- Energy: {energy}J\n- Penalty: {penalty}x\n- Patch: {patch}\n- Ignore: {ignore}\n\n"

    if "optimization inversion" in query:
        response += "Optimization Inversion occurs when infrastructure overhead dominates AI workloads. The system inverts priorities from Utility to Cost when energy budgets bind."
    elif "landauer" in query:
        response += f"The Landauer Penalty amplifies cost metrics using a sigmoid function. Current penalty is {penalty}x, meaning cost considerations are weighted {penalty} times higher."
    elif "exceed" in query or "budget" in query:
        response += f"When energy exceeds 100J budget, the system enters {regime} Regime. Cost-aware plans dominate to prevent thermal runaway."
    elif "tao" in query or "thermal" in query:
        response += "TAO prevents thermal runaway by automatically reweighting plans. During phase transition, cost-aware strategies take precedence."
    elif "1%" in query or "fallacy" in query:
        response += "The 1% Fallacy: Model inference consumes negligible energy compared to infrastructure (cooling, orchestration). Infrastructure accounts for 85%+ of energy."
    elif "eu" in query or "article" in query:
        response += "Instinct addresses Article 53 (Transparency), Article 55 (Risk Mitigation), and Annex XI (Reporting) of the EU AI Act through hardware telemetry binding."
    elif "threshold" in query:
        response += "The energy budget threshold is 100 Joules. Above this, the system transitions to Inversion Regime."
    elif "patch" in query or "ignore" in query or "scored" in query:
        response += f"Plans are scored using weighted utility and cost. Patch (Utility) score: {patch}, Ignore (Cost) score: {ignore}. Weights shift based on Landauer Penalty."
    else:
        response += f"Current regime is {regime}. When energy rises above 100J, cost-aware plans dominate to prevent thermal runaway."

    elapsed = (time.time() - start) * 1000 + random.uniform(50, 200)

    return {
        "response": response,
        "latency_ms": round(elapsed, 2)
    }


if __name__ == "__main__":
    import sys

    data_path = "evaluation/test_data.jsonl"
    output_path = "evaluation/results.json"

    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    print(f"Running evaluation on {data_path}...")
    results = run_evaluation(data_path, output_path)

    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    agg = results['aggregates']
    print(f"Total Samples:        {agg['total_samples']}")
    print(f"Avg Relevance Score:  {agg['avg_relevance_score']:.2f}")
    print(f"Avg Groundedness:     {agg['avg_groundedness_score']:.2f}")
    print(f"Loops Detected:       {agg['loops_detected']}")
    print(f"Avg Latency Score:    {agg['avg_latency_score']:.2f}")
    print(f"Overall Pass Rate:    {agg['pass_rate']:.0%}")
    print("="*50)
