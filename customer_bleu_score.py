from collections import Counter
import math

def get_ngrams(tokens, n):
    """Extract n-grams from a list of tokens."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def calculate_bleu(references, candidate, max_n=4, weights=None, smoothing=False):
    """
    Calculate BLEU score for a candidate sentence against reference sentences.
    
    Args:
        references (list): List of reference sentences (tokenized as lists of words).
        candidate (list): Candidate sentence (tokenized as list of words).
        max_n (int): Maximum n-gram order (default=4).
        weights (tuple): Weights for n-grams (default: equal weights).
        smoothing (bool): Apply simple smoothing for zero precisions.
    
    Returns:
        float: BLEU score.
    """
    if not candidate or not references:
        return 0.0
    
    # Set default weights
    if weights is None:
        weights = (1.0 / max_n,) * max_n
    
    # Calculate candidate length
    c = len(candidate)
    
    # Find effective reference length (closest to candidate length)
    r = min((len(ref) for ref in references), key=lambda x: abs(x - c))
    
    # Calculate brevity penalty
    bp = 1.0 if c > r else math.exp(1 - r / c) if c > 0 else 0.0
    
    # Calculate n-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        candidate_ngrams = Counter(get_ngrams(candidate, n))
        if not candidate_ngrams:
            precisions.append(0.0)
            continue
        
        # Get maximum n-gram counts from references
        max_ref_counts = Counter()
        for ref in references:
            ref_ngrams = Counter(get_ngrams(ref, n))
            for ngram in ref_ngrams:
                max_ref_counts[ngram] = max(max_ref_counts[ngram], ref_ngrams[ngram])
        
        # Calculate clipped counts
        clipped_count = sum(min(count, max_ref_counts[ngram]) for ngram, count in candidate_ngrams.items())
        total_count = sum(candidate_ngrams.values())
        precision = clipped_count / total_count if total_count > 0 else 0.0
        
        # Apply smoothing if precision is 0
        if precision == 0.0 and smoothing:
            precision = 1e-5  # Small constant to avoid zero
        precisions.append(precision)
    
    # Calculate geometric mean of precisions
    if all(p == 0.0 for p in precisions):
        return 0.0
    
    log_sum = sum(w * math.log(p) for w, p in zip(weights, precisions) if p > 0)
    bleu = bp * math.exp(log_sum)
    
    return bleu

# Example usage
references = [
    ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
    ["the", "fast", "brown", "fox", "leaps", "over", "the", "idle", "dog"]
]
candidate = ["the", "quick", "brown", "fox", "jumps", "over", "a", "lazy", "dog"]

bleu_score = calculate_bleu(references, candidate, max_n=4, smoothing=True)
print(f"Custom BLEU Score: {bleu_score:.4f}")
