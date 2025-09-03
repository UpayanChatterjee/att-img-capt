"""
Evaluation metrics for image captioning.
This module implements BLEU, CIDEr, ROUGE, and other metrics for caption evaluation.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
import math
import re
from pathlib import Path
import json


class BLEUScore:
    """
    Implementation of BLEU (Bilingual Evaluation Understudy) score.
    Reference: "BLEU: a Method for Automatic Evaluation of Machine Translation"
    """
    
    def __init__(self, max_n: int = 4, smooth: bool = True):
        """
        Initialize BLEU score calculator.
        
        Args:
            max_n: Maximum n-gram to consider
            smooth: Whether to apply smoothing for zero counts
        """
        self.max_n = max_n
        self.smooth = smooth
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-grams from token list."""
        if n > len(tokens):
            return Counter()
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i + n]))
        
        return Counter(ngrams)
    
    def _compute_precision(self, 
                          candidate: List[str], 
                          references: List[List[str]], 
                          n: int) -> Tuple[int, int]:
        """
        Compute n-gram precision.
        
        Args:
            candidate: Candidate tokens
            references: List of reference token lists
            n: n-gram size
            
        Returns:
            Tuple of (matches, total)
        """
        candidate_ngrams = self._get_ngrams(candidate, n)
        
        if not candidate_ngrams:
            return 0, 0
        
        # Get maximum count for each n-gram across all references
        max_ref_counts = Counter()
        for reference in references:
            ref_ngrams = self._get_ngrams(reference, n)
            for ngram in ref_ngrams:
                max_ref_counts[ngram] = max(max_ref_counts[ngram], ref_ngrams[ngram])
        
        # Count matches
        matches = 0
        for ngram, count in candidate_ngrams.items():
            matches += min(count, max_ref_counts[ngram])
        
        total = sum(candidate_ngrams.values())
        
        return matches, total
    
    def _brevity_penalty(self, candidate: List[str], references: List[List[str]]) -> float:
        """
        Compute brevity penalty.
        
        Args:
            candidate: Candidate tokens
            references: List of reference token lists
            
        Returns:
            Brevity penalty
        """
        candidate_len = len(candidate)
        
        # Find the reference length closest to candidate length
        ref_lengths = [len(ref) for ref in references]
        closest_ref_len = min(ref_lengths, key=lambda x: abs(x - candidate_len))
        
        if candidate_len > closest_ref_len:
            return 1.0
        else:
            return math.exp(1 - closest_ref_len / candidate_len) if candidate_len > 0 else 0.0
    
    def compute_bleu(self, 
                    candidate: List[str], 
                    references: List[List[str]]) -> Dict[str, float]:
        """
        Compute BLEU score for a single candidate against references.
        
        Args:
            candidate: Candidate tokens
            references: List of reference token lists
            
        Returns:
            Dictionary with BLEU scores
        """
        if not candidate or not references:
            return {f'bleu_{i+1}': 0.0 for i in range(self.max_n)}
        
        # Compute precision for each n-gram
        precisions = []
        for n in range(1, self.max_n + 1):
            matches, total = self._compute_precision(candidate, references, n)
            
            if total == 0:
                precision = 0.0
            elif matches == 0:
                # Apply smoothing for zero matches
                precision = 1.0 / total if self.smooth else 0.0
            else:
                precision = matches / total
            
            precisions.append(precision)
        
        # Compute brevity penalty
        bp = self._brevity_penalty(candidate, references)
        
        # Compute BLEU scores
        bleu_scores = {}
        for n in range(1, self.max_n + 1):
            if any(p == 0 for p in precisions[:n]):
                bleu_score = 0.0
            else:
                # Geometric mean of precisions
                log_precision_sum = sum(math.log(p) for p in precisions[:n])
                bleu_score = bp * math.exp(log_precision_sum / n)
            
            bleu_scores[f'bleu_{n}'] = bleu_score
        
        return bleu_scores
    
    def corpus_bleu(self, 
                   candidates: List[List[str]], 
                   references: List[List[List[str]]]) -> Dict[str, float]:
        """
        Compute corpus-level BLEU score.
        
        Args:
            candidates: List of candidate token lists
            references: List of reference lists (each containing multiple references)
            
        Returns:
            Dictionary with corpus BLEU scores
        """
        assert len(candidates) == len(references)
        
        # Accumulate counts across corpus
        total_matches = [0] * self.max_n
        total_counts = [0] * self.max_n
        total_candidate_len = 0
        total_ref_len = 0
        
        for candidate, refs in zip(candidates, references):
            # Update n-gram counts
            for n in range(1, self.max_n + 1):
                matches, total = self._compute_precision(candidate, refs, n)
                total_matches[n-1] += matches
                total_counts[n-1] += total
            
            # Update length counts for brevity penalty
            candidate_len = len(candidate)
            ref_lengths = [len(ref) for ref in refs]
            closest_ref_len = min(ref_lengths, key=lambda x: abs(x - candidate_len))
            
            total_candidate_len += candidate_len
            total_ref_len += closest_ref_len
        
        # Compute corpus-level precisions
        precisions = []
        for n in range(self.max_n):
            if total_counts[n] == 0:
                precision = 0.0
            elif total_matches[n] == 0:
                precision = 1.0 / total_counts[n] if self.smooth else 0.0
            else:
                precision = total_matches[n] / total_counts[n]
            precisions.append(precision)
        
        # Compute corpus-level brevity penalty
        if total_candidate_len > total_ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - total_ref_len / total_candidate_len) if total_candidate_len > 0 else 0.0
        
        # Compute corpus BLEU scores
        bleu_scores = {}
        for n in range(1, self.max_n + 1):
            if any(p == 0 for p in precisions[:n]):
                bleu_score = 0.0
            else:
                log_precision_sum = sum(math.log(p) for p in precisions[:n])
                bleu_score = bp * math.exp(log_precision_sum / n)
            
            bleu_scores[f'bleu_{n}'] = bleu_score
        
        return bleu_scores


class ROUGEScore:
    """
    Implementation of ROUGE (Recall-Oriented Understudy for Gisting Evaluation) score.
    """
    
    def __init__(self, max_n: int = 2):
        """
        Initialize ROUGE score calculator.
        
        Args:
            max_n: Maximum n-gram to consider
        """
        self.max_n = max_n
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-grams from token list."""
        if n > len(tokens):
            return Counter()
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i + n]))
        
        return Counter(ngrams)
    
    def compute_rouge_n(self, 
                       candidate: List[str], 
                       references: List[List[str]], 
                       n: int) -> Dict[str, float]:
        """
        Compute ROUGE-N score.
        
        Args:
            candidate: Candidate tokens
            references: List of reference token lists
            n: n-gram size
            
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        if not candidate or not references:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        candidate_ngrams = self._get_ngrams(candidate, n)
        
        # Combine all reference n-grams
        all_ref_ngrams = Counter()
        for reference in references:
            ref_ngrams = self._get_ngrams(reference, n)
            all_ref_ngrams.update(ref_ngrams)
        
        if not candidate_ngrams or not all_ref_ngrams:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Count overlapping n-grams
        overlap = 0
        for ngram in candidate_ngrams:
            overlap += min(candidate_ngrams[ngram], all_ref_ngrams[ngram])
        
        # Compute precision and recall
        precision = overlap / sum(candidate_ngrams.values())
        recall = overlap / sum(all_ref_ngrams.values())
        
        # Compute F1 score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def compute_rouge_l(self, 
                       candidate: List[str], 
                       references: List[List[str]]) -> Dict[str, float]:
        """
        Compute ROUGE-L score based on longest common subsequence.
        
        Args:
            candidate: Candidate tokens
            references: List of reference token lists
            
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        if not candidate or not references:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        def lcs_length(seq1: List[str], seq2: List[str]) -> int:
            """Compute longest common subsequence length."""
            m, n = len(seq1), len(seq2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if seq1[i-1] == seq2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        # Find the best LCS with any reference
        best_lcs = 0
        best_ref_len = 0
        
        for reference in references:
            lcs_len = lcs_length(candidate, reference)
            if lcs_len > best_lcs:
                best_lcs = lcs_len
                best_ref_len = len(reference)
        
        if best_lcs == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Compute precision and recall
        precision = best_lcs / len(candidate)
        recall = best_lcs / best_ref_len
        
        # Compute F1 score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


class CIDErScore:
    """
    Implementation of CIDEr (Consensus-based Image Description Evaluation) score.
    Reference: "CIDEr: Consensus-based Image Description Evaluation"
    """
    
    def __init__(self, max_n: int = 4, sigma: float = 6.0):
        """
        Initialize CIDEr score calculator.
        
        Args:
            max_n: Maximum n-gram to consider
            sigma: Gaussian standard deviation for length penalty
        """
        self.max_n = max_n
        self.sigma = sigma
        self.document_frequencies = None
        self.ref_lengths = None
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-grams from token list."""
        if n > len(tokens):
            return Counter()
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(' '.join(tokens[i:i + n]))
        
        return Counter(ngrams)
    
    def compute_document_frequencies(self, 
                                   all_references: List[List[List[str]]]) -> Dict[int, Dict[str, int]]:
        """
        Compute document frequencies for all n-grams in the reference corpus.
        
        Args:
            all_references: List of reference lists for each image
            
        Returns:
            Dictionary mapping n-gram size to document frequencies
        """
        document_frequencies = {n: defaultdict(int) for n in range(1, self.max_n + 1)}
        
        for image_refs in all_references:
            for n in range(1, self.max_n + 1):
                # Collect all n-grams from all references for this image
                ngrams_in_image = set()
                for ref in image_refs:
                    ref_ngrams = self._get_ngrams(ref, n)
                    ngrams_in_image.update(ref_ngrams.keys())
                
                # Increment document frequency for each unique n-gram
                for ngram in ngrams_in_image:
                    document_frequencies[n][ngram] += 1
        
        return document_frequencies
    
    def _compute_cider_score(self,
                           candidate: List[str],
                           references: List[List[str]],
                           n: int) -> float:
        """
        Compute CIDEr score for specific n-gram.
        
        Args:
            candidate: Candidate tokens
            references: Reference token lists
            n: n-gram size
            
        Returns:
            CIDEr score for n-gram
        """
        if not self.document_frequencies:
            raise ValueError("Document frequencies not computed. Call compute_document_frequencies first.")
        
        candidate_ngrams = self._get_ngrams(candidate, n)
        
        # Compute TF-IDF vectors
        candidate_tfidf = {}
        ref_tfidfs = []
        
        # Total number of images in the corpus
        total_images = len(self.ref_lengths)
        
        # Candidate TF-IDF
        candidate_len = len(candidate)
        for ngram, count in candidate_ngrams.items():
            tf = count / candidate_len if candidate_len > 0 else 0
            df = self.document_frequencies[n][ngram]
            idf = math.log(total_images / max(1, df))
            candidate_tfidf[ngram] = tf * idf
        
        # Reference TF-IDFs
        for ref in references:
            ref_ngrams = self._get_ngrams(ref, n)
            ref_tfidf = {}
            ref_len = len(ref)
            
            for ngram, count in ref_ngrams.items():
                tf = count / ref_len if ref_len > 0 else 0
                df = self.document_frequencies[n][ngram]
                idf = math.log(total_images / max(1, df))
                ref_tfidf[ngram] = tf * idf
            
            ref_tfidfs.append(ref_tfidf)
        
        # Compute cosine similarity with each reference
        scores = []
        for ref_tfidf in ref_tfidfs:
            # Dot product
            dot_product = 0.0
            for ngram in set(candidate_tfidf.keys()) | set(ref_tfidf.keys()):
                dot_product += candidate_tfidf.get(ngram, 0) * ref_tfidf.get(ngram, 0)
            
            # Norms
            candidate_norm = math.sqrt(sum(v**2 for v in candidate_tfidf.values()))
            ref_norm = math.sqrt(sum(v**2 for v in ref_tfidf.values()))
            
            # Cosine similarity
            if candidate_norm > 0 and ref_norm > 0:
                similarity = dot_product / (candidate_norm * ref_norm)
            else:
                similarity = 0.0
            
            scores.append(similarity)
        
        # Average over all references
        return np.mean(scores) if scores else 0.0
    
    def compute_cider(self,
                     candidate: List[str],
                     references: List[List[str]]) -> float:
        """
        Compute CIDEr score for a candidate against references.
        
        Args:
            candidate: Candidate tokens
            references: Reference token lists
            
        Returns:
            CIDEr score
        """
        if not candidate or not references:
            return 0.0
        
        cider_scores = []
        
        # Compute CIDEr for each n-gram
        for n in range(1, self.max_n + 1):
            score = self._compute_cider_score(candidate, references, n)
            cider_scores.append(score)
        
        # Average across n-grams
        return np.mean(cider_scores)


class CaptionEvaluator:
    """
    Comprehensive caption evaluator that computes multiple metrics.
    """
    
    def __init__(self, 
                 metrics: List[str] = ['bleu', 'rouge', 'cider'],
                 bleu_max_n: int = 4,
                 rouge_max_n: int = 2):
        """
        Initialize caption evaluator.
        
        Args:
            metrics: List of metrics to compute
            bleu_max_n: Maximum n-gram for BLEU
            rouge_max_n: Maximum n-gram for ROUGE
        """
        self.metrics = metrics
        
        # Initialize metric calculators
        self.bleu = BLEUScore(max_n=bleu_max_n) if 'bleu' in metrics else None
        self.rouge = ROUGEScore(max_n=rouge_max_n) if 'rouge' in metrics else None
        self.cider = CIDErScore() if 'cider' in metrics else None
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for evaluation.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split into tokens
        tokens = text.split()
        
        # Remove empty tokens
        tokens = [token for token in tokens if token]
        
        return tokens
    
    def prepare_corpus_for_cider(self, 
                                all_references: List[List[str]]) -> None:
        """
        Prepare corpus for CIDEr evaluation.
        
        Args:
            all_references: All reference captions in the corpus
        """
        if self.cider is None:
            return
        
        # Preprocess all references
        processed_refs = []
        ref_lengths = []
        
        for ref_list in all_references:
            processed_ref_list = []
            for ref in ref_list:
                tokens = self.preprocess_text(ref)
                processed_ref_list.append(tokens)
                ref_lengths.append(len(tokens))
            processed_refs.append(processed_ref_list)
        
        # Compute document frequencies
        self.cider.document_frequencies = self.cider.compute_document_frequencies(processed_refs)
        self.cider.ref_lengths = ref_lengths
    
    def evaluate_caption(self,
                        candidate: str,
                        references: List[str]) -> Dict[str, Any]:
        """
        Evaluate a single caption against references.
        
        Args:
            candidate: Candidate caption
            references: List of reference captions
            
        Returns:
            Dictionary of evaluation scores
        """
        # Preprocess texts
        candidate_tokens = self.preprocess_text(candidate)
        reference_tokens = [self.preprocess_text(ref) for ref in references]
        
        results = {}
        
        # Compute BLEU scores
        if self.bleu is not None:
            bleu_scores = self.bleu.compute_bleu(candidate_tokens, reference_tokens)
            results.update(bleu_scores)
        
        # Compute ROUGE scores
        if self.rouge is not None:
            for n in range(1, self.rouge.max_n + 1):
                rouge_scores = self.rouge.compute_rouge_n(candidate_tokens, reference_tokens, n)
                for metric, score in rouge_scores.items():
                    results[f'rouge_{n}_{metric}'] = score
            
            # ROUGE-L
            rouge_l_scores = self.rouge.compute_rouge_l(candidate_tokens, reference_tokens)
            for metric, score in rouge_l_scores.items():
                results[f'rouge_l_{metric}'] = score
        
        # Compute CIDEr score
        if self.cider is not None and self.cider.document_frequencies is not None:
            cider_score = self.cider.compute_cider(candidate_tokens, reference_tokens)
            results['cider'] = cider_score
        
        return results
    
    def evaluate_corpus(self,
                       candidates: List[str],
                       references: List[List[str]]) -> Dict[str, float]:
        """
        Evaluate a corpus of captions.
        
        Args:
            candidates: List of candidate captions
            references: List of reference caption lists
            
        Returns:
            Dictionary of average evaluation scores
        """
        assert len(candidates) == len(references)
        
        # Prepare CIDEr if needed
        if self.cider is not None:
            self.prepare_corpus_for_cider(references)
        
        # Evaluate each caption
        all_scores = []
        for candidate, refs in zip(candidates, references):
            scores = self.evaluate_caption(candidate, refs)
            all_scores.append(scores)
        
        # Compute average scores
        avg_scores = {}
        if all_scores:
            for metric in all_scores[0].keys():
                scores = [score_dict[metric] for score_dict in all_scores]
                avg_scores[metric] = np.mean(scores)
        
        # Compute corpus-level BLEU if available
        if self.bleu is not None:
            candidate_tokens = [self.preprocess_text(cand) for cand in candidates]
            reference_tokens = [[self.preprocess_text(ref) for ref in refs] for refs in references]
            
            corpus_bleu = self.bleu.corpus_bleu(candidate_tokens, reference_tokens)
            for metric, score in corpus_bleu.items():
                avg_scores[f'corpus_{metric}'] = score
        
        return avg_scores


def test_evaluation_metrics():
    """Test function for evaluation metrics."""
    print("Testing evaluation metrics...")
    
    # Sample data
    candidate = "a man is riding a bike on the street"
    references = [
        "a person is riding a bicycle on a road",
        "a man rides his bike down the street",
        "someone is cycling on the street"
    ]
    
    # Test individual metrics
    print("\n=== Testing BLEU ===")
    bleu = BLEUScore()
    candidate_tokens = candidate.split()
    reference_tokens = [ref.split() for ref in references]
    bleu_scores = bleu.compute_bleu(candidate_tokens, reference_tokens)
    for metric, score in bleu_scores.items():
        print(f"{metric}: {score:.4f}")
    
    print("\n=== Testing ROUGE ===")
    rouge = ROUGEScore()
    rouge_1 = rouge.compute_rouge_n(candidate_tokens, reference_tokens, 1)
    rouge_l = rouge.compute_rouge_l(candidate_tokens, reference_tokens)
    print(f"ROUGE-1 F1: {rouge_1['f1']:.4f}")
    print(f"ROUGE-L F1: {rouge_l['f1']:.4f}")
    
    print("\n=== Testing Caption Evaluator ===")
    evaluator = CaptionEvaluator()
    scores = evaluator.evaluate_caption(candidate, references)
    
    print("Evaluation scores:")
    for metric, score in scores.items():
        print(f"  {metric}: {score:.4f}")
    
    # Test corpus evaluation
    print("\n=== Testing Corpus Evaluation ===")
    candidates = [candidate, "a dog is playing in the park"]
    all_references = [references, ["a dog plays in a park", "the dog is playing outside"]]
    
    corpus_scores = evaluator.evaluate_corpus(candidates, all_references)
    print("Corpus evaluation scores:")
    for metric, score in corpus_scores.items():
        print(f"  {metric}: {score:.4f}")
    
    print("\nEvaluation metrics test completed!")


if __name__ == "__main__":
    test_evaluation_metrics()