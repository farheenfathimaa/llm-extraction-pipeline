import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
import pandas as pd
import numpy as np
from collections import defaultdict

@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    metric_name: str
    score: float
    details: Dict[str, Any]
    timestamp: str
    notes: Optional[str] = None

class EvaluationMetrics:
    """Evaluation metrics for pipeline outputs"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.evaluation_history = []
    
    def evaluate_pipeline_output(self, 
                                pipeline_results: Dict[str, Any],
                                original_text: str,
                                ground_truth: Optional[Dict[str, Any]] = None) -> Dict[str, EvaluationResult]:
        """Evaluate complete pipeline output"""
        
        evaluation_results = {}
        
        # 1. Completeness check
        evaluation_results['completeness'] = self._evaluate_completeness(pipeline_results)
        
        # 2. Consistency check
        evaluation_results['consistency'] = self._evaluate_consistency(pipeline_results)
        
        # 3. Token efficiency
        evaluation_results['token_efficiency'] = self._evaluate_token_efficiency(pipeline_results, original_text)
        
        # 4. Processing time
        evaluation_results['processing_time'] = self._evaluate_processing_time(pipeline_results)
        
        # 5. Content quality (if ground truth available)
        if ground_truth:
            evaluation_results['content_quality'] = self._evaluate_content_quality(
                pipeline_results, ground_truth
            )
        
        # 6. Cost efficiency
        evaluation_results['cost_efficiency'] = self._evaluate_cost_efficiency(pipeline_results)
        
        # Store evaluation history
        self.evaluation_history.append({
            'timestamp': datetime.now().isoformat(),
            'results': evaluation_results,
            'pipeline_results': pipeline_results
        })
        
        return evaluation_results
    
    def _evaluate_completeness(self, pipeline_results: Dict[str, Any]) -> EvaluationResult:
        """Evaluate completeness of pipeline execution"""
        
        expected_chains = ['summarization', 'key_insights', 'policy_conclusions', 'validation']
        pipeline_data = pipeline_results.get('pipeline_results', {})
        
        completed_chains = []
        failed_chains = []
        
        for chain_name in expected_chains:
            if chain_name in pipeline_data:
                if 'error' not in pipeline_data[chain_name]:
                    completed_chains.append(chain_name)
                else:
                    failed_chains.append(chain_name)
        
        completeness_score = len(completed_chains) / len(expected_chains)
        
        return EvaluationResult(
            metric_name="completeness",
            score=completeness_score,
            details={
                'completed_chains': completed_chains,
                'failed_chains': failed_chains,
                'total_expected': len(expected_chains),
                'total_completed': len(completed_chains)
            },
            timestamp=datetime.now().isoformat(),
            notes=f"Pipeline completed {len(completed_chains)}/{len(expected_chains)} chains"
        )
    
    def _evaluate_consistency(self, pipeline_results: Dict[str, Any]) -> EvaluationResult:
        """Evaluate consistency between different chain outputs"""
        
        pipeline_data = pipeline_results.get('pipeline_results', {})
        consistency_issues = []
        consistency_score = 1.0
        
        # Check if validation chain flagged any issues
        if 'validation' in pipeline_data and 'validation_results' in pipeline_data['validation']:
            validation_results = pipeline_data['validation']['validation_results']
            
            for result in validation_results:
                if not result.get('is_accurate', True):
                    consistency_issues.append({
                        'item': result.get('item'),
                        'issues': result.get('issues', [])
                    })
                    consistency_score -= 0.1  # Reduce score for each inconsistency
        
        consistency_score = max(0.0, consistency_score)
        
        return EvaluationResult(
            metric_name="consistency",
            score=consistency_score,
            details={
                'consistency_issues': consistency_issues,
                'total_issues': len(consistency_issues),
                'validation_available': 'validation' in pipeline_data
            },
            timestamp=datetime.now().isoformat(),
            notes=f"Found {len(consistency_issues)} consistency issues"
        )
    
    def _evaluate_token_efficiency(self, pipeline_results: Dict[str, Any], 
                                 original_text: str) -> EvaluationResult:
        """Evaluate token efficiency of the pipeline"""
        
        pipeline_data = pipeline_results.get('pipeline_results', {})
        total_tokens = 0
        total_cost = 0.0
        
        for chain_name, chain_result in pipeline_data.items():
            if isinstance(chain_result, dict) and 'metadata' in chain_result:
                metadata = chain_result['metadata']
                total_tokens += metadata.get('tokens_used', 0)
                total_cost += metadata.get('cost', 0.0)
        
        # Calculate efficiency metrics
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            input_tokens = len(encoding.encode(original_text))
            
            efficiency_ratio = input_tokens / max(total_tokens, 1)
            cost_per_1k_chars = (total_cost / max(len(original_text), 1)) * 1000
            
        except Exception:
            efficiency_ratio = 0.0
            cost_per_1k_chars = 0.0
            input_tokens = 0
        
        return EvaluationResult(
            metric_name="token_efficiency",
            score=min(1.0, efficiency_ratio),
            details={
                'total_tokens_used': total_tokens,
                'input_tokens': input_tokens,
                'efficiency_ratio': efficiency_ratio,
                'total_cost': total_cost,
                'cost_per_1k_chars': cost_per_1k_chars
            },
            timestamp=datetime.now().isoformat(),
            notes=f"Used {total_tokens} tokens, cost ${total_cost:.4f}"
        )
    
    def _evaluate_processing_time(self, pipeline_results: Dict[str, Any]) -> EvaluationResult:
        """Evaluate processing time efficiency"""
        
        pipeline_data = pipeline_results.get('pipeline_results', {})
        chain_times = []
        
        # Extract timestamps from chain results
        for chain_name, chain_result in pipeline_data.items():
            if isinstance(chain_result, dict) and 'metadata' in chain_result:
                timestamp = chain_result['metadata'].get('timestamp')
                if timestamp:
                    chain_times.append(timestamp)
        
        if len(chain_times) >= 2:
            try:
                start_time = datetime.fromisoformat(min(chain_times))
                end_time = datetime.fromisoformat(max(chain_times))
                total_time = (end_time - start_time).total_seconds()
                
                # Score based on processing time (lower is better)
                # Assume good performance is under 30 seconds
                time_score = max(0.0, 1.0 - (total_time / 30.0))
                
            except Exception:
                total_time = 0.0
                time_score = 0.0
        else:
            total_time = 0.0
            time_score = 0.0
        
        return EvaluationResult(
            metric_name="processing_time",
            score=time_score,
            details={
                'total_processing_time': total_time,
                'chain_count': len(chain_times),
                'avg_time_per_chain': total_time / max(len(chain_times), 1)
            },
            timestamp=datetime.now().isoformat(),
            notes=f"Processed in {total_time:.2f} seconds"
        )
    
    def _evaluate_content_quality(self, pipeline_results: Dict[str, Any], 
                                ground_truth: Dict[str, Any]) -> EvaluationResult:
        """Evaluate content quality against ground truth"""
        
        pipeline_data = pipeline_results.get('pipeline_results', {})
        quality_scores = []
        
        # Compare summarization
        if 'summarization' in pipeline_data and 'summary' in ground_truth:
            summary_score = self._compare_summaries(
                pipeline_data['summarization'], ground_truth['summary']
            )
            quality_scores.append(summary_score)
        
        # Compare key insights
        if 'key_insights' in pipeline_data and 'insights' in ground_truth:
            insights_score = self._compare_insights(
                pipeline_data['key_insights'], ground_truth['insights']
            )
            quality_scores.append(insights_score)
        
        # Compare policy conclusions
        if 'policy_conclusions' in pipeline_data and 'conclusions' in ground_truth:
            conclusions_score = self._compare_conclusions(
                pipeline_data['policy_conclusions'], ground_truth['conclusions']
            )
            quality_scores.append(conclusions_score)
        
        overall_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        return EvaluationResult(
            metric_name="content_quality",
            score=overall_quality,
            details={
                'component_scores': quality_scores,
                'comparisons_made': len(quality_scores)
            },
            timestamp=datetime.now().isoformat(),
            notes=f"Content quality based on {len(quality_scores)} comparisons"
        )
    
    def _evaluate_cost_efficiency(self, pipeline_results: Dict[str, Any]) -> EvaluationResult:
        """Evaluate cost efficiency of the pipeline"""
        
        pipeline_data = pipeline_results.get('pipeline_results', {})
        total_cost = 0.0
        successful_chains = 0
        
        for chain_name, chain_result in pipeline_data.items():
            if isinstance(chain_result, dict) and 'metadata' in chain_result:
                total_cost += chain_result['metadata'].get('cost', 0.0)
                if 'error' not in chain_result:
                    successful_chains += 1
        
        # Calculate cost efficiency (successful chains per dollar)
        cost_efficiency = successful_chains / max(total_cost, 0.001)
        
        # Normalize to 0-1 scale (assume good efficiency is 100 successful chains per dollar)
        efficiency_score = min(1.0, cost_efficiency / 100.0)
        
        return EvaluationResult(
            metric_name="cost_efficiency",
            score=efficiency_score,
            details={
                'total_cost': total_cost,
                'successful_chains': successful_chains,
                'cost_per_successful_chain': total_cost / max(successful_chains, 1),
                'efficiency_ratio': cost_efficiency
            },
            timestamp=datetime.now().isoformat(),
            notes=f"Cost efficiency: {cost_efficiency:.2f} successful chains per dollar"
        )
    
    def _compare_summaries(self, generated_summary: Dict[str, Any], 
                          ground_truth_summary: str) -> float:
        """Compare generated summary with ground truth"""
        
        generated_text = generated_summary.get('content', '')
        if isinstance(generated_text, dict):
            generated_text = generated_text.get('content', '')
        
        # Simple similarity based on common words
        generated_words = set(generated_text.lower().split())
        ground_truth_words = set(ground_truth_summary.lower().split())
        
        if not generated_words or not ground_truth_words:
            return 0.0
        
        intersection = generated_words.intersection(ground_truth_words)
        union = generated_words.union(ground_truth_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _compare_insights(self, generated_insights: Dict[str, Any], 
                         ground_truth_insights: List[str]) -> float:
        """Compare generated insights with ground truth"""
        
        # Extract insights from generated content
        generated_content = generated_insights.get('content', {})
        if isinstance(generated_content, str):
            try:
                generated_content = json.loads(generated_content)
            except:
                return 0.0
        
        generated_insights_list = generated_content.get('insights', [])
        
        if not generated_insights_list or not ground_truth_insights:
            return 0.0
        
        # Calculate similarity between insight lists
        total_similarity = 0.0
        comparisons = 0
        
        for generated_insight in generated_insights_list:
            insight_text = generated_insight.get('insight', '') if isinstance(generated_insight, dict) else str(generated_insight)
            
            best_similarity = 0.0
            for ground_truth_insight in ground_truth_insights:
                similarity = self._text_similarity(insight_text, ground_truth_insight)
                best_similarity = max(best_similarity, similarity)
            
            total_similarity += best_similarity
            comparisons += 1
        
        return total_similarity / comparisons if comparisons > 0 else 0.0
    
    def _compare_conclusions(self, generated_conclusions: Dict[str, Any], 
                           ground_truth_conclusions: List[str]) -> float:
        """Compare generated policy conclusions with ground truth"""
        
        generated_content = generated_conclusions.get('content', {})
        if isinstance(generated_content, str):
            try:
                generated_content = json.loads(generated_content)
            except:
                return 0.0
        
        generated_conclusions_list = generated_content.get('policy_conclusions', [])
        
        if not generated_conclusions_list or not ground_truth_conclusions:
            return 0.0
        
        # Calculate similarity between conclusion lists
        total_similarity = 0.0
        comparisons = 0
        
        for generated_conclusion in generated_conclusions_list:
            conclusion_text = generated_conclusion.get('conclusion', '') if isinstance(generated_conclusion, dict) else str(generated_conclusion)
            
            best_similarity = 0.0
            for ground_truth_conclusion in ground_truth_conclusions:
                similarity = self._text_similarity(conclusion_text, ground_truth_conclusion)
                best_similarity = max(best_similarity, similarity)
            
            total_similarity += best_similarity
            comparisons += 1
        
        return total_similarity / comparisons if comparisons > 0 else 0.0
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, EvaluationResult]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_score': np.mean([result.score for result in evaluation_results.values()]),
            'metric_scores': {name: result.score for name, result in evaluation_results.items()},
            'detailed_results': {name: result.details for name, result in evaluation_results.items()},
            'recommendations': self._generate_recommendations(evaluation_results)
        }
        
        return report
    
    def _generate_recommendations(self, evaluation_results: Dict[str, EvaluationResult]) -> List[str]:
        """Generate recommendations based on evaluation results"""
        
        recommendations = []
        
        # Check completeness
        if evaluation_results.get('completeness') and evaluation_results['completeness'].score < 0.8:
            recommendations.append("Consider improving pipeline robustness - some chains are failing")
        
        # Check consistency
        if evaluation_results.get('consistency') and evaluation_results['consistency'].score < 0.8:
            recommendations.append("Review chain outputs for consistency issues")
        
        # Check token efficiency
        if evaluation_results.get('token_efficiency') and evaluation_results['token_efficiency'].score < 0.5:
            recommendations.append("Optimize prompts to reduce token usage")
        
        # Check processing time
        if evaluation_results.get('processing_time') and evaluation_results['processing_time'].score < 0.6:
            recommendations.append("Consider optimizing processing time or using faster models")
        
        # Check cost efficiency
        if evaluation_results.get('cost_efficiency') and evaluation_results['cost_efficiency'].score < 0.5:
            recommendations.append("Review cost optimization strategies")
        
        return recommendations
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """Get history of all evaluations"""
        return self.evaluation_history
    
    def export_evaluation_results(self, evaluation_results: Dict[str, EvaluationResult], 
                                 filename: str = None) -> str:
        """Export evaluation results to JSON file"""
        
        if filename is None:
            filename = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert EvaluationResult objects to dictionaries
        exportable_results = {}
        for name, result in evaluation_results.items():
            exportable_results[name] = {
                'metric_name': result.metric_name,
                'score': result.score,
                'details': result.details,
                'timestamp': result.timestamp,
                'notes': result.notes
            }
        
        with open(filename, 'w') as f:
            json.dump(exportable_results, f, indent=2)
        
        return filename