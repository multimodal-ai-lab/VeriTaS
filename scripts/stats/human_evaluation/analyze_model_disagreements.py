#!/usr/bin/env python3
"""
Analyze disagreement patterns between different ensemble members (models).

This script examines inter-model agreement within the automated ensemble,
similar to how analyze_disagreements.py examines inter-annotator agreement.

Key analyses:
1. Inter-model agreement across all predictions
2. Inter-model agreement in cases where ensemble disagrees with humans
3. Model-specific tendencies and patterns
4. Correlation between model uncertainty and human disagreement

Usage:
    python scripts/analyze_model_disagreements.py comparison_results.json
    python scripts/analyze_model_disagreements.py comparison_results.json -o model_analysis.json
    python scripts/analyze_model_disagreements.py comparison_results.json --detailed
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_results(results_path: Path) -> dict:
    """Load comparison results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def calculate_inter_model_agreement(model_labels: list[dict]) -> dict:
    """
    Calculate inter-model agreement for a specific property.

    Args:
        model_labels: List of {'model': str, 'tendency': float} dicts

    Returns:
        - agreement_rate: proportion of model pairs that agree on polarity
        - num_models: number of models
        - all_agree: whether all models agree
        - tendency_variance: variance in tendency values
        - tendencies: list of tendency values
    """
    if not model_labels or len(model_labels) < 2:
        return {
            'num_models': len(model_labels) if model_labels else 0,
            'agreement_rate': None,
            'all_agree': None,
            'tendency_variance': None,
            'tendencies': [m['tendency'] for m in model_labels] if model_labels else [],
            'models': [m['model'] for m in model_labels] if model_labels else [],
        }

    tendencies = [m['tendency'] for m in model_labels]
    models = [m['model'] for m in model_labels]

    # Categorize each model's tendency
    def categorize(tendency: float) -> str:
        if tendency > 0.1:
            return 'positive'
        elif tendency < -0.1:
            return 'negative'
        else:
            return 'neutral'

    categories = [categorize(t) for t in tendencies]

    # Calculate pairwise agreement
    agreements = 0
    total_pairs = 0

    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            total_pairs += 1
            if categories[i] == categories[j]:
                agreements += 1

    agreement_rate = agreements / total_pairs if total_pairs > 0 else None

    # Check if all agree
    all_agree = len(set(categories)) == 1

    # Calculate variance in tendencies
    mean_tendency = sum(tendencies) / len(tendencies)
    variance = sum((t - mean_tendency) ** 2 for t in tendencies) / len(tendencies)

    return {
        'num_models': len(model_labels),
        'agreement_rate': agreement_rate,
        'all_agree': all_agree,
        'tendency_variance': variance,
        'tendencies': tendencies,
        'models': models,
        'categories': categories,
    }


def analyze_inter_model_agreement_overall(data: dict) -> dict:
    """
    Analyze inter-model agreement across all claims.

    Returns statistics on how often models agree with each other.
    """
    properties = ['clarity', 'veracity', 'context_coverage', 'intent']
    stats = {}

    for prop in properties:
        agreement_rates = []
        variances = []
        all_agree_count = 0
        total_claims = 0

        for claim_comp in data['per_claim_comparisons']:
            if not claim_comp['verdict_available']:
                continue

            # Get aggregated comparison
            agg_comp = claim_comp.get('aggregated_comparison', {})
            if prop not in agg_comp:
                continue

            comp = agg_comp[prop]
            model_labels = comp.get('automated_individual_models')

            if not model_labels or len(model_labels) < 2:
                continue

            total_claims += 1
            inter_model = calculate_inter_model_agreement(model_labels)

            if inter_model['agreement_rate'] is not None:
                agreement_rates.append(inter_model['agreement_rate'])

            if inter_model['tendency_variance'] is not None:
                variances.append(inter_model['tendency_variance'])

            if inter_model['all_agree']:
                all_agree_count += 1

        stats[prop] = {
            'total_claims': total_claims,
            'avg_agreement_rate': sum(agreement_rates) / len(agreement_rates) if agreement_rates else None,
            'avg_tendency_variance': sum(variances) / len(variances) if variances else None,
            'all_agree_count': all_agree_count,
            'all_agree_pct': all_agree_count / total_claims if total_claims > 0 else 0,
        }

    return stats


def analyze_model_agreement_by_human_agreement(data: dict) -> dict:
    """
    Compare inter-model agreement in cases where ensemble agrees vs disagrees with humans.

    This answers: Are models more uncertain (disagree more) when the ensemble is wrong?
    """
    properties = ['clarity', 'veracity', 'context_coverage', 'intent']
    stats = {}

    for prop in properties:
        agreement_cases = []  # Cases where ensemble agrees with humans
        disagreement_cases = []  # Cases where ensemble disagrees with humans

        for claim_comp in data['per_claim_comparisons']:
            if not claim_comp['verdict_available']:
                continue

            agg_comp = claim_comp.get('aggregated_comparison', {})
            if prop not in agg_comp:
                continue

            comp = agg_comp[prop]
            model_labels = comp.get('automated_individual_models')

            if not model_labels or len(model_labels) < 2:
                continue

            inter_model = calculate_inter_model_agreement(model_labels)

            # Categorize by whether ensemble agrees with humans
            if comp.get('agreement') is True:
                agreement_cases.append(inter_model)
            elif comp.get('agreement') is False:
                disagreement_cases.append(inter_model)

        # Calculate stats for agreement cases
        if agreement_cases:
            agree_rates = [c['agreement_rate'] for c in agreement_cases if c['agreement_rate'] is not None]
            agree_variances = [c['tendency_variance'] for c in agreement_cases if c['tendency_variance'] is not None]
            all_agree_count = sum(1 for c in agreement_cases if c['all_agree'])
        else:
            agree_rates = []
            agree_variances = []
            all_agree_count = 0

        # Calculate stats for disagreement cases
        if disagreement_cases:
            disagree_rates = [c['agreement_rate'] for c in disagreement_cases if c['agreement_rate'] is not None]
            disagree_variances = [c['tendency_variance'] for c in disagreement_cases if c['tendency_variance'] is not None]
            all_agree_count_disagree = sum(1 for c in disagreement_cases if c['all_agree'])
        else:
            disagree_rates = []
            disagree_variances = []
            all_agree_count_disagree = 0

        stats[prop] = {
            'agreement_cases': {
                'count': len(agreement_cases),
                'avg_model_agreement': sum(agree_rates) / len(agree_rates) if agree_rates else None,
                'avg_variance': sum(agree_variances) / len(agree_variances) if agree_variances else None,
                'all_models_agree_count': all_agree_count,
                'all_models_agree_pct': all_agree_count / len(agreement_cases) if agreement_cases else 0,
            },
            'disagreement_cases': {
                'count': len(disagreement_cases),
                'avg_model_agreement': sum(disagree_rates) / len(disagree_rates) if disagree_rates else None,
                'avg_variance': sum(disagree_variances) / len(disagree_variances) if disagree_variances else None,
                'all_models_agree_count': all_agree_count_disagree,
                'all_models_agree_pct': all_agree_count_disagree / len(disagreement_cases) if disagreement_cases else 0,
            }
        }

    return stats


def analyze_model_agreement_in_disagreement_cases(data: dict, disagreement_analysis: dict = None) -> dict:
    """
    Analyze inter-model agreement specifically in cases where the ensemble
    disagrees with aggregated human annotations.

    This answers: When the ensemble is wrong, do individual models disagree?
    """
    properties = ['clarity', 'veracity', 'context_coverage', 'intent']
    stats = {}

    for prop in properties:
        agreement_rates_in_disagreements = []
        variances_in_disagreements = []
        all_models_agree_count = 0
        split_model_decisions = 0
        total_disagreements = 0

        for claim_comp in data['per_claim_comparisons']:
            if not claim_comp['verdict_available']:
                continue

            # Get aggregated comparison
            agg_comp = claim_comp.get('aggregated_comparison', {})
            if prop not in agg_comp:
                continue

            comp = agg_comp[prop]

            # Only look at cases where ensemble disagrees with humans
            if comp.get('agreement') is not False:
                continue

            total_disagreements += 1

            model_labels = comp.get('automated_individual_models')
            if not model_labels or len(model_labels) < 2:
                continue

            inter_model = calculate_inter_model_agreement(model_labels)

            if inter_model['agreement_rate'] is not None:
                agreement_rates_in_disagreements.append(inter_model['agreement_rate'])

            if inter_model['tendency_variance'] is not None:
                variances_in_disagreements.append(inter_model['tendency_variance'])

            if inter_model['all_agree']:
                all_models_agree_count += 1
            elif not inter_model['all_agree'] and inter_model['num_models'] > 1:
                split_model_decisions += 1

        stats[prop] = {
            'total_disagreements_with_humans': total_disagreements,
            'avg_model_agreement_rate': sum(agreement_rates_in_disagreements) / len(agreement_rates_in_disagreements) if agreement_rates_in_disagreements else None,
            'avg_tendency_variance': sum(variances_in_disagreements) / len(variances_in_disagreements) if variances_in_disagreements else None,
            'all_models_agree_count': all_models_agree_count,
            'all_models_agree_pct': all_models_agree_count / total_disagreements if total_disagreements > 0 else 0,
            'split_model_decisions': split_model_decisions,
            'split_model_decisions_pct': split_model_decisions / total_disagreements if total_disagreements > 0 else 0,
        }

    return stats


def analyze_model_specific_patterns(data: dict) -> dict:
    """
    Analyze patterns for individual models.

    Returns:
        Per-model statistics on tendency distributions and agreement with ensemble
    """
    properties = ['clarity', 'veracity', 'context_coverage', 'intent']
    model_stats = defaultdict(lambda: {
        'total_predictions': 0,
        'properties': defaultdict(lambda: {
            'tendencies': [],
            'agrees_with_ensemble': 0,
            'disagrees_with_ensemble': 0,
        }),
    })

    for claim_comp in data['per_claim_comparisons']:
        if not claim_comp['verdict_available']:
            continue

        agg_comp = claim_comp.get('aggregated_comparison', {})

        for prop in properties:
            if prop not in agg_comp:
                continue

            comp = agg_comp[prop]
            ensemble_tendency = comp.get('automated_tendency')
            model_labels = comp.get('automated_individual_models')

            if not model_labels or ensemble_tendency is None:
                continue

            # Categorize ensemble
            def categorize(tendency: float) -> str:
                if tendency > 0.1:
                    return 'positive'
                elif tendency < -0.1:
                    return 'negative'
                else:
                    return 'neutral'

            ensemble_cat = categorize(ensemble_tendency)

            for model_label in model_labels:
                model_name = model_label['model']
                model_tendency = model_label['tendency']
                model_cat = categorize(model_tendency)

                model_stats[model_name]['total_predictions'] += 1
                model_stats[model_name]['properties'][prop]['tendencies'].append(model_tendency)

                if model_cat == ensemble_cat:
                    model_stats[model_name]['properties'][prop]['agrees_with_ensemble'] += 1
                else:
                    model_stats[model_name]['properties'][prop]['disagrees_with_ensemble'] += 1

    # Calculate summary statistics
    for model_name, stats in model_stats.items():
        for prop, prop_stats in stats['properties'].items():
            tendencies = prop_stats['tendencies']
            if tendencies:
                prop_stats['mean_tendency'] = sum(tendencies) / len(tendencies)
                prop_stats['variance'] = sum((t - prop_stats['mean_tendency']) ** 2 for t in tendencies) / len(tendencies)
                prop_stats['mean_abs_tendency'] = sum(abs(t) for t in tendencies) / len(tendencies)

                total = prop_stats['agrees_with_ensemble'] + prop_stats['disagrees_with_ensemble']
                prop_stats['agreement_with_ensemble_rate'] = prop_stats['agrees_with_ensemble'] / total if total > 0 else None

                # Remove raw tendencies from output (too verbose)
                del prop_stats['tendencies']

        stats['properties'] = dict(stats['properties'])

    return dict(model_stats)


def print_overall_summary(stats: dict):
    """Print summary of inter-model agreement across all claims."""
    print("\n" + "="*80)
    print("INTER-MODEL AGREEMENT (ALL CLAIMS)")
    print("="*80)

    for prop, prop_stats in stats.items():
        if prop_stats['total_claims'] == 0:
            continue

        print(f"\n{prop.replace('_', ' ').title()}:")
        print(f"  Total claims analyzed: {prop_stats['total_claims']}")

        if prop_stats['avg_agreement_rate'] is not None:
            print(f"  Avg inter-model agreement rate: {prop_stats['avg_agreement_rate']:.2%}")

        print(f"  Cases where all models agree: {prop_stats['all_agree_count']} ({prop_stats['all_agree_pct']:.1%})")

        if prop_stats['avg_tendency_variance'] is not None:
            print(f"  Avg tendency variance: {prop_stats['avg_tendency_variance']:.3f}")

    print("\n" + "="*80)


def print_disagreement_summary(stats: dict):
    """Print summary of inter-model agreement in disagreement cases."""
    print("\n" + "="*80)
    print("INTER-MODEL AGREEMENT (WHEN ENSEMBLE DISAGREES WITH HUMANS)")
    print("="*80)

    for prop, prop_stats in stats.items():
        if prop_stats['total_disagreements_with_humans'] == 0:
            continue

        print(f"\n{prop.replace('_', ' ').title()}:")
        print(f"  Total disagreements with humans: {prop_stats['total_disagreements_with_humans']}")

        if prop_stats['avg_model_agreement_rate'] is not None:
            print(f"  Avg inter-model agreement rate: {prop_stats['avg_model_agreement_rate']:.2%}")

        print(f"  All models agree (wrong together): {prop_stats['all_models_agree_count']} ({prop_stats['all_models_agree_pct']:.1%})")
        print(f"  Models split (some wrong, some right?): {prop_stats['split_model_decisions']} ({prop_stats['split_model_decisions_pct']:.1%})")

        if prop_stats['avg_tendency_variance'] is not None:
            print(f"  Avg tendency variance: {prop_stats['avg_tendency_variance']:.3f}")

    print("\n" + "="*80)


def print_model_patterns(model_stats: dict):
    """Print per-model pattern analysis."""
    print("\n" + "="*80)
    print("MODEL-SPECIFIC PATTERNS")
    print("="*80)

    properties = ['clarity', 'veracity', 'context_coverage', 'intent']

    for model_name, stats in sorted(model_stats.items()):
        print(f"\n{model_name}:")
        print(f"  Total predictions: {stats['total_predictions']}")

        for prop in properties:
            if prop not in stats['properties']:
                continue

            prop_stats = stats['properties'][prop]
            print(f"\n  {prop.replace('_', ' ').title()}:")

            if 'mean_tendency' in prop_stats:
                print(f"    Mean tendency: {prop_stats['mean_tendency']:+.3f}")
                print(f"    Mean certainty: {prop_stats['mean_abs_tendency']:.3f}")
                print(f"    Variance: {prop_stats['variance']:.3f}")

            if prop_stats.get('agreement_with_ensemble_rate') is not None:
                print(f"    Agreement with ensemble: {prop_stats['agreement_with_ensemble_rate']:.1%}")

    print("\n" + "="*80)


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(
        description='Analyze inter-model disagreement patterns in ensemble predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        'results_file',
        type=str,
        help='Path to comparison results JSON file (must include individual model labels)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output JSON file path for full analysis results'
    )

    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Print detailed model-specific patterns'
    )

    args = parser.parse_args()

    # Load results
    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"Error: File not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading comparison results from {results_path}...")
    data = load_results(results_path)

    # Check if individual model labels are available
    sample_claim = next((c for c in data['per_claim_comparisons'] if c['verdict_available']), None)
    if sample_claim:
        agg_comp = sample_claim.get('aggregated_comparison', {})
        sample_prop = next(iter(agg_comp.values()), {})
        if 'automated_individual_models' not in sample_prop or sample_prop['automated_individual_models'] is None:
            print("\nError: This comparison results file does not include individual model labels.", file=sys.stderr)
            print("Please regenerate the comparison results using the updated compare_to_annotations.py script.", file=sys.stderr)
            sys.exit(1)

    # Run analyses
    print("Analyzing overall inter-model agreement...")
    overall_stats = analyze_inter_model_agreement_overall(data)

    print("Analyzing inter-model agreement in disagreement cases...")
    disagreement_stats = analyze_model_agreement_in_disagreement_cases(data)

    print("Analyzing model-specific patterns...")
    model_patterns = analyze_model_specific_patterns(data)

    # Combine results
    full_analysis = {
        'metadata': data.get('metadata', {}),
        'overall_inter_model_agreement': overall_stats,
        'inter_model_agreement_in_disagreements': disagreement_stats,
        'model_specific_patterns': model_patterns,
    }

    # Print summaries
    print_overall_summary(overall_stats)
    print_disagreement_summary(disagreement_stats)

    if args.detailed:
        print_model_patterns(model_patterns)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(full_analysis, f, indent=2, default=str)
        print(f"\nFull analysis saved to: {output_path}")

    print("\n" + "="*80)
    print("INTERPRETATION GUIDE")
    print("="*80)
    print("""
Key Questions Answered:

1. Do ensemble models agree with each other overall?
   → See "INTER-MODEL AGREEMENT (ALL CLAIMS)"
   → High agreement (>80%) suggests models are consistent/redundant
   → Low agreement suggests models capture different aspects

2. When ensemble disagrees with humans, do models disagree internally?
   → See "INTER-MODEL AGREEMENT (WHEN ENSEMBLE DISAGREES WITH HUMANS)"
   → "All models agree (wrong together)" = systematic ensemble error
   → "Models split" = some models may be correct, averaging pulls ensemble wrong

3. Are there dominant vs dissenting models?
   → See "MODEL-SPECIFIC PATTERNS" (use --detailed flag)
   → Check agreement_with_ensemble_rate for each model
   → Lower rates suggest a model provides diverse/contrarian views

4. Which models are most certain/uncertain?
   → See "Mean certainty" (absolute value of tendency)
   → Higher values = more confident predictions
   → Compare with accuracy to identify overconfident models

Actionable Insights:

- High inter-model agreement + high human disagreement
  → Systematic bias; all models making same mistake

- Low inter-model agreement + high human disagreement
  → Models uncertain; ensemble averaging may not help

- Some models agree with humans when ensemble doesn't
  → Consider re-weighting ensemble or investigating why those models differ
    """)


if __name__ == '__main__':
    main()
