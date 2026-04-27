#!/usr/bin/env python3
"""
Analyze disagreement patterns between automated and human annotations.

This script takes the output from compare_to_annotations.py and performs
a deep dive into cases where automated and human annotations disagree.

Key analyses:
1. Inter-annotator agreement in disagreement cases
2. Whether disagreements involve model uncertainty or human uncertainty
3. Patterns in disagreement types

Usage:
    python scripts/analyze_disagreements.py comparison_results.json
    python scripts/analyze_disagreements.py comparison_results.json -o disagreement_analysis.json
    python scripts/analyze_disagreements.py comparison_results.json --detailed
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


def calculate_inter_annotator_agreement(annotations: list[dict], property_name: str) -> dict:
    """
    Calculate inter-annotator agreement for a specific property.

    Returns:
        - agreement_rate: proportion of annotator pairs that agree
        - annotations_values: list of (annotator_id, value, confidence) tuples
        - all_agree: whether all annotators agree
        - majority_value: the most common value (if exists)
    """
    # Extract values from annotations
    values = []
    for ann in annotations:
        if ann.get('dismissed'):
            continue

        comparisons = ann.get('comparisons', {})
        if property_name in comparisons:
            comp = comparisons[property_name]
            human_value = comp.get('human_value')
            human_confidence = comp.get('human_confidence')
            annotator_id = ann['annotator'].get('user_id', ann['annotation_id'])

            if human_value is not None:
                values.append({
                    'annotator_id': annotator_id,
                    'annotator_email': ann['annotator'].get('email', 'unknown'),
                    'value': human_value,
                    'confidence': human_confidence,
                })

    if len(values) < 2:
        return {
            'num_annotators': len(values),
            'agreement_rate': None,
            'all_agree': None,
            'majority_value': values[0]['value'] if values else None,
            'annotations': values,
        }

    # Calculate pairwise agreement
    agreements = 0
    total_pairs = 0

    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            total_pairs += 1
            # Normalize values for comparison
            val_i = str(values[i]['value']).strip().lower()
            val_j = str(values[j]['value']).strip().lower()

            # Map to positive/negative/unknown
            def categorize(val: str) -> str:
                if 'unknown' in val or val == '' or val is None:
                    return 'unknown'
                elif val in ['clear', 'pristine', 'correct', 'true', 'sufficient', 'legitimate', 'intact']:
                    return 'positive'
                else:
                    return 'negative'

            if categorize(val_i) == categorize(val_j):
                agreements += 1

    agreement_rate = agreements / total_pairs if total_pairs > 0 else None

    # Check if all agree
    first_category = str(values[0]['value']).strip().lower()
    all_agree = all(str(v['value']).strip().lower() == first_category for v in values)

    # Find majority value
    value_counts = defaultdict(int)
    for v in values:
        value_counts[v['value']] += 1
    majority_value = max(value_counts.items(), key=lambda x: x[1])[0] if value_counts else None

    return {
        'num_annotators': len(values),
        'agreement_rate': agreement_rate,
        'all_agree': all_agree,
        'majority_value': majority_value,
        'annotations': values,
        'value_counts': dict(value_counts),
    }


def analyze_single_disagreement(claim_comparison: dict, property_name: str) -> dict | None:
    """
    Analyze a single disagreement case for a specific property.

    Returns detailed analysis including:
    - Inter-annotator agreement
    - Which annotators agree with automated vs disagree
    - Confidence/certainty levels
    """
    agg_comp = claim_comparison.get('aggregated_comparison', {})
    if not agg_comp or property_name not in agg_comp:
        return None

    prop_comp = agg_comp[property_name]

    # Only analyze disagreements
    if prop_comp.get('agreement') is not False:
        return None

    # Get automated values
    automated_tendency = prop_comp.get('automated_tendency')
    human_tendency = prop_comp.get('human_tendency')

    # Calculate inter-annotator agreement
    inter_annotator = calculate_inter_annotator_agreement(
        claim_comparison['individual_comparisons'],
        property_name
    )

    # Analyze which annotators agree with automated
    annotators_agreeing_with_auto = []
    annotators_disagreeing_with_auto = []

    for ann in claim_comparison['individual_comparisons']:
        if ann.get('dismissed'):
            continue

        comparisons = ann.get('comparisons', {})
        if property_name in comparisons:
            comp = comparisons[property_name]
            if comp.get('agreement') is True:
                annotators_agreeing_with_auto.append({
                    'annotator_id': ann['annotator'].get('user_id'),
                    'email': ann['annotator'].get('email'),
                    'value': comp.get('human_value'),
                    'confidence': comp.get('human_confidence'),
                })
            elif comp.get('agreement') is False:
                annotators_disagreeing_with_auto.append({
                    'annotator_id': ann['annotator'].get('user_id'),
                    'email': ann['annotator'].get('email'),
                    'value': comp.get('human_value'),
                    'confidence': comp.get('human_confidence'),
                })

    return {
        'claim_id': claim_comparison['claim_id'],
        'property': property_name,
        'automated_tendency': automated_tendency,
        'human_aggregated_tendency': human_tendency,
        'difference': prop_comp.get('difference'),
        'inter_annotator_agreement': inter_annotator,
        'annotators_agreeing_with_auto': annotators_agreeing_with_auto,
        'annotators_disagreeing_with_auto': annotators_disagreeing_with_auto,
        'split_decision': len(annotators_agreeing_with_auto) > 0 and len(annotators_disagreeing_with_auto) > 0,
    }


def analyze_all_disagreements(data: dict) -> dict:
    """
    Analyze all disagreements in the comparison data.

    Returns:
        Comprehensive analysis of disagreement patterns
    """
    disagreements_by_property = defaultdict(list)

    # Collect all disagreements
    for claim_comp in data['per_claim_comparisons']:
        if not claim_comp['verdict_available']:
            continue

        properties = ['clarity', 'veracity', 'context_coverage', 'intent']

        for prop in properties:
            disagreement = analyze_single_disagreement(claim_comp, prop)
            if disagreement:
                disagreements_by_property[prop].append(disagreement)

    # Calculate statistics
    stats = {}

    for prop, disagreements in disagreements_by_property.items():
        if not disagreements:
            stats[prop] = {
                'total_disagreements': 0,
            }
            continue

        # Inter-annotator agreement in disagreement cases
        # Weight by number of annotator pairs for more accurate averaging
        inter_annotator_data = [
            (d['inter_annotator_agreement']['agreement_rate'],
             d['inter_annotator_agreement']['num_annotators'])
            for d in disagreements
            if d['inter_annotator_agreement']['agreement_rate'] is not None
        ]

        # Calculate weighted average (weighted by number of pairs)
        if inter_annotator_data:
            # Number of pairs = n*(n-1)/2
            weighted_sum = 0
            total_pairs = 0
            for agreement_rate, n_annotators in inter_annotator_data:
                n_pairs = n_annotators * (n_annotators - 1) // 2
                weighted_sum += agreement_rate * n_pairs
                total_pairs += n_pairs

            avg_inter_annotator_agreement = weighted_sum / total_pairs if total_pairs > 0 else None
            # Also keep simple average for comparison
            simple_avg = sum(rate for rate, _ in inter_annotator_data) / len(inter_annotator_data)
        else:
            avg_inter_annotator_agreement = None
            simple_avg = None

        # Annotator count distribution in disagreement cases
        annotator_counts = defaultdict(int)
        for d in disagreements:
            n = d['inter_annotator_agreement']['num_annotators']
            annotator_counts[n] += 1

        # Cases where humans all agree but disagree with automated
        humans_agree_count = sum(
            1 for d in disagreements
            if d['inter_annotator_agreement']['all_agree']
        )

        # Cases where some humans agree with automated, others don't
        split_decisions = sum(1 for d in disagreements if d['split_decision'])

        # Cases where no humans agree with automated
        no_humans_agree = sum(
            1 for d in disagreements
            if len(d['annotators_agreeing_with_auto']) == 0
        )

        # Average difference magnitude in disagreement cases
        avg_difference = sum(d['difference'] for d in disagreements if d['difference'] is not None) / len(disagreements)

        stats[prop] = {
            'total_disagreements': len(disagreements),
            'annotator_count_distribution': dict(annotator_counts),
            'avg_inter_annotator_agreement_weighted': avg_inter_annotator_agreement,
            'avg_inter_annotator_agreement_simple': simple_avg,
            'cases_with_multiple_annotators': sum(1 for d in disagreements if d['inter_annotator_agreement']['num_annotators'] > 1),
            'cases_humans_all_agree': humans_agree_count,
            'cases_humans_all_agree_pct': humans_agree_count / len(disagreements) if disagreements else 0,
            'cases_split_decision': split_decisions,
            'cases_split_decision_pct': split_decisions / len(disagreements) if disagreements else 0,
            'cases_no_humans_agree_with_auto': no_humans_agree,
            'cases_no_humans_agree_with_auto_pct': no_humans_agree / len(disagreements) if disagreements else 0,
            'avg_difference_magnitude': avg_difference,
        }

    # Also stratify by annotator count
    stratified_stats = {}
    for prop, disagreements in disagreements_by_property.items():
        stratified_stats[prop] = defaultdict(list)
        for d in disagreements:
            n_annotators = d['inter_annotator_agreement']['num_annotators']
            stratified_stats[prop][n_annotators].append(d)

    return {
        'disagreements_by_property': {k: list(v) for k, v in disagreements_by_property.items()},
        'summary_statistics': stats,
        'stratified_by_annotator_count': {k: dict(v) for k, v in stratified_stats.items()},
    }


def analyze_annotator_patterns(data: dict) -> dict:
    """
    Analyze patterns specific to individual annotators.

    Returns:
        Per-annotator statistics on agreement/disagreement patterns
    """
    annotator_stats = defaultdict(lambda: {
        'total_annotations': 0,
        'agreements_with_auto': 0,
        'disagreements_with_auto': 0,
        'properties': defaultdict(lambda: {'agree': 0, 'disagree': 0}),
    })

    for claim_comp in data['per_claim_comparisons']:
        if not claim_comp['verdict_available']:
            continue

        for ann in claim_comp['individual_comparisons']:
            if ann.get('dismissed'):
                continue

            annotator_id = ann['annotator'].get('email', ann['annotator'].get('user_id'))
            annotator_stats[annotator_id]['total_annotations'] += 1
            annotator_stats[annotator_id]['annotator_info'] = ann['annotator']

            comparisons = ann.get('comparisons', {})
            properties = ['clarity', 'veracity', 'context_coverage', 'intent']

            for prop in properties:
                if prop in comparisons:
                    comp = comparisons[prop]
                    agreement = comp.get('agreement')

                    if agreement is True:
                        annotator_stats[annotator_id]['agreements_with_auto'] += 1
                        annotator_stats[annotator_id]['properties'][prop]['agree'] += 1
                    elif agreement is False:
                        annotator_stats[annotator_id]['disagreements_with_auto'] += 1
                        annotator_stats[annotator_id]['properties'][prop]['disagree'] += 1

    # Calculate rates
    for annotator_id, stats in annotator_stats.items():
        total = stats['agreements_with_auto'] + stats['disagreements_with_auto']
        if total > 0:
            stats['agreement_rate'] = stats['agreements_with_auto'] / total
            stats['disagreement_rate'] = stats['disagreements_with_auto'] / total
        else:
            stats['agreement_rate'] = None
            stats['disagreement_rate'] = None

        # Convert properties to dict
        stats['properties'] = dict(stats['properties'])

    return dict(annotator_stats)


def print_summary(analysis: dict):
    """Print a summary of the disagreement analysis."""
    print("\n" + "="*80)
    print("DISAGREEMENT ANALYSIS SUMMARY")
    print("="*80)

    stats = analysis['summary_statistics']

    for prop, prop_stats in stats.items():
        if prop_stats['total_disagreements'] == 0:
            continue

        print(f"\n{prop.replace('_', ' ').title()}:")
        print(f"  Total disagreements: {prop_stats['total_disagreements']}")

        # Show annotator count distribution
        annotator_dist = prop_stats.get('annotator_count_distribution', {})
        if annotator_dist:
            print(f"  Annotator count distribution:")
            for count in sorted(annotator_dist.keys()):
                print(f"    {count} annotator{'s' if count > 1 else ''}: {annotator_dist[count]} case{'s' if annotator_dist[count] > 1 else ''}")

        # Show inter-annotator agreement (only for cases with 2+ annotators)
        n_multi = prop_stats.get('cases_with_multiple_annotators', 0)
        if n_multi > 0:
            weighted_avg = prop_stats.get('avg_inter_annotator_agreement_weighted')
            if weighted_avg is not None:
                print(f"  Avg inter-annotator agreement (weighted by pairs, n={n_multi}): {weighted_avg:.2%}")
        else:
            print(f"  Inter-annotator agreement: N/A (all cases have single annotator)")

        print(f"  Cases where all humans agree (but disagree with auto): "
              f"{prop_stats['cases_humans_all_agree']} ({prop_stats['cases_humans_all_agree_pct']:.1%})")

        print(f"  Cases with split decisions (some humans agree with auto): "
              f"{prop_stats['cases_split_decision']} ({prop_stats['cases_split_decision_pct']:.1%})")

        print(f"  Cases where no humans agree with auto: "
              f"{prop_stats['cases_no_humans_agree_with_auto']} ({prop_stats['cases_no_humans_agree_with_auto_pct']:.1%})")

        print(f"  Avg difference magnitude: {prop_stats['avg_difference_magnitude']:.3f}")

    print("\n" + "="*80)


def print_detailed_disagreements(analysis: dict, limit: int | None = None):
    """Print detailed information about each disagreement."""
    print("\n" + "="*80)
    print("DETAILED DISAGREEMENT ANALYSIS")
    print("="*80)

    for prop, disagreements in analysis['disagreements_by_property'].items():
        if not disagreements:
            continue

        print(f"\n{'─'*80}")
        print(f"{prop.replace('_', ' ').upper()}")
        print(f"{'─'*80}")

        for i, d in enumerate(disagreements):
            if limit and i >= limit:
                print(f"\n... and {len(disagreements) - limit} more disagreements")
                break

            print(f"\nClaim {d['claim_id']}:")
            print(f"  Automated tendency: {d['automated_tendency']:+.3f}")
            print(f"  Human aggregated tendency: {d['human_aggregated_tendency']:+.3f}")
            print(f"  Difference: {d['difference']:.3f}")

            inter_ann = d['inter_annotator_agreement']
            print(f"\n  Inter-annotator agreement:")
            print(f"    Number of annotators: {inter_ann['num_annotators']}")
            if inter_ann['agreement_rate'] is not None:
                print(f"    Agreement rate: {inter_ann['agreement_rate']:.2%}")
            print(f"    All annotators agree: {inter_ann['all_agree']}")

            if inter_ann['annotations']:
                print(f"    Human annotations:")
                for ann in inter_ann['annotations']:
                    print(f"      - {ann['annotator_email']}: {ann['value']} (confidence: {ann['confidence']})")

            if d['split_decision']:
                print(f"\n  SPLIT DECISION:")
                print(f"    {len(d['annotators_agreeing_with_auto'])} annotator(s) agree with automated")
                print(f"    {len(d['annotators_disagreeing_with_auto'])} annotator(s) disagree with automated")

            print()


def print_annotator_patterns(annotator_stats: dict):
    """Print per-annotator pattern analysis."""
    print("\n" + "="*80)
    print("ANNOTATOR-SPECIFIC PATTERNS")
    print("="*80)

    print(f"\n{'Annotator':<40} {'Total':<10} {'Agree':<10} {'Disagree':<10} {'Agree %':<10}")
    print("─"*80)

    for annotator_id, stats in sorted(annotator_stats.items()):
        email = stats['annotator_info'].get('email', annotator_id)
        if len(email) > 38:
            email = email[:35] + "..."

        total = stats['agreements_with_auto'] + stats['disagreements_with_auto']
        agree_rate = stats['agreement_rate']

        agree_pct = f"{agree_rate:.1%}" if agree_rate is not None else "N/A"

        print(f"{email:<40} {total:<10} {stats['agreements_with_auto']:<10} "
              f"{stats['disagreements_with_auto']:<10} {agree_pct:<10}")

    print("\n" + "="*80)


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(
        description='Analyze disagreement patterns between automated and human annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        'results_file',
        type=str,
        help='Path to comparison results JSON file'
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
        help='Print detailed disagreement information'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='Limit number of detailed disagreements to print per property (default: 10)'
    )

    args = parser.parse_args()

    # Load results
    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"Error: File not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading comparison results from {results_path}...")
    data = load_results(results_path)

    # Run analyses
    print("Analyzing disagreement patterns...")
    disagreement_analysis = analyze_all_disagreements(data)

    print("Analyzing annotator-specific patterns...")
    annotator_patterns = analyze_annotator_patterns(data)

    # Combine results
    full_analysis = {
        'metadata': data.get('metadata', {}),
        'disagreement_analysis': disagreement_analysis,
        'annotator_patterns': annotator_patterns,
    }

    # Print summary
    print_summary(disagreement_analysis)
    print_annotator_patterns(annotator_patterns)

    # Print detailed if requested
    if args.detailed:
        print_detailed_disagreements(disagreement_analysis, limit=args.limit)

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

1. When automatic and human disagree, do humans agree with each other?
   → See "Avg inter-annotator agreement (weighted by pairs, n=X)"
   → This is only calculated for cases with 2+ annotators (shown as n=X)
   → Weighted by number of annotator pairs (2 annotators = 1 pair, 3 = 3 pairs)
   → High values (>70%) suggest humans are consistent even when disagreeing with automation
   → Low values suggest the case is genuinely ambiguous

2. Are disagreements due to split opinions?
   → See "Cases with split decisions"
   → High percentages suggest some humans agree with automation, others don't
   → This indicates the automated prediction may be valid but controversial

3. Are there systematic automated errors?
   → See "Cases where all humans agree (but disagree with auto)"
   → High values suggest automation may have systematic biases
   → "Cases where no humans agree with auto" shows strongest errors

4. Do different annotators have different agreement patterns?
   → See "ANNOTATOR-SPECIFIC PATTERNS" table
   → Varying agreement rates may indicate different annotation strategies

5. How does annotator count affect the analysis?
   → See "Annotator count distribution" for each property
   → Single-annotator cases cannot contribute to inter-annotator agreement
   → Claims with more annotators provide stronger evidence for/against patterns
   → The weighted average gives more weight to claims with more annotator pairs
    """)


if __name__ == '__main__':
    main()
