#!/usr/bin/env python3
"""
Present comparison results in a readable format.

Focuses on aggregated human vs automatic comparisons, ignoring individual annotations.
Now includes inter-annotator and inter-model agreement analyses in PDF reports.

Usage:
    python scripts/present_comparison_results.py comparison_results.json
    python scripts/present_comparison_results.py comparison_results.json --format summary
    python scripts/present_comparison_results.py comparison_results.json --format detailed
    python scripts/present_comparison_results.py comparison_results.json --pdf report.pdf
    python scripts/present_comparison_results.py comparison_results.json --csv results.csv --pdf report.pdf
"""

import argparse
import json
import sys
from pathlib import Path

# Import analysis functions
try:
    # Try importing from same directory
    from analyze_disagreements import analyze_all_disagreements, analyze_annotator_patterns
    from analyze_model_disagreements import (
        analyze_inter_model_agreement_overall,
        analyze_model_agreement_in_disagreement_cases,
        analyze_model_agreement_by_human_agreement,
        analyze_model_specific_patterns
    )
    ANALYSIS_AVAILABLE = True
except ImportError:
    # If running as script, try importing from module path
    try:
        import sys
        from pathlib import Path
        script_dir = Path(__file__).parent
        sys.path.insert(0, str(script_dir))
        from analyze_disagreements import analyze_all_disagreements, analyze_annotator_patterns
        from analyze_model_disagreements import (
            analyze_inter_model_agreement_overall,
            analyze_model_agreement_in_disagreement_cases,
            analyze_model_agreement_by_human_agreement,
            analyze_model_specific_patterns
        )
        ANALYSIS_AVAILABLE = True
    except ImportError:
        ANALYSIS_AVAILABLE = False
        print("Warning: Advanced analysis modules not available. PDF will contain basic comparison only.", file=sys.stderr)


def load_results(results_path: Path) -> dict:
    """Load comparison results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def format_tendency(tendency: float | None) -> str:
    """Format tendency value with color coding."""
    if tendency is None:
        return "     N/A"

    # Format with sign and 2 decimal places
    sign = "+" if tendency >= 0 else ""
    return f"{sign}{tendency:>6.3f}"


def format_percentage(value: float | None) -> str:
    """Format percentage value."""
    if value is None:
        return "  N/A"
    return f"{value:>5.1%}"


def format_difference(diff: float | None) -> str:
    """Format absolute difference."""
    if diff is None:
        return "   N/A"
    return f"{diff:>6.3f}"


def print_summary_table(data: dict):
    """Print summary statistics in table format."""
    summary = data['summary']

    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"Total claims processed:        {summary['total_claims']}")
    print(f"Claims with verdicts:          {summary['claims_with_verdicts']}")
    print(f"Failed predictions:            {summary['claims_with_failed_verdicts']}")
    print(f"Total human annotations:       {summary['total_annotations']}")
    print()

    # Property-level statistics table
    print("PROPERTY-LEVEL AGREEMENT")
    print("-"*100)
    print(f"{'Property':<20} {'Sample Size':<20} {'Agreement Rate':<20} {'Mean Abs Diff':<20}")
    print(f"{'':20} {'(both/total)':<20} {'':20} {'':20}")
    print("-"*100)

    properties = ['clarity', 'veracity', 'context_coverage', 'intent']
    agreement_rates = summary['agreement_rates']
    mean_diffs = summary['mean_absolute_differences']
    prop_stats = summary['property_statistics']

    for prop in properties:
        prop_display = prop.replace('_', ' ').title()
        agreement = format_percentage(agreement_rates.get(prop))
        diff = format_difference(mean_diffs.get(prop))

        # Get sample size
        sample_size = prop_stats.get(prop, {}).get('total', 0)
        no_data = prop_stats.get(prop, {}).get('no_data', 0)
        sample_str = f"{sample_size} / {sample_size + no_data}"

        print(f"{prop_display:<20} {sample_str:<20} {agreement:<20} {diff:<20}")

    print("="*100)

    # Certainty comparison
    automated_certs = summary.get('automated_certainties', {})
    human_certs = summary.get('human_certainties', {})

    if automated_certs and human_certs:
        print()
        print("CERTAINTY COMPARISON")
        print("-"*100)
        print(f"{'Property':<20} {'Automated':<15} {'Human':<15} {'Difference':<15} {'More Certain':<20}")
        print(f"{'':20} {'Certainty':<15} {'Certainty':<15} {'(H - A)':<15} {'':20}")
        print("-"*100)

        for prop in properties:
            prop_display = prop.replace('_', ' ').title()
            auto_cert = automated_certs.get(prop)
            human_cert = human_certs.get(prop)

            if auto_cert is not None and human_cert is not None:
                diff = human_cert - auto_cert
                more_certain = "Human" if diff > 0.05 else ("Automated" if diff < -0.05 else "Similar")

                auto_str = f"{auto_cert:.3f}"
                human_str = f"{human_cert:.3f}"
                diff_str = f"{diff:+.3f}"

                print(f"{prop_display:<20} {auto_str:<15} {human_str:<15} {diff_str:<15} {more_certain:<20}")
            else:
                print(f"{prop_display:<20} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<20}")

        print("="*100)
        print()
        print("Note: Certainty = |tendency|, where 0 = very uncertain, 1 = very certain")


def print_detailed_comparisons(data: dict):
    """Print detailed per-claim comparisons."""
    comparisons = data['per_claim_comparisons']

    print("\n" + "="*80)
    print("DETAILED PER-CLAIM COMPARISONS")
    print("="*80)

    for comparison in comparisons:
        claim_id = comparison['claim_id']

        if not comparison['verdict_available']:
            print(f"\nClaim {claim_id}: VERDICT PREDICTION FAILED")
            continue

        agg_comp = comparison.get('aggregated_comparison')
        if not agg_comp:
            print(f"\nClaim {claim_id}: NO AGGREGATED ANNOTATIONS")
            continue

        print(f"\n{'─'*80}")
        print(f"CLAIM {claim_id}")
        print(f"{'─'*80}")
        print(f"Number of human annotations: {comparison['num_annotations']}")
        print()

        # Table header
        print(f"{'Property':<20} {'Automated':<12} {'Human Avg':<12} {'Difference':<12} {'Agreement':<12}")
        print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

        properties = ['clarity', 'veracity', 'context_coverage', 'intent']

        for prop in properties:
            if prop not in agg_comp:
                continue

            comp = agg_comp[prop]
            prop_display = prop.replace('_', ' ').title()

            auto = format_tendency(comp.get('automated_tendency'))
            human = format_tendency(comp.get('human_tendency'))
            diff = format_difference(comp.get('difference'))

            agreement = comp.get('agreement')
            if agreement is None:
                agree_str = "N/A"
            elif agreement:
                agree_str = "✓ AGREE"
            else:
                agree_str = "✗ DISAGREE"

            print(f"{prop_display:<20} {auto:<12} {human:<12} {diff:<12} {agree_str:<12}")

    print("\n" + "="*80)


def print_compact_table(data: dict):
    """Print all claims in a compact table format."""
    comparisons = data['per_claim_comparisons']

    print("\n" + "="*100)
    print("ALL CLAIMS COMPARISON (AGGREGATED)")
    print("="*100)
    print(f"{'Claim':<8} {'Clarity':<20} {'Veracity':<20} {'Context':<20} {'Intent':<20}")
    print(f"{'ID':<8} {'Auto/Human/Agr':<20} {'Auto/Human/Agr':<20} {'Auto/Human/Agr':<20} {'Auto/Human/Agr':<20}")
    print("-"*100)

    for comparison in comparisons:
        claim_id = comparison['claim_id']

        if not comparison['verdict_available']:
            print(f"{claim_id:<8} {'VERDICT FAILED':<80}")
            continue

        agg_comp = comparison.get('aggregated_comparison')
        if not agg_comp:
            print(f"{claim_id:<8} {'NO ANNOTATIONS':<80}")
            continue

        line = f"{claim_id:<8} "

        properties = ['clarity', 'veracity', 'context_coverage', 'intent']

        for prop in properties:
            if prop not in agg_comp:
                line += f"{'N/A':<20} "
                continue

            comp = agg_comp[prop]
            auto = comp.get('automated_tendency')
            human = comp.get('human_tendency')
            agreement = comp.get('agreement')

            if auto is None or human is None:
                cell = "N/A"
            else:
                agree_symbol = "✓" if agreement else "✗"
                cell = f"{auto:+.2f}/{human:+.2f} {agree_symbol}"

            line += f"{cell:<20} "

        print(line)

    print("="*100)


def print_disagreements_only(data: dict):
    """Print only claims with disagreements."""
    comparisons = data['per_claim_comparisons']

    print("\n" + "="*80)
    print("CLAIMS WITH DISAGREEMENTS")
    print("="*80)

    has_disagreements = False

    for comparison in comparisons:
        claim_id = comparison['claim_id']

        if not comparison['verdict_available']:
            continue

        agg_comp = comparison.get('aggregated_comparison')
        if not agg_comp:
            continue

        # Check for any disagreements
        disagreements = []
        properties = ['clarity', 'veracity', 'context_coverage', 'intent']

        for prop in properties:
            if prop not in agg_comp:
                continue

            comp = agg_comp[prop]
            if comp.get('agreement') is False:
                disagreements.append({
                    'property': prop,
                    'automated': comp.get('automated_tendency'),
                    'human': comp.get('human_tendency'),
                    'difference': comp.get('difference'),
                })

        if disagreements:
            has_disagreements = True
            print(f"\nClaim {claim_id}:")
            for d in disagreements:
                prop_display = d['property'].replace('_', ' ').title()
                print(f"  {prop_display:<20}: Auto={format_tendency(d['automated'])}, "
                      f"Human={format_tendency(d['human'])}, "
                      f"Diff={format_difference(d['difference'])}")

    if not has_disagreements:
        print("\nNo disagreements found!")

    print("\n" + "="*80)


def add_inter_annotator_section(story, styles, heading_style, disagreement_analysis, annotator_patterns, properties):
    """Add inter-annotator agreement section to PDF."""
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import Table, TableStyle, Paragraph, Spacer

    story.append(Paragraph("Inter-Annotator Agreement Analysis", heading_style))
    story.append(Paragraph(
        "This section analyzes disagreement patterns between automated and human annotations, "
        "focusing on inter-annotator consistency and disagreement types.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.2*inch))

    stats = disagreement_analysis['summary_statistics']

    # Caption for disagreement patterns table
    story.append(Paragraph("<b>Table 1: Disagreement Patterns by Property</b>", styles['Normal']))
    story.append(Paragraph(
        "For each property, this shows: total disagreements, how many annotators per case (e.g., '1:5' means 5 cases with 1 annotator), "
        "weighted inter-annotator agreement (only for cases with 2+ annotators), how many cases where all humans agree but disagree with automation, "
        "split decisions (some humans agree with auto, others don't), and cases where no human agrees with automation.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.1*inch))

    # Create table data
    table_data = [
        ['Property', 'Total\nDisagree', 'Annotator\nCount Dist', 'Inter-Ann\nAgreement', 'All Humans\nAgree', 'Split\nDecision', 'None Agree\nw/ Auto']
    ]

    for prop in properties:
        if prop not in stats or stats[prop].get('total_disagreements', 0) == 0:
            continue

        prop_stats = stats[prop]
        prop_display = prop.replace('_', ' ').title()

        # Format annotator distribution
        dist = prop_stats.get('annotator_count_distribution', {})
        dist_str = ', '.join(f"{k}:{v}" for k, v in sorted(dist.items()))

        # Format inter-annotator agreement
        weighted_avg = prop_stats.get('avg_inter_annotator_agreement_weighted')
        n_multi = prop_stats.get('cases_with_multiple_annotators', 0)
        if weighted_avg is not None and n_multi > 0:
            inter_ann_str = f"{weighted_avg:.0%}\n(n={n_multi})"
        else:
            inter_ann_str = "N/A"

        table_data.append([
            prop_display,
            str(prop_stats['total_disagreements']),
            dist_str,
            inter_ann_str,
            f"{prop_stats['cases_humans_all_agree']}\n({prop_stats['cases_humans_all_agree_pct']:.0%})",
            f"{prop_stats['cases_split_decision']}\n({prop_stats['cases_split_decision_pct']:.0%})",
            f"{prop_stats['cases_no_humans_agree_with_auto']}\n({prop_stats['cases_no_humans_agree_with_auto_pct']:.0%})",
        ])

    if len(table_data) > 1:
        table = Table(table_data, colWidths=[1*inch, 0.6*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.7*inch, 0.8*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a4a4a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 1), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.white]),
        ]))
        story.append(table)

    story.append(Spacer(1, 0.2*inch))

    # Annotator patterns
    story.append(Paragraph("Annotator-Specific Patterns", styles['Heading3']))
    story.append(Spacer(1, 0.1*inch))

    # Caption for annotator patterns table
    story.append(Paragraph("<b>Table 2: Per-Annotator Agreement with Automation</b>", styles['Normal']))
    story.append(Paragraph(
        "Shows how often each human annotator agrees with the automated system across all properties they annotated. "
        "'Total' is the number of property-level annotations, 'Agree/Disagree' count polarity matches/mismatches, "
        "and 'Rate' is the percentage agreement with automation.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.1*inch))

    ann_table_data = [['Annotator', 'Total', 'Agree', 'Disagree', 'Rate']]
    for annotator_id, stats in sorted(annotator_patterns.items()):
        email = stats['annotator_info'].get('email', annotator_id)
        if len(email) > 35:
            email = email[:32] + "..."

        total = stats['agreements_with_auto'] + stats['disagreements_with_auto']
        agree_rate = stats['agreement_rate']

        ann_table_data.append([
            email,
            str(total),
            str(stats['agreements_with_auto']),
            str(stats['disagreements_with_auto']),
            f"{agree_rate:.0%}" if agree_rate is not None else "N/A"
        ])

    if len(ann_table_data) > 1:
        ann_table = Table(ann_table_data, colWidths=[3*inch, 0.7*inch, 0.7*inch, 0.8*inch, 0.7*inch])
        ann_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a4a4a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.white]),
        ]))
        story.append(ann_table)


def add_inter_model_section(story, styles, heading_style, model_overall_stats, model_disagreement_stats, model_comparison_stats, model_patterns, properties):
    """Add inter-model agreement section to PDF."""
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import Table, TableStyle, Paragraph, Spacer

    story.append(Paragraph("Inter-Model Agreement Analysis", heading_style))
    story.append(Paragraph(
        "This section analyzes how individual ensemble members agree with each other, "
        "both overall and specifically in cases where the ensemble disagrees with humans.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.2*inch))

    # Overall inter-model agreement
    story.append(Paragraph("Overall Inter-Model Agreement", styles['Heading3']))
    story.append(Spacer(1, 0.1*inch))

    # Caption for overall inter-model table
    story.append(Paragraph("<b>Table 3: Overall Inter-Model Agreement Across All Claims</b>", styles['Normal']))
    story.append(Paragraph(
        "For each property, shows how often individual ensemble models agree with each other across all claims. "
        "'Avg Agreement Rate' is the percentage of model pairs that agree on polarity. "
        "'All Models Agree' shows cases where every model predicts the same category. "
        "'Tendency Variance' measures spread of model predictions (higher = more disagreement between models).",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.1*inch))

    overall_table_data = [
        ['Property', 'Total\nClaims', 'Avg Agreement\nRate', 'All Models\nAgree', 'Tendency\nVariance']
    ]

    for prop in properties:
        if prop not in model_overall_stats or model_overall_stats[prop].get('total_claims', 0) == 0:
            continue

        prop_stats = model_overall_stats[prop]
        prop_display = prop.replace('_', ' ').title()

        overall_table_data.append([
            prop_display,
            str(prop_stats['total_claims']),
            f"{prop_stats['avg_agreement_rate']:.0%}" if prop_stats['avg_agreement_rate'] is not None else "N/A",
            f"{prop_stats['all_agree_count']}\n({prop_stats['all_agree_pct']:.0%})",
            f"{prop_stats['avg_tendency_variance']:.3f}" if prop_stats['avg_tendency_variance'] is not None else "N/A",
        ])

    if len(overall_table_data) > 1:
        overall_table = Table(overall_table_data, colWidths=[1.5*inch, 0.8*inch, 1*inch, 1*inch, 1*inch])
        overall_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a4a4a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.white]),
        ]))
        story.append(overall_table)

    story.append(Spacer(1, 0.2*inch))

    # Inter-model agreement in disagreement cases
    story.append(Paragraph("Inter-Model Agreement (When Ensemble Disagrees with Humans)", styles['Heading3']))
    story.append(Spacer(1, 0.1*inch))

    # Caption for disagreement inter-model table
    story.append(Paragraph("<b>Table 4: Inter-Model Agreement in Human Disagreement Cases</b>", styles['Normal']))
    story.append(Paragraph(
        "Analyzes model agreement specifically for claims where the ensemble disagrees with humans. "
        "'Total Disagree' is the number of claims where ensemble and human verdicts differ. "
        "'All Models Agree (Wrong)' means all models made the same prediction but humans disagree - suggests systematic bias. "
        "'Models Split' means some models agreed with humans while others didn't - ensemble averaging may hide correct minority opinions.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.1*inch))

    disagree_table_data = [
        ['Property', 'Total\nDisagree', 'Avg Model\nAgreement', 'All Models\nAgree (Wrong)', 'Models\nSplit']
    ]

    for prop in properties:
        if prop not in model_disagreement_stats or model_disagreement_stats[prop].get('total_disagreements_with_humans', 0) == 0:
            continue

        prop_stats = model_disagreement_stats[prop]
        prop_display = prop.replace('_', ' ').title()

        disagree_table_data.append([
            prop_display,
            str(prop_stats['total_disagreements_with_humans']),
            f"{prop_stats['avg_model_agreement_rate']:.0%}" if prop_stats['avg_model_agreement_rate'] is not None else "N/A",
            f"{prop_stats['all_models_agree_count']}\n({prop_stats['all_models_agree_pct']:.0%})",
            f"{prop_stats['split_model_decisions']}\n({prop_stats['split_model_decisions_pct']:.0%})",
        ])

    if len(disagree_table_data) > 1:
        disagree_table = Table(disagree_table_data, colWidths=[1.5*inch, 0.9*inch, 1.1*inch, 1.2*inch, 0.9*inch])
        disagree_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a4a4a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.white]),
        ]))
        story.append(disagree_table)

    story.append(Spacer(1, 0.2*inch))

    # Comparison: agreement vs disagreement cases
    story.append(Paragraph("Model Uncertainty Calibration", styles['Heading3']))
    story.append(Spacer(1, 0.1*inch))

    # Caption for comparison table
    story.append(Paragraph("<b>Table 4.5: Inter-Model Agreement - When Ensemble is Right vs Wrong</b>", styles['Normal']))
    story.append(Paragraph(
        "Compares model agreement in cases where the ensemble matches human judgment vs cases where it doesn't. "
        "<b>Key insight:</b> If 'Variance' is higher when ensemble disagrees with humans, this indicates good calibration - "
        "models are more uncertain when the ensemble is wrong. If variance is similar or lower, models are overconfident in their mistakes.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.1*inch))

    comparison_table_data = [
        ['Property', 'Ensemble=Human\n(Correct)', '', 'Ensemble≠Human\n(Wrong)', '', 'Calibration'],
        ['', 'Count', 'Variance', 'Count', 'Variance', '']
    ]

    for prop in properties:
        if prop not in model_comparison_stats:
            continue

        prop_stats = model_comparison_stats[prop]
        agree_stats = prop_stats['agreement_cases']
        disagree_stats = prop_stats['disagreement_cases']

        prop_display = prop.replace('_', ' ').title()

        # Format values
        agree_count = agree_stats['count']
        disagree_count = disagree_stats['count']
        agree_var = agree_stats['avg_variance']
        disagree_var = disagree_stats['avg_variance']

        # Determine calibration
        if agree_var is not None and disagree_var is not None:
            if disagree_var > agree_var * 1.2:
                calibration = "✓ Good"
            elif disagree_var < agree_var * 0.8:
                calibration = "✗ Overconfident"
            else:
                calibration = "~ Similar"
        else:
            calibration = "N/A"

        comparison_table_data.append([
            prop_display,
            str(agree_count),
            f"{agree_var:.3f}" if agree_var is not None else "N/A",
            str(disagree_count),
            f"{disagree_var:.3f}" if disagree_var is not None else "N/A",
            calibration
        ])

    if len(comparison_table_data) > 2:
        comparison_table = Table(comparison_table_data, colWidths=[1.3*inch, 0.7*inch, 0.8*inch, 0.7*inch, 0.8*inch, 1*inch])
        comparison_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 1), colors.HexColor('#4a4a4a')),
            ('TEXTCOLOR', (0, 0), (-1, 1), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 1), 8),
            ('FONTSIZE', (0, 2), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 1), 6),
            ('SPAN', (1, 0), (2, 0)),  # Merge cells for "Ensemble=Human"
            ('SPAN', (3, 0), (4, 0)),  # Merge cells for "Ensemble≠Human"
            ('SPAN', (5, 0), (5, 1)),  # Merge cells for "Calibration"
            ('BACKGROUND', (0, 2), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 2), (-1, -1), [colors.beige, colors.white]),
        ]))
        story.append(comparison_table)

    story.append(Spacer(1, 0.2*inch))

    # Model-specific patterns
    story.append(Paragraph("Model-Specific Patterns", styles['Heading3']))
    story.append(Paragraph(
        "The following tables show detailed statistics for each individual model in the ensemble. "
        "'Mean Tendency' shows the average prediction for each property (positive/negative direction). "
        "'Mean Certainty' is the average absolute tendency (how confident the model is, regardless of direction). "
        "'Ensemble Agreement' shows how often the individual model agrees with the final ensemble prediction - "
        "lower rates indicate a model that provides diverse/contrarian views.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.15*inch))

    # Create a table for each model
    table_num = 5  # Start numbering after the previous tables
    for model_name, stats in sorted(model_patterns.items()):
        story.append(Paragraph(f"<b>Table {table_num}: {model_name}</b>", styles['Normal']))
        table_num += 1
        story.append(Spacer(1, 0.05*inch))

        model_table_data = [['Property', 'Mean Tendency', 'Mean Certainty', 'Ensemble Agreement']]

        for prop in properties:
            if prop not in stats['properties']:
                continue

            prop_stats = stats['properties'][prop]
            prop_display = prop.replace('_', ' ').title()

            model_table_data.append([
                prop_display,
                f"{prop_stats.get('mean_tendency', 0):+.3f}",
                f"{prop_stats.get('mean_abs_tendency', 0):.3f}",
                f"{prop_stats.get('agreement_with_ensemble_rate', 0):.0%}" if prop_stats.get('agreement_with_ensemble_rate') is not None else "N/A",
            ])

        if len(model_table_data) > 1:
            model_table = Table(model_table_data, colWidths=[2*inch, 1.2*inch, 1.2*inch, 1.5*inch])
            model_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#666666')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(model_table)
            story.append(Spacer(1, 0.15*inch))


def export_pdf(data: dict, output_path: Path):
    """Export aggregated comparisons to PDF."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        print("Error: reportlab not installed. Install with: pip install reportlab", file=sys.stderr)
        sys.exit(1)

    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=TA_CENTER
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#333333'),
        spaceAfter=12,
        spaceBefore=12
    )

    # Title
    story.append(Paragraph("Veritas Comparison Report", title_style))
    story.append(Paragraph("Aggregated Human vs Automated Verdicts", styles['Heading3']))
    story.append(Spacer(1, 0.3*inch))

    # Metadata
    metadata = data.get('metadata', {})
    story.append(Paragraph(f"<b>Generated:</b> {metadata.get('timestamp', 'N/A')}", styles['Normal']))
    story.append(Paragraph(f"<b>Input File:</b> {metadata.get('input_file', 'N/A')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))

    # Key Definitions
    story.append(Paragraph("Key Definitions", heading_style))
    story.append(Paragraph(
        "<b>Tendency Scale:</b> All predictions use a continuous scale from -1.0 to +1.0, where the sign indicates "
        "the verdict (positive/negative) and the magnitude indicates confidence (0 = very uncertain, 1 = very certain).",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph(
        "<b>Agreement Definition:</b> Agreement is measured by <i>polarity matching</i>, NOT exact value matching. "
        "Two predictions agree if they fall in the same category using a threshold of ±0.1:",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.05*inch))

    # Create definitions table
    definitions_data = [
        ['Property', 'Positive (>0.1)', 'Neutral (±0.1)', 'Negative (<-0.1)'],
        ['Clarity', 'Clear', 'Unknown', 'Ambiguous'],
        ['Veracity', 'True', 'Unknown', 'False'],
        ['Context Coverage', 'Sufficient', 'Unknown', 'Insufficient'],
        ['Intent', 'Legitimate', 'Unknown', 'Illegitimate'],
    ]

    from reportlab.lib import colors
    definitions_table = Table(definitions_data, colWidths=[1.5*inch, 1.5*inch, 1.3*inch, 1.5*inch])
    definitions_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a4a4a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(definitions_table)
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph(
        "<b>Example:</b> Automated tendency +0.89 and Human tendency +0.30 both count as 'Positive' → Agreement ✓ "
        "(even though confidence levels differ). But +0.67 vs -0.67 would be disagreement ✗ (different polarities).",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph(
        "<b>Mean Absolute Difference:</b> To measure <i>how far apart</i> predictions are (regardless of agreement), "
        "we also report the mean absolute difference in tendency values. Range: 0 (identical) to 2 (maximum difference, e.g., +1 vs -1).",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.3*inch))

    # Summary Statistics
    summary = data['summary']
    story.append(Paragraph("Summary Statistics", heading_style))
    story.append(Paragraph(
        "Overview of the comparison between automated pipeline verdicts and aggregated human annotations.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.1*inch))

    summary_data = [
        ['Metric', 'Value'],
        ['Total Claims Processed', str(summary['total_claims'])],
        ['Claims with Verdicts', str(summary['claims_with_verdicts'])],
        ['Failed Predictions', str(summary['claims_with_failed_verdicts'])],
        ['Total Human Annotations', str(summary['total_annotations'])],
    ]

    summary_table = Table(summary_data, colWidths=[3.5*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a4a4a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))

    # Agreement Rates
    story.append(Paragraph("Property-Level Agreement", heading_style))
    story.append(Paragraph(
        "<b>How to read:</b> Sample Size shows claims with both values / total claims. "
        "Agreement rate shows the percentage of claims where automated and human agree on <i>polarity</i> "
        "(see 'Key Definitions' above - agreement uses ±0.1 threshold, not exact matching). "
        "Mean absolute difference shows the average <i>distance</i> between tendency values (range: 0-2). "
        "High agreement + low difference = strong consistency. High agreement + high difference = agree on verdict but with very different confidence levels. "
        "<b>Note:</b> Statistics only computed for claims where <i>both</i> human and automated provided a value.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.1*inch))

    agreement_data = [
        ['Property', 'Sample Size', 'Agreement Rate', 'Mean Abs Difference'],
    ]

    properties = ['clarity', 'veracity', 'context_coverage', 'intent']
    agreement_rates = summary['agreement_rates']
    mean_diffs = summary['mean_absolute_differences']
    prop_stats = summary['property_statistics']

    for prop in properties:
        prop_display = prop.replace('_', ' ').title()
        agreement = agreement_rates.get(prop)
        diff = mean_diffs.get(prop)

        # Get sample size (total claims with both human and automated values)
        sample_size = prop_stats.get(prop, {}).get('total', 0)
        no_data = prop_stats.get(prop, {}).get('no_data', 0)
        sample_str = f"{sample_size} / {sample_size + no_data}"

        agreement_str = f"{agreement:.1%}" if agreement is not None else "N/A"
        diff_str = f"{diff:.3f}" if diff is not None else "N/A"

        agreement_data.append([prop_display, sample_str, agreement_str, diff_str])

    agreement_table = Table(agreement_data, colWidths=[1.5*inch, 1.25*inch, 1.5*inch, 1.5*inch])
    agreement_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a4a4a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(agreement_table)
    story.append(Spacer(1, 0.3*inch))

    # Certainty Comparison
    story.append(Paragraph("Certainty Comparison", heading_style))
    story.append(Paragraph(
        "<b>How to read:</b> Certainty is measured as the absolute value of tendency (range: 0-1). "
        "Higher values indicate stronger conviction (regardless of polarity). A value near 1.0 means very certain, "
        "near 0.0 means uncertain/neutral. This shows whether automated or human annotations tend to be more decisive.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.1*inch))

    automated_certs = summary.get('automated_certainties', {})
    human_certs = summary.get('human_certainties', {})

    certainty_data = [
        ['Property', 'Automated\nCertainty', 'Human\nCertainty', 'Difference\n(H - A)', 'More Certain'],
    ]

    for prop in properties:
        prop_display = prop.replace('_', ' ').title()
        auto_cert = automated_certs.get(prop)
        human_cert = human_certs.get(prop)

        if auto_cert is not None and human_cert is not None:
            diff = human_cert - auto_cert
            more_certain = "Human" if diff > 0.05 else ("Automated" if diff < -0.05 else "Similar")
            certainty_data.append([
                prop_display,
                f"{auto_cert:.3f}",
                f"{human_cert:.3f}",
                f"{diff:+.3f}",
                more_certain
            ])
        else:
            certainty_data.append([prop_display, "N/A", "N/A", "N/A", "N/A"])

    certainty_table = Table(certainty_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1.25*inch])
    certainty_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a4a4a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(certainty_table)
    story.append(PageBreak())

    # All Claims Comparison
    story.append(Paragraph("All Claims Comparison", heading_style))
    story.append(Paragraph(
        "<b>How to read:</b> Each row shows a claim with its automated (A) and aggregated human (H) "
        "tendency values for each property. Tendencies range from -1.0 (strongly negative) to +1.0 (strongly positive), "
        "with 0 indicating neutral/unknown. ✓ indicates agreement on polarity (both in same category per definitions above), "
        "✗ indicates disagreement (different categories). "
        "Note: ✓ does NOT mean identical values - e.g., +0.9 and +0.3 both show ✓ (both positive). "
        "N/A means no data available for that property.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.15*inch))

    comparisons = data['per_claim_comparisons']

    claims_data = [
        ['Claim\nID', 'Clarity\nA/H/✓', 'Veracity\nA/H/✓', 'Context\nA/H/✓', 'Intent\nA/H/✓'],
    ]

    for comparison in comparisons:
        claim_id = str(comparison['claim_id'])

        if not comparison['verdict_available']:
            claims_data.append([claim_id, 'FAIL', '', '', ''])
            continue

        agg_comp = comparison.get('aggregated_comparison')
        if not agg_comp:
            claims_data.append([claim_id, 'N/A', '', '', ''])
            continue

        row = [claim_id]

        for prop in properties:
            if prop not in agg_comp:
                row.append('N/A')
                continue

            comp = agg_comp[prop]
            auto = comp.get('automated_tendency')
            human = comp.get('human_tendency')
            agreement = comp.get('agreement')

            if auto is None or human is None:
                cell = 'N/A'
            else:
                agree_symbol = "✓" if agreement else "✗"
                cell = f"{auto:+.2f}/{human:+.2f}\n{agree_symbol}"

            row.append(cell)

        claims_data.append(row)

    claims_table = Table(claims_data, colWidths=[0.6*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
    claims_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a4a4a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 1), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.white]),
    ]))
    story.append(claims_table)
    story.append(PageBreak())

    # Disagreements
    story.append(Paragraph("Claims with Disagreements", heading_style))
    story.append(Paragraph(
        "<b>How to read:</b> This section lists only claims where automated and human annotations disagree on polarity. "
        "For each disagreement, we show: Auto (automated tendency), Human (aggregated human tendency), "
        "and Diff (absolute difference). Larger differences indicate stronger disagreement.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.15*inch))

    disagreements_found = False

    for comparison in comparisons:
        claim_id = comparison['claim_id']

        if not comparison['verdict_available']:
            continue

        agg_comp = comparison.get('aggregated_comparison')
        if not agg_comp:
            continue

        # Check for disagreements
        disagreements = []

        for prop in properties:
            if prop not in agg_comp:
                continue

            comp = agg_comp[prop]
            if comp.get('agreement') is False:
                disagreements.append({
                    'property': prop,
                    'automated': comp.get('automated_tendency'),
                    'human': comp.get('human_tendency'),
                    'difference': comp.get('difference'),
                })

        if disagreements:
            disagreements_found = True
            story.append(Paragraph(f"<b>Claim {claim_id}:</b>", styles['Normal']))

            for d in disagreements:
                prop_display = d['property'].replace('_', ' ').title()
                auto_str = f"{d['automated']:+.3f}" if d['automated'] is not None else "N/A"
                human_str = f"{d['human']:+.3f}" if d['human'] is not None else "N/A"
                diff_str = f"{d['difference']:.3f}" if d['difference'] is not None else "N/A"

                story.append(Paragraph(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;{prop_display}: Auto={auto_str}, Human={human_str}, Diff={diff_str}",
                    styles['Normal']
                ))

            story.append(Spacer(1, 0.1*inch))

    if not disagreements_found:
        story.append(Paragraph("No disagreements found!", styles['Normal']))

    # Add advanced analyses if available
    if ANALYSIS_AVAILABLE:
        story.append(PageBreak())

        # Run analyses
        try:
            disagreement_analysis = analyze_all_disagreements(data)
            annotator_patterns = analyze_annotator_patterns(data)

            # Check if individual model data is available
            sample_claim = next((c for c in comparisons if c['verdict_available']), None)
            has_model_data = False
            if sample_claim:
                agg_comp = sample_claim.get('aggregated_comparison', {})
                sample_prop = next(iter(agg_comp.values()), {})
                if 'automated_individual_models' in sample_prop and sample_prop['automated_individual_models'] is not None:
                    has_model_data = True
                    model_overall_stats = analyze_inter_model_agreement_overall(data)
                    model_disagreement_stats = analyze_model_agreement_in_disagreement_cases(data)
                    model_comparison_stats = analyze_model_agreement_by_human_agreement(data)
                    model_patterns = analyze_model_specific_patterns(data)

            # Add Inter-Annotator Agreement section
            add_inter_annotator_section(story, styles, heading_style, disagreement_analysis, annotator_patterns, properties)

            # Add Inter-Model Agreement section if data available
            if has_model_data:
                story.append(PageBreak())
                add_inter_model_section(story, styles, heading_style, model_overall_stats, model_disagreement_stats, model_comparison_stats, model_patterns, properties)

        except Exception as e:
            print(f"Warning: Could not add advanced analyses to PDF: {e}", file=sys.stderr)

    # Build PDF
    doc.build(story)
    print(f"\nPDF exported to: {output_path}")


def export_csv(data: dict, output_path: Path):
    """Export aggregated comparisons to CSV."""
    import csv

    comparisons = data['per_claim_comparisons']

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'claim_id',
            'clarity_auto', 'clarity_human', 'clarity_diff', 'clarity_agree',
            'veracity_auto', 'veracity_human', 'veracity_diff', 'veracity_agree',
            'context_auto', 'context_human', 'context_diff', 'context_agree',
            'intent_auto', 'intent_human', 'intent_diff', 'intent_agree',
        ])

        # Data rows
        for comparison in comparisons:
            if not comparison['verdict_available']:
                continue

            agg_comp = comparison.get('aggregated_comparison')
            if not agg_comp:
                continue

            row = [comparison['claim_id']]

            properties = ['clarity', 'veracity', 'context_coverage', 'intent']

            for prop in properties:
                if prop not in agg_comp:
                    row.extend([None, None, None, None])
                    continue

                comp = agg_comp[prop]
                row.extend([
                    comp.get('automated_tendency'),
                    comp.get('human_tendency'),
                    comp.get('difference'),
                    comp.get('agreement'),
                ])

            writer.writerow(row)

    print(f"\nExported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Present comparison results in readable format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        'results_file',
        type=str,
        help='Path to comparison results JSON file'
    )

    parser.add_argument(
        '--format',
        choices=['summary', 'detailed', 'compact', 'disagreements', 'all'],
        default='all',
        help='Output format (default: all)'
    )

    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help='Export to CSV file'
    )

    parser.add_argument(
        '--pdf',
        type=str,
        default=None,
        help='Export to PDF file'
    )

    args = parser.parse_args()

    # Load results
    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"Error: File not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    data = load_results(results_path)

    # Print requested format
    if args.format in ['summary', 'all']:
        print_summary_table(data)

    if args.format in ['compact', 'all']:
        print_compact_table(data)

    if args.format in ['disagreements', 'all']:
        print_disagreements_only(data)

    if args.format == 'detailed':
        print_detailed_comparisons(data)

    # Export to CSV if requested
    if args.csv:
        export_csv(data, Path(args.csv))

    # Export to PDF if requested
    if args.pdf:
        export_pdf(data, Path(args.pdf))


if __name__ == '__main__':
    main()
