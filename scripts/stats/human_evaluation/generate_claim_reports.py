#!/usr/bin/env python3
"""
Generate individual PDF reports for each claim showing human vs automated annotations.

Creates one PDF per claim with:
- Claim text and context
- Link to fact-check article (if available)
- Sections for each property (clarity, veracity, context_coverage, intent)
- Human annotations with comments
- Model annotations with reasoning

Usage:
    python scripts/compare_automatic_human/generate_claim_reports.py annotations_export.zip comparison_results.json -o output_folder
    python scripts/compare_automatic_human/generate_claim_reports.py annotations_export.zip comparison_results.json -o output_folder --disagreements-only
"""

import argparse
import json
import logging
from pathlib import Path
from zipfile import ZipFile

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


# Value display mappings
CLARITY_VALUES = {
    'clear': 'Clear',
    'vague': 'Vague',
    'unclear': 'Unclear'
}

VERACITY_VALUES = {
    'true': 'True',
    'false': 'False',
    'mixture': 'Mixture',
    'unknown': 'Unknown'
}

INTENT_VALUES = {
    'legitimate': 'Legitimate',
    'illegitimate': 'Illegitimate',
    'unclear': 'Unclear'
}

CONTEXT_COVERAGE_VALUES = {
    'sufficient': 'Sufficient',
    'insufficient': 'Insufficient'
}

MEDIA_AUTHENTICITY_VALUES = {
    'pristine': 'Pristine',
    'edited': 'Edited',
    'fabricated': 'Fabricated',
    'unknown': 'Unknown'
}

MEDIA_CONTEXTUALIZATION_VALUES = {
    'correct': 'Correct',
    'incorrect': 'Incorrect',
    'unrelated': 'Unrelated'
}


def load_data(zip_path: Path) -> tuple[dict, dict, dict]:
    """Load claims, annotations, and pipeline data from the export zip."""
    logger.info(f"Loading data from {zip_path}")

    with ZipFile(zip_path, 'r') as zip_file:
        # Load claims
        with zip_file.open('claims.json') as f:
            claims_data = json.load(f)
            claims = {c['claim_id']: c for c in claims_data['claims']}

        # Load annotations
        with zip_file.open('annotations.json') as f:
            annotations_data = json.load(f)
            annotations_by_claim = annotations_data['grouped_by_claim']

        # Load pipeline data (for fact-check articles)
        with zip_file.open('pipeline_data.json') as f:
            pipeline_data = json.load(f)

            # Build articles lookup
            articles_by_id = {a['id']: a for a in pipeline_data.get('articles', [])}

            # Group reviews by claim_id and attach article info
            pipeline_by_claim = {}
            for review in pipeline_data.get('reviews', []):
                claim_id = review.get('claim_id')
                if claim_id:
                    # Get article for this review
                    article = articles_by_id.get(review.get('article_id'))

                    if claim_id not in pipeline_by_claim:
                        pipeline_by_claim[claim_id] = {'reviews': []}

                    review_with_article = review.copy()
                    if article:
                        review_with_article['article'] = article

                    pipeline_by_claim[claim_id]['reviews'].append(review_with_article)

    logger.info(f"Loaded {len(claims)} claims, {len(annotations_by_claim)} annotation groups")
    return claims, annotations_by_claim, pipeline_by_claim


def load_comparison_results(results_path: Path) -> dict:
    """Load comparison results."""
    logger.info(f"Loading comparison results from {results_path}")
    with open(results_path, 'r') as f:
        return json.load(f)


def has_disagreements(claim_comparison: dict) -> bool:
    """Check if a claim has any disagreements between human and automated annotations."""
    if not claim_comparison.get('verdict_available'):
        return False

    # Check individual comparisons
    for individual in claim_comparison.get('individual_comparisons', []):
        for prop_name, comparison in individual.get('comparisons', {}).items():
            if not comparison.get('agreement', True):
                return True

    # Check aggregated comparison
    aggregated = claim_comparison.get('aggregated_comparison', {})
    for prop_name, comparison in aggregated.items():
        if comparison and not comparison.get('agreement', True):
            return True

    return False


def create_styles():
    """Create custom paragraph styles."""
    styles = getSampleStyleSheet()

    # Title style
    styles.add(ParagraphStyle(
        name='ClaimTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=12,
        alignment=TA_CENTER
    ))

    # Section header style
    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2c5aa0'),
        spaceAfter=8,
        spaceBefore=12,
        borderPadding=(0, 0, 5, 0),
        borderColor=colors.HexColor('#2c5aa0'),
        borderWidth=2,
        borderRadius=2
    ))

    # Subsection header
    styles.add(ParagraphStyle(
        name='SubsectionHeader',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#444444'),
        spaceAfter=6,
        spaceBefore=8
    ))

    # Claim text style
    styles.add(ParagraphStyle(
        name='ClaimText',
        parent=styles['BodyText'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=12,
        borderPadding=10,
        borderColor=colors.HexColor('#e0e0e0'),
        borderWidth=1,
        backColor=colors.HexColor('#f9f9f9')
    ))

    # Annotation style
    styles.add(ParagraphStyle(
        name='Annotation',
        parent=styles['BodyText'],
        fontSize=10,
        spaceAfter=8,
        leftIndent=20
    ))

    # Comment style
    styles.add(ParagraphStyle(
        name='Comment',
        parent=styles['BodyText'],
        fontSize=9,
        textColor=colors.HexColor('#555555'),
        spaceAfter=6,
        leftIndent=30,
        fontName='Helvetica-Oblique'
    ))

    return styles


def tendency_to_label(tendency: float, property_name: str) -> str:
    """Convert tendency value to human-readable label."""
    if property_name == 'clarity':
        if tendency > 0.5:
            return 'Clear'
        elif tendency < -0.5:
            return 'Unclear'
        else:
            return 'Vague'
    elif property_name == 'veracity':
        if tendency > 0.6:
            return 'True'
        elif tendency < -0.6:
            return 'False'
        elif abs(tendency) < 0.3:
            return 'Mixture'
        else:
            return 'Leaning ' + ('True' if tendency > 0 else 'False')
    elif property_name == 'context_coverage':
        return 'Sufficient' if tendency > 0 else 'Insufficient'
    elif property_name == 'intent':
        if tendency > 0.5:
            return 'Legitimate'
        elif tendency < -0.5:
            return 'Illegitimate'
        else:
            return 'Unclear'
    elif property_name == 'media_authenticity' or property_name == 'authenticity':
        if tendency > 0.6:
            return 'Pristine'
        elif tendency < -0.3:
            return 'Fabricated'
        else:
            return 'Edited'
    elif property_name == 'media_contextualization' or property_name == 'contextualization':
        if tendency > 0.5:
            return 'Correct'
        elif tendency < -0.3:
            return 'Incorrect'
        else:
            return 'Unclear/Unrelated'
    return f"{tendency:.2f}"


def format_confidence(confidence: int) -> str:
    """Format confidence level."""
    labels = {0: 'Very Uncertain', 1: 'Uncertain', 2: 'Somewhat Certain', 3: 'Certain'}
    return labels.get(confidence, f'Level {confidence}')


def escape_xml(text: str) -> str:
    """Escape XML special characters for reportlab."""
    if text is None:
        return ''
    text = str(text)
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    return text


def generate_claim_report(
    claim_id: int,
    claim_data: dict,
    annotations: list[dict],
    comparison: dict,
    pipeline_data: dict,
    output_path: Path,
    styles: dict
):
    """Generate a PDF report for a single claim."""

    # Create PDF document
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )

    story = []

    # Title
    story.append(Paragraph(f"Claim #{claim_id} - Analysis Report", styles['ClaimTitle']))
    story.append(Spacer(1, 0.2*inch))

    # Claim text
    claim_text = escape_xml(claim_data.get('text', 'No claim text available'))
    story.append(Paragraph("<b>Claim:</b>", styles['SubsectionHeader']))
    story.append(Paragraph(claim_text, styles['ClaimText']))
    story.append(Spacer(1, 0.15*inch))

    # Fact-check article link (if available)
    reviews = pipeline_data.get('reviews', [])
    if reviews:
        review = reviews[0]  # Take first review
        article = review.get('article', {})
        article_url = article.get('url', '')
        if article_url:
            story.append(Paragraph(
                f'<b>Fact-check article:</b> <link href="{escape_xml(article_url)}" color="blue">{escape_xml(article_url)}</link>',
                styles['Annotation']
            ))
            story.append(Spacer(1, 0.15*inch))

    # Appearances (social media posts)
    appearances = claim_data.get('context', {}).get('appearances', [])
    if appearances:
        story.append(Paragraph("<b>Appearances:</b>", styles['SubsectionHeader']))
        for i, app in enumerate(appearances[:5], 1):  # Limit to first 5
            url = app.get('url', 'No URL')
            story.append(Paragraph(
                f'{i}. <link href="{escape_xml(url)}" color="blue">{escape_xml(url)}</link>',
                styles['Annotation']
            ))
        if len(appearances) > 5:
            story.append(Paragraph(f'<i>... and {len(appearances) - 5} more</i>', styles['Comment']))
        story.append(Spacer(1, 0.2*inch))

    # Properties sections
    properties = ['clarity', 'veracity', 'context_coverage', 'intent']
    property_labels = {
        'clarity': 'Clarity',
        'veracity': 'Veracity',
        'context_coverage': 'Context Coverage',
        'intent': 'Intent'
    }

    for prop in properties:
        story.append(Paragraph(f"═══ {property_labels[prop]} ═══", styles['SectionHeader']))
        story.append(Spacer(1, 0.1*inch))

        # Human Annotations Section
        story.append(Paragraph("<b>Human Annotations:</b>", styles['SubsectionHeader']))

        human_annotations = []
        for ann in annotations:
            if ann.get('status') != 'completed':
                continue

            claim_level = ann.get('claim_level', {})
            prop_data = claim_level.get(prop)

            if prop_data:
                annotator = ann.get('annotator', {})
                email = annotator.get('email', 'Unknown')
                value = prop_data.get('value', 'N/A')
                confidence = prop_data.get('confidence', 0)
                uncertainty = prop_data.get('uncertainty_rationale', '')

                human_annotations.append({
                    'email': email,
                    'value': value,
                    'confidence': confidence,
                    'uncertainty': uncertainty
                })

        if human_annotations:
            for h_ann in human_annotations:
                value_display = h_ann['value'].title() if h_ann['value'] else 'N/A'
                conf_display = format_confidence(h_ann['confidence'])

                story.append(Paragraph(
                    f'<b>•</b> <b>{escape_xml(h_ann["email"])}</b>: {value_display} '
                    f'(Confidence: {conf_display})',
                    styles['Annotation']
                ))

                if h_ann['uncertainty']:
                    story.append(Paragraph(
                        f'<i>Comment:</i> {escape_xml(h_ann["uncertainty"])}',
                        styles['Comment']
                    ))
        else:
            story.append(Paragraph('<i>No human annotations available</i>', styles['Comment']))

        story.append(Spacer(1, 0.15*inch))

        # Automated Annotations Section
        story.append(Paragraph("<b>Automated Annotations:</b>", styles['SubsectionHeader']))

        # Get automated data from comparison
        if comparison.get('verdict_available') and comparison.get('individual_comparisons'):
            # Use first individual comparison to get automated data
            first_comp = comparison['individual_comparisons'][0]
            prop_comparison = first_comp.get('comparisons', {}).get(prop, {})

            # Aggregated ensemble result
            auto_tendency = prop_comparison.get('automated_tendency')
            auto_explanation = prop_comparison.get('automated_explanation', '')

            if auto_tendency is not None:
                label = tendency_to_label(auto_tendency, prop)
                story.append(Paragraph(
                    f'<b>Ensemble Result:</b> {label} (tendency: {auto_tendency:.3f})',
                    styles['Annotation']
                ))

                if auto_explanation:
                    story.append(Paragraph(
                        f'<i>Explanation:</i> {escape_xml(auto_explanation)}',
                        styles['Comment']
                    ))

                story.append(Spacer(1, 0.1*inch))

                # Individual models
                individual_models = prop_comparison.get('automated_individual_models', [])
                if individual_models:
                    story.append(Paragraph('<b>Individual Models:</b>', styles['Annotation']))

                    for model_data in individual_models:
                        model_name = model_data.get('model', 'Unknown')
                        model_tendency = model_data.get('tendency')
                        model_explanation = model_data.get('explanation', '')

                        if model_tendency is not None:
                            model_label = tendency_to_label(model_tendency, prop)
                            story.append(Paragraph(
                                f'  <b>→ {escape_xml(model_name)}:</b> {model_label} '
                                f'(tendency: {model_tendency:.3f})',
                                styles['Annotation']
                            ))

                            if model_explanation:
                                # Truncate very long explanations
                                if len(model_explanation) > 500:
                                    model_explanation = model_explanation[:497] + '...'
                                story.append(Paragraph(
                                    f'    <i>{escape_xml(model_explanation)}</i>',
                                    styles['Comment']
                                ))
            else:
                story.append(Paragraph('<i>No automated annotation available</i>', styles['Comment']))
        else:
            story.append(Paragraph('<i>Verdict not available or failed</i>', styles['Comment']))

        story.append(Spacer(1, 0.25*inch))

    # Media Properties Section
    # Check if claim has media items or if any annotations have media evaluations
    media_items = claim_data.get('media', [])
    has_human_media_anns = any(ann.get('media_evaluations') for ann in annotations if ann.get('status') == 'completed')

    if media_items or has_human_media_anns:
        story.append(Paragraph("═══════════════════════", styles['SectionHeader']))
        story.append(Paragraph("MEDIA-LEVEL PROPERTIES", styles['SectionHeader']))
        story.append(Paragraph("═══════════════════════", styles['SectionHeader']))
        story.append(Spacer(1, 0.15*inch))

        # Get automated media comparisons
        automated_media_by_ref = {}
        if comparison.get('verdict_available') and comparison.get('individual_comparisons'):
            first_comp = comparison['individual_comparisons'][0]
            media_comparisons = first_comp.get('comparisons', {}).get('media', [])
            for mc in media_comparisons:
                media_ref = mc.get('media_id', '')
                automated_media_by_ref[media_ref] = mc

        # Collect human annotations by media evaluation ID
        human_media_annotations = {}
        for ann in annotations:
            if ann.get('status') != 'completed':
                continue

            media_evals = ann.get('media_evaluations', {})
            for media_eval_id, evals in media_evals.items():
                if media_eval_id not in human_media_annotations:
                    human_media_annotations[media_eval_id] = []

                annotator = ann.get('annotator', {})
                human_media_annotations[media_eval_id].append({
                    'email': annotator.get('email', 'Unknown'),
                    'authenticity': evals.get('media_authenticity', {}),
                    'contextualization': evals.get('media_contextualization', {})
                })

        # Display each media item from the claim
        for media_item in media_items:
            media_id = media_item.get('media_id')
            media_type = media_item.get('type', 'unknown')
            media_filename = media_item.get('filename', f'media_{media_id}')

            # Construct the media reference used in automated comparisons
            media_ref = f"<{media_type}:{media_id}>"

            story.append(Paragraph(
                f"<b>Media Item: {escape_xml(media_filename)} ({media_type})</b>",
                styles['SubsectionHeader']
            ))
            story.append(Spacer(1, 0.1*inch))

            # Get automated comparison for this media item
            media_comp = automated_media_by_ref.get(media_ref)

            # Media Authenticity
            story.append(Paragraph("<b>Media Authenticity:</b>", styles['Annotation']))

            # Human annotations (show all human annotations for this claim's media)
            # Note: We show all human media annotations here since we can't always perfectly map
            # the evaluation IDs to specific media files
            if human_media_annotations:
                story.append(Paragraph("<i>Human Annotations:</i>", styles['Comment']))
                for media_eval_id, media_anns in human_media_annotations.items():
                    for m_ann in media_anns:
                        auth = m_ann['authenticity']
                        if auth:
                            value = auth.get('value', 'N/A')
                            confidence = auth.get('confidence', 0)
                            tags = auth.get('tags', [])
                            uncertainty = auth.get('uncertainty_rationale', '')

                            value_display = MEDIA_AUTHENTICITY_VALUES.get(value, value.title() if value else 'N/A')
                            conf_display = format_confidence(confidence)

                            tags_display = f" [Tags: {', '.join(tags)}]" if tags else ""

                            story.append(Paragraph(
                                f'<b>•</b> <b>{escape_xml(m_ann["email"])}</b>: {value_display} '
                                f'(Confidence: {conf_display}){tags_display}',
                                styles['Annotation']
                            ))

                            if uncertainty:
                                story.append(Paragraph(
                                    f'<i>Comment:</i> {escape_xml(uncertainty)}',
                                    styles['Comment']
                                ))
            else:
                story.append(Paragraph("<i>No human annotations</i>", styles['Comment']))

            # Automated annotations for media authenticity
            if media_comp:
                auth_comp = media_comp.get('authenticity', {})
                auto_value = auth_comp.get('automated_value')
                individual_models = auth_comp.get('automated_individual_models', [])

                if auto_value is not None:
                    story.append(Paragraph("<i>Automated Annotation:</i>", styles['Comment']))
                    label = tendency_to_label(auto_value, 'authenticity')
                    story.append(Paragraph(
                        f'<b>Ensemble Result:</b> {label} (tendency: {auto_value:.3f})',
                        styles['Annotation']
                    ))

                    # Show individual models if available
                    if individual_models:
                        story.append(Paragraph('<b>Individual Models:</b>', styles['Annotation']))
                        for model_data in individual_models:
                            model_name = model_data.get('model', 'Unknown')
                            model_tendency = model_data.get('tendency')
                            model_explanation = model_data.get('explanation', '')

                            if model_tendency is not None:
                                model_label = tendency_to_label(model_tendency, 'authenticity')
                                story.append(Paragraph(
                                    f'  <b>→ {escape_xml(model_name)}:</b> {model_label} '
                                    f'(tendency: {model_tendency:.3f})',
                                    styles['Annotation']
                                ))

                                if model_explanation:
                                    if len(model_explanation) > 500:
                                        model_explanation = model_explanation[:497] + '...'
                                    story.append(Paragraph(
                                        f'    <i>{escape_xml(model_explanation)}</i>',
                                        styles['Comment']
                                    ))
                    else:
                        story.append(Paragraph(
                            '<i>Note: Individual model predictions not available for media properties</i>',
                            styles['Comment']
                        ))
                else:
                    story.append(Paragraph("<i>No automated annotation available</i>", styles['Comment']))
            else:
                story.append(Paragraph("<i>No automated annotation available</i>", styles['Comment']))

            story.append(Spacer(1, 0.1*inch))

            # Media Contextualization
            story.append(Paragraph("<b>Media Contextualization:</b>", styles['Annotation']))

            # Human annotations
            if human_media_annotations:
                story.append(Paragraph("<i>Human Annotations:</i>", styles['Comment']))
                for media_eval_id, media_anns in human_media_annotations.items():
                    for m_ann in media_anns:
                        context = m_ann['contextualization']
                        if context:
                            value = context.get('value', 'N/A')
                            confidence = context.get('confidence', 0)
                            tags = context.get('tags', [])
                            uncertainty = context.get('uncertainty_rationale', '')

                            value_display = MEDIA_CONTEXTUALIZATION_VALUES.get(value, value.title() if value else 'N/A')
                            conf_display = format_confidence(confidence)

                            tags_display = f" [Tags: {', '.join(tags)}]" if tags else ""

                            story.append(Paragraph(
                                f'<b>•</b> <b>{escape_xml(m_ann["email"])}</b>: {value_display} '
                                f'(Confidence: {conf_display}){tags_display}',
                                styles['Annotation']
                            ))

                            if uncertainty:
                                story.append(Paragraph(
                                    f'<i>Comment:</i> {escape_xml(uncertainty)}',
                                    styles['Comment']
                                ))
            else:
                story.append(Paragraph("<i>No human annotations</i>", styles['Comment']))

            # Automated annotations for media contextualization
            if media_comp:
                context_comp = media_comp.get('contextualization', {})
                auto_value = context_comp.get('automated_value')
                individual_models = context_comp.get('automated_individual_models', [])

                if auto_value is not None:
                    story.append(Paragraph("<i>Automated Annotation:</i>", styles['Comment']))
                    label = tendency_to_label(auto_value, 'contextualization')
                    story.append(Paragraph(
                        f'<b>Ensemble Result:</b> {label} (tendency: {auto_value:.3f})',
                        styles['Annotation']
                    ))

                    # Show individual models if available
                    if individual_models:
                        story.append(Paragraph('<b>Individual Models:</b>', styles['Annotation']))
                        for model_data in individual_models:
                            model_name = model_data.get('model', 'Unknown')
                            model_tendency = model_data.get('tendency')
                            model_explanation = model_data.get('explanation', '')

                            if model_tendency is not None:
                                model_label = tendency_to_label(model_tendency, 'contextualization')
                                story.append(Paragraph(
                                    f'  <b>→ {escape_xml(model_name)}:</b> {model_label} '
                                    f'(tendency: {model_tendency:.3f})',
                                    styles['Annotation']
                                ))

                                if model_explanation:
                                    if len(model_explanation) > 500:
                                        model_explanation = model_explanation[:497] + '...'
                                    story.append(Paragraph(
                                        f'    <i>{escape_xml(model_explanation)}</i>',
                                        styles['Comment']
                                    ))
                    else:
                        story.append(Paragraph(
                            '<i>Note: Individual model predictions not available for media properties</i>',
                            styles['Comment']
                        ))
                else:
                    story.append(Paragraph("<i>No automated annotation available</i>", styles['Comment']))
            else:
                story.append(Paragraph("<i>No automated annotation available</i>", styles['Comment']))

            story.append(Spacer(1, 0.25*inch))

    # Build PDF
    doc.build(story)
    logger.info(f"Generated report for claim {claim_id}: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate individual PDF reports for each claim'
    )
    parser.add_argument(
        'annotations_zip',
        type=Path,
        help='Path to annotations export zip file'
    )
    parser.add_argument(
        'comparison_results',
        type=Path,
        help='Path to comparison results JSON file'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('claim_reports'),
        help='Output directory for PDF reports (default: claim_reports)'
    )
    parser.add_argument(
        '--disagreements-only',
        action='store_true',
        help='Only generate reports for claims with disagreements'
    )

    args = parser.parse_args()

    # Load data
    claims, annotations_by_claim, pipeline_by_claim = load_data(args.annotations_zip)
    comparison_data = load_comparison_results(args.comparison_results)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output}")

    # Create styles
    styles = create_styles()

    # Process each claim
    per_claim_comparisons = comparison_data.get('per_claim_comparisons', [])
    comparison_by_id = {c['claim_id']: c for c in per_claim_comparisons}

    generated_count = 0
    skipped_count = 0

    for claim_id, claim_data in claims.items():
        # Get annotations and comparison for this claim
        annotations = annotations_by_claim.get(str(claim_id), [])
        comparison = comparison_by_id.get(claim_id, {})
        pipeline_data = pipeline_by_claim.get(claim_id, {})

        # Skip if disagreements-only mode and no disagreements
        if args.disagreements_only and not has_disagreements(comparison):
            skipped_count += 1
            continue

        # Generate report
        output_path = args.output / f"claim_{claim_id:04d}.pdf"
        try:
            generate_claim_report(
                claim_id=claim_id,
                claim_data=claim_data,
                annotations=annotations,
                comparison=comparison,
                pipeline_data=pipeline_data,
                output_path=output_path,
                styles=styles
            )
            generated_count += 1
        except Exception as e:
            logger.error(f"Failed to generate report for claim {claim_id}: {e}")
            import traceback
            traceback.print_exc()

    logger.info(f"\n{'='*60}")
    logger.info(f"Report generation complete!")
    logger.info(f"Generated: {generated_count} reports")
    logger.info(f"Skipped: {skipped_count} claims")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
