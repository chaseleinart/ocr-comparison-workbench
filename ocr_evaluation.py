from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics.g_eval import Rubric
from typing import Dict, Any


def evaluate_invoice_ocr(generated_ocr: str, ground_truth_invoice: str) -> Dict[str, Any]:
    """
    G-Eval based evaluation for invoice OCR.

    Metrics:
        - Field Accuracy (invoice header + key-value pairs)
        - Line Item Quality (items, quantities, unit prices)
        - Financial Consistency (subtotals, tax, discounts, grand total)
    """
    try:
        # Single test case describing the task
        test_case = LLMTestCase(
            input=(
                "Invoice OCR Extraction Task.\n"
                "Compare the OCR output with the ground truth invoice. "
                "Focus on correctness of header fields, line items, and totals."
            ),
            actual_output=generated_ocr,
            expected_output=ground_truth_invoice,
        )

        # 1) Field Accuracy Metric (vendor name, invoice number, dates, addresses, etc.)
        field_accuracy_metric = GEval(
            name="Invoice Field Accuracy",
            criteria=(
                "Evaluate how accurately the OCR output captures key invoice fields: "
                "supplier name, buyer name, invoice number, invoice date, tax IDs, "
                "addresses, and other header key-value pairs."
            ),
            evaluation_steps=[
                "Identify key header fields in the ground truth (e.g., supplier/buyer, invoice number, date, tax IDs, addresses).",
                "Check if each of these fields is present in the OCR output.",
                "Verify the correctness of the values (especially IDs, dates, and addresses).",
                "Penalize missing or hallucinated header fields.",
            ],
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            rubric=[
                Rubric(
                    score_range=(0, 2),
                    expected_outcome=(
                        "Most key invoice fields are missing or incorrect; header information is unreliable."
                    ),
                ),
                Rubric(
                    score_range=(3, 5),
                    expected_outcome=(
                        "Some key fields are correct, but several important header fields "
                        "(like invoice number, date, or tax IDs) are missing or wrong."
                    ),
                ),
                Rubric(
                    score_range=(6, 8),
                    expected_outcome=(
                        "Most key header fields are present and correct with only minor discrepancies."
                    ),
                ),
                Rubric(
                    score_range=(9, 10),
                    expected_outcome=(
                        "All important header fields are present and correct; header section is highly reliable."
                    ),
                ),
            ],
            threshold=0.7,
        )

        # 2) Line Item Quality Metric (description, qty, unit price, per-row totals)
        line_item_metric = GEval(
            name="Invoice Line Item Quality",
            criteria=(
                "Evaluate how well the OCR output captures line items, including item descriptions, "
                "quantities, unit prices, and per-item totals."
            ),
            evaluation_steps=[
                "Identify line items in the ground truth invoice (rows of an items table).",
                "Check that the same number of line items exist in the OCR output.",
                "For each line, compare description, quantity, unit price, and per-item total.",
                "Penalize missing rows, merged/split rows, or hallucinated items.",
            ],
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            rubric=[
                Rubric(
                    score_range=(0, 2),
                    expected_outcome=(
                        "Line items are largely incorrect: many rows missing, spurious rows, "
                        "or values unusable for downstream processing."
                    ),
                ),
                Rubric(
                    score_range=(3, 5),
                    expected_outcome=(
                        "Some line items are captured, but there are multiple errors in quantities, "
                        "prices, or item descriptions."
                    ),
                ),
                Rubric(
                    score_range=(6, 8),
                    expected_outcome=(
                        "Most line items are present and correct, with only minor mistakes."
                    ),
                ),
                Rubric(
                    score_range=(9, 10),
                    expected_outcome=(
                        "All or almost all line items match ground truth in count and values; "
                        "line-level data is highly reliable."
                    ),
                ),
            ],
            threshold=0.7,
        )

        # 3) Financial Consistency Metric (subtotal, tax, discounts, grand total)
        financial_metric = GEval(
            name="Invoice Financial Consistency",
            criteria=(
                "Evaluate whether financial fields (subtotals, taxes, discounts, shipping, "
                "grand total, currency) are correctly captured and internally consistent."
            ),
            evaluation_steps=[
                "Identify financial fields in the ground truth (subtotals, taxes, discounts, shipping, grand total, currency).",
                "Check presence and correctness of these fields in the OCR output.",
                "Verify that numeric amounts match the ground truth (not approximated).",
                "Penalize hallucinated financial fields or mismatched totals.",
            ],
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            rubric=[
                Rubric(
                    score_range=(0, 2),
                    expected_outcome=(
                        "Most financial fields (subtotal, tax, total, etc.) are missing or incorrect; "
                        "totals cannot be trusted."
                    ),
                ),
                Rubric(
                    score_range=(3, 5),
                    expected_outcome=(
                        "Some financial fields are correct but key ones like grand total or tax "
                        "have noticeable errors."
                    ),
                ),
                Rubric(
                    score_range=(6, 8),
                    expected_outcome=(
                        "Financial fields are mostly correct with only small discrepancies."
                    ),
                ),
                Rubric(
                    score_range=(9, 10),
                    expected_outcome=(
                        "All financial fields and totals are present, correct, and consistent with the items."
                    ),
                ),
            ],
            threshold=0.7,
        )

        # Run evaluation
        metrics = [field_accuracy_metric, line_item_metric, financial_metric]
        for metric in metrics:
            metric.measure(test_case)

        # Aggregate scores
        overall_score = (
            field_accuracy_metric.score
            + line_item_metric.score
            + financial_metric.score
        ) / 3

        detailed_metrics = {
            "field_accuracy": {
                "score": field_accuracy_metric.score,
                "reason": field_accuracy_metric.reason,
            },
            "line_item_quality": {
                "score": line_item_metric.score,
                "reason": line_item_metric.reason,
            },
            "financial_consistency": {
                "score": financial_metric.score,
                "reason": financial_metric.reason,
            },
        }

        return {
            "overall_score": overall_score,
            "detailed_metrics": detailed_metrics,
            "passed": overall_score >= 0.7,
        }

    except Exception as e:
        return {
            "error": f"Error evaluating invoice OCR: {str(e)}",
            "overall_score": 0.0,
            "detailed_metrics": {},
            "passed": False,
        }
