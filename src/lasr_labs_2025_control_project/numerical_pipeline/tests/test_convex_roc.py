"""Tests for ROC curve convexity functionality."""

import numpy as np

from lasr_labs_2025_control_project.numerical_pipeline.utils import (
    calculate_tpr_fpr_from_scores,
    make_roc_curve_convex,
)


def test_make_roc_curve_convex_basic():
    """Test basic convex hull functionality."""
    # Create a simple non-convex ROC curve
    fpr_values = [
        0.0,
        0.2,
        0.1,
        0.8,
        1.0,
    ]  # Note: 0.1 comes after 0.2, making it non-convex
    tpr_values = [0.0, 0.4, 0.3, 0.9, 1.0]

    convex_fpr, convex_tpr = make_roc_curve_convex(fpr_values, tpr_values)

    # Check that we get valid ROC points
    assert len(convex_fpr) == len(convex_tpr)
    assert all(0 <= x <= 1 for x in convex_fpr)
    assert all(0 <= x <= 1 for x in convex_tpr)

    # Check that FPR values are sorted
    assert convex_fpr == sorted(convex_fpr)

    # Check that (0,0) and (1,1) are included if they were in original
    if 0.0 in fpr_values and 0.0 in tpr_values:
        assert 0.0 in convex_fpr and 0.0 in convex_tpr
    if 1.0 in fpr_values and 1.0 in tpr_values:
        assert 1.0 in convex_fpr and 1.0 in convex_tpr


def test_convex_hull_removes_dominated_points():
    """Test that convex hull removes dominated points."""
    # Point (0.5, 0.3) should be dominated by the line from (0,0) to (1,1)
    fpr_values = [0.0, 0.5, 1.0]
    tpr_values = [0.0, 0.3, 1.0]

    convex_fpr, convex_tpr = make_roc_curve_convex(fpr_values, tpr_values)

    # The dominated point should be removed, leaving only (0,0) and (1,1)
    assert len(convex_fpr) == 2
    assert (0.0, 1.0) == (min(convex_fpr), max(convex_fpr))


def test_calculate_tpr_fpr_from_scores_convexity():
    """Test that calculate_tpr_fpr_from_scores returns convex curves."""
    # Create some sample scores
    positive_scores = [0.8, 0.9, 0.7, 0.95, 0.85]
    negative_scores = [0.2, 0.3, 0.1, 0.4, 0.25]

    tpr_values, fpr_values = calculate_tpr_fpr_from_scores(
        positive_scores, negative_scores
    )

    # Basic validation
    assert len(tpr_values) == len(fpr_values)
    assert all(0 <= x <= 1 for x in tpr_values)
    assert all(0 <= x <= 1 for x in fpr_values)

    # Check that curve is convex by verifying it's the same when passed through convex hull again
    convex_fpr_again, convex_tpr_again = make_roc_curve_convex(fpr_values, tpr_values)

    # Should be approximately the same (allowing for floating point precision)
    assert len(convex_fpr_again) == len(fpr_values)
    assert np.allclose(convex_fpr_again, fpr_values, atol=1e-10)
    assert np.allclose(convex_tpr_again, tpr_values, atol=1e-10)


def test_perfect_classifier():
    """Test convex hull with perfect classifier scores."""
    positive_scores = [0.9, 0.95, 0.98, 1.0]
    negative_scores = [0.0, 0.1, 0.2, 0.3]

    tpr_values, fpr_values = calculate_tpr_fpr_from_scores(
        positive_scores, negative_scores
    )

    # Should include (0,0) at high threshold and (1,1) at low threshold
    assert 0.0 in fpr_values
    assert 0.0 in tpr_values
    assert 1.0 in tpr_values  # Perfect recall should be achievable
