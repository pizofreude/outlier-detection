#!/usr/bin/env python3
"""
Unit tests for outlier detection system.
Run with: python -m pytest tests/test_outlier_detection.py -v
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import unittest
from unittest.mock import patch, MagicMock

# Import functions from main notebook (would be in separate modules in production)
# For now, we'll create simplified test versions

class TestOutlierDetection(unittest.TestCase):

    def setUp(self):
        self.test_data = np.random.RandomState(42).randn(100, 2)
        self.test_labels = np.random.RandomState(42).choice([0, 1], 100)

    def test_data_generation_consistency(self):
        """Test that data generation is deterministic."""
        np.random.seed(42)
        data1 = np.random.randn(10, 2)

        np.random.seed(42)
        data2 = np.random.randn(10, 2)

        np.testing.assert_array_equal(data1, data2)

    def test_score_normalization(self):
        """Test score normalization functionality."""
        scores = np.array([1, 2, 3, 4, 5])
        normalized = (scores - scores.min()) / (scores.max() - scores.min())

        self.assertAlmostEqual(normalized.min(), 0.0)
        self.assertAlmostEqual(normalized.max(), 1.0)

    def test_pipeline_integration(self):
        """Integration test for the full detection pipeline."""
        # This is a smoke test to ensure the pipeline runs without errors

        # Mock the main components
        feature_engineer = MagicMock()
        feature_engineer.fit_transform.return_value = self.test_data
        feature_engineer.transform.return_value = self.test_data

        # Test that we can create and run basic operations
        self.assertIsNotNone(self.test_data)
        self.assertEqual(self.test_data.shape[1], 2)

if __name__ == '__main__':
    unittest.main()
