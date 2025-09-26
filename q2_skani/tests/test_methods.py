# ----------------------------------------------------------------------------
# Copyright (c) 2025, Bokulich Lab.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import subprocess
from typing import List
from unittest.mock import patch, MagicMock

import pandas as pd
import skbio
from q2_types.feature_data_mag import MAGSequencesDirFmt
from qiime2.plugin.testing import TestPluginBase

from q2_skani.skani import (
    compare_seqs,
    _construct_triangle_cmd,
    _process_skani_matrix,
    _run_skani,
)


class SkaniTests(TestPluginBase):
    package = "q2_skani.tests"

    def test_construct_triangle_cmd_minimal(self):
        """Test command construction with minimal required parameters."""
        cmd = _construct_triangle_cmd(
            fasta_list="genome_list.txt",
            output_file="output.tsv",
            skani_args={},
        )
        expected = [
            "skani",
            "triangle",
            "-v",
            "--distance",
            "-l",
            "genome_list.txt",
            "-o",
            "output.tsv",
        ]
        self.assertListEqual(cmd, expected)

    def test_construct_triangle_cmd_with_all_parameters(self):
        """Test command construction with all optional parameters."""
        cmd = _construct_triangle_cmd(
            fasta_list="genome_list.txt",
            output_file="output.tsv",
            skani_args={
                "threads": 4,
                "min_af": 20.0,
                "compression": 150,
                "marker_c": 2000,
                "screen": 85.0,
                "ci": True,
                "detailed": True,
                "diagonal": True,
                "sparse": True,
                "full_matrix": True,
                "median": True,
                "no_learned_ani": True,
                "robust": True,
                "faster_small": True,
                "preset": "fast",
            },
        )

        # Define expected components
        expected_basic = [
            "skani",
            "triangle",
            "-l",
            "genome_list.txt",
            "-o",
            "output.tsv",
        ]

        expected_params = [
            ("-t", "4"),
            ("--min-af", "20.0"),
            ("-c", "150"),
            ("-m", "2000"),
            ("-s", "85.0"),
        ]

        expected_boolean_flags = [
            "--ci",
            "--detailed",
            "--diagonal",
            "--sparse",
            "--full-matrix",
            "--median",
            "--no-learned-ani",
            "--robust",
            "--faster-small",
            "--fast",
        ]

        # Check all basic components
        for component in expected_basic:
            self.assertIn(component, cmd)

        # Check all parameter flags with their values
        for flag, value in expected_params:
            self.assertIn(flag, cmd)
            self.assertIn(value, cmd)

        # Check all boolean flags
        for flag in expected_boolean_flags:
            self.assertIn(flag, cmd)

    def test_construct_triangle_cmd_with_preset(self):
        """Test command construction with different preset values."""
        presets = ["fast", "medium", "slow", "small-genomes"]

        for preset in presets:
            cmd = _construct_triangle_cmd(
                fasta_list="genome_list.txt",
                output_file="output.tsv",
                skani_args={"preset": preset},
            )
            self.assertIn(f"--{preset}", cmd)

    @patch("subprocess.run")
    def test_run_skani_success(self, mock_subprocess):
        """Test successful skani execution."""
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="Success",
            stderr="",
        )

        cmd = ["skani", "triangle", "-l", "test.txt"]
        _run_skani(cmd)

        mock_subprocess.assert_called_once_with(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )

    @patch("subprocess.run")
    def test_run_skani_failure(self, mock_subprocess):
        """Test skani execution failure handling."""

        # Mock failed subprocess run
        error = subprocess.CalledProcessError(returncode=1, cmd=["skani", "triangle"])
        error.stdout = "Error output"
        error.stderr = "Error details"
        mock_subprocess.side_effect = error

        cmd = ["skani", "triangle", "-l", "test.txt"]
        with self.assertRaises(RuntimeError) as cm:
            _run_skani(cmd)

        error_msg = str(cm.exception)
        self.assertIn("Skani failed with exit code 1", error_msg)
        self.assertIn("Error output", error_msg)
        self.assertIn("Error details", error_msg)

    def test_process_skani_matrix(self):
        """Test processing of skani matrix output."""
        matrix_file = self.get_data_path("test.matrix")

        df_obs = _process_skani_matrix(str(matrix_file))
        df_exp = pd.DataFrame({
            "genome1": [0.0, 0.1, 0.2],
            "genome2": [0.1, 0.0, 0.3],
            "genome3": [0.2, 0.3, 0.0]
        },
            index = pd.Index(["genome1", "genome2", "genome3"], name="id")
        )
        pd.testing.assert_frame_equal(df_obs, df_exp)

    def test_process_skani_matrix_with_na(self):
        """Test processing of skani matrix with NA values."""
        matrix_file = self.get_data_path("test_na.matrix")

        df = _process_skani_matrix(str(matrix_file))

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (3, 3))
        self.assertTrue(pd.isna(df.loc["genome1", "genome2"]))
        self.assertTrue(pd.isna(df.loc["genome2", "genome1"]))
        self.assertEqual(df.loc["genome1", "genome3"], 0.2)

    def test_process_skani_matrix_invalid_file(self):
        """Test error handling for invalid matrix file."""
        matrix_file = self.get_data_path("test_invalid.matrix")

        # Test that processing raises an error
        with self.assertRaises(Exception):  # May raise various exceptions
            _process_skani_matrix(str(matrix_file))

    @patch("q2_skani.skani._run_skani")
    def test_compare_seqs(self, mock_run):
        """Test the main compare_seqs function."""
        mags = MAGSequencesDirFmt(self.get_data_path("mags-fake"), "r")

        def mock_run_skani(cmd: List[str]):
            # Create mock output file based on the command
            output_idx = cmd.index("-o") + 1
            output_file = cmd[output_idx]

            # Copy the test matrix file to the output location
            import shutil

            shutil.copy(self.get_data_path("test.matrix"), output_file)

        mock_run.side_effect = mock_run_skani
        result = compare_seqs(
            genomes=mags,
            threads=4,
            min_af=20.0,
            compression=150,
            full_matrix=True,
            preset="fast",
        )

        # Check the results
        self.assertIsInstance(result, skbio.DistanceMatrix)
        self.assertEqual(result.shape, (3, 3))
        self.assertEqual(list(result.ids), ["genome1", "genome2", "genome3"])
        self.assertEqual(result["genome1", "genome2"], 0.1)
        self.assertEqual(result["genome2", "genome3"], 0.3)

    @patch("q2_skani.skani._run_skani")
    def test_compare_seqs_no_fasta_files(self, mock_run):
        """Test error handling when no FASTA files are found."""
        mags = MAGSequencesDirFmt()

        def mock_run_skani(cmd: List[str]):
            # Check if the genome list file is empty
            list_file_idx = cmd.index("-l") + 1
            list_file = cmd[list_file_idx]

            with open(list_file, "r") as f:
                content = f.read().strip()
                if not content:
                    raise RuntimeError("No input files provided to skani")

        mock_run.side_effect = mock_run_skani
        with self.assertRaisesRegex(RuntimeError, "Failed to run Skani comparison"):
            compare_seqs(genomes=mags)

    @patch("q2_skani.skani._run_skani")
    def test_compare_seqs_skani_error(self, mock_run):
        """Test error handling when skani fails."""
        mags = MAGSequencesDirFmt(self.get_data_path("mags-fake"), "r")

        def mock_run_skani(cmd: List[str]):
            raise RuntimeError("skani failed: insufficient memory")

        mock_run.side_effect = mock_run_skani
        with self.assertRaisesRegex(RuntimeError, "Failed to run Skani comparison"):
            compare_seqs(genomes=mags)

    @patch("q2_skani.skani._run_skani")
    def test_compare_seqs_with_boolean_parameters(self, mock_run):
        """Test compare_seqs with various boolean parameters."""
        mags = MAGSequencesDirFmt(self.get_data_path("mags-fake"), "r")
        captured_cmd = []

        def mock_run_skani(cmd: List[str]):
            captured_cmd.extend(cmd)
            # Create mock output
            output_idx = cmd.index("-o") + 1
            output_file = cmd[output_idx]
            import shutil

            shutil.copy(self.get_data_path("test.matrix"), output_file)

        mock_run.side_effect = mock_run_skani
        compare_seqs(
            genomes=mags,
            ci=True,
            detailed=True,
            diagonal=True,
            median=True,
            robust=True,
        )

        # Verify boolean flags were included
        self.assertIn("--ci", captured_cmd)
        self.assertIn("--detailed", captured_cmd)
        self.assertIn("--diagonal", captured_cmd)
        self.assertIn("--median", captured_cmd)
        self.assertIn("--robust", captured_cmd)

    @patch("q2_skani.skani._run_skani")
    def test_compare_seqs_default_parameters(self, mock_run):
        """Test compare_seqs with default parameters."""
        mags = MAGSequencesDirFmt(self.get_data_path("mags-fake"), "r")
        captured_cmd = []

        def mock_run_skani(cmd: List[str]):
            captured_cmd.extend(cmd)
            # Create mock output
            output_idx = cmd.index("-o") + 1
            output_file = cmd[output_idx]
            import shutil

            shutil.copy(self.get_data_path("test.matrix"), output_file)

        mock_run.side_effect = mock_run_skani
        compare_seqs(genomes=mags)

        # Check default values were used
        self.assertIn("-t", captured_cmd)
        self.assertIn("3", captured_cmd)  # default threads
        self.assertIn("--min-af", captured_cmd)
        self.assertIn("15.0", captured_cmd)  # default min_af
        self.assertIn("-s", captured_cmd)
        self.assertIn("80.0", captured_cmd)  # default screen
        self.assertIn("--full-matrix", captured_cmd)  # default full_matrix=True

    def test_complete_pipeline_integration(self):
        """Test the complete pipeline with three pseudo-genomes without mocking."""

        # similar1: 39b93470-1aef-4962-aef2-13088fd498ac
        # similar2: 00175141-f0a8-4031-8fa2-2a47593dfb01
        # different: 2ad520f7-99d1-4f22-8b97-f5bafd9606d8

        mags = MAGSequencesDirFmt(self.get_data_path("mags"), "r")

        result = compare_seqs(
            genomes=mags,
            threads=1,
            full_matrix=True,
            min_af=5.0,  # Lower threshold for test genomes
            screen=50.0,  # Lower screening threshold
        )

        # Verify basic properties of the result
        self.assertIsInstance(result, skbio.DistanceMatrix)
        self.assertEqual(result.shape, (3, 3))

        # Check that all genomes are in the result
        for genome in [
            "39b93470-1aef-4962-aef2-13088fd498ac",
            "00175141-f0a8-4031-8fa2-2a47593dfb01",
            "2ad520f7-99d1-4f22-8b97-f5bafd9606d8",
        ]:
            self.assertIn(genome, result.ids)

        # Check diagonal values (self-comparison should be 0)
        for genome in result.ids:
            self.assertEqual(result[genome, genome], 0.0)

        # Check that similar genomes have small distance
        similar_distance = result[
            "39b93470-1aef-4962-aef2-13088fd498ac",
            "00175141-f0a8-4031-8fa2-2a47593dfb01",
        ]
        self.assertGreater(similar_distance, 0)  # Not identical
        self.assertLess(similar_distance, 5)  # But very similar (< 5% distance)

        # Check that different genome has large distance from both similar genomes
        diff_to_similar1 = result[
            "39b93470-1aef-4962-aef2-13088fd498ac",
            "2ad520f7-99d1-4f22-8b97-f5bafd9606d8",
        ]
        diff_to_similar2 = result[
            "00175141-f0a8-4031-8fa2-2a47593dfb01",
            "2ad520f7-99d1-4f22-8b97-f5bafd9606d8",
        ]

        # skani returns 100.00 for completely different genomes when using --distance
        self.assertGreater(diff_to_similar1, 90)  # Very different (> 90% distance)
        self.assertGreater(diff_to_similar2, 90)  # Very different (> 90% distance)

        # Check that distance matrix is symmetric
        self.assertEqual(
            result[
                "39b93470-1aef-4962-aef2-13088fd498ac",
                "00175141-f0a8-4031-8fa2-2a47593dfb01",
            ],
            result[
                "00175141-f0a8-4031-8fa2-2a47593dfb01",
                "39b93470-1aef-4962-aef2-13088fd498ac",
            ],
        )
        self.assertEqual(
            result[
                "39b93470-1aef-4962-aef2-13088fd498ac",
                "2ad520f7-99d1-4f22-8b97-f5bafd9606d8",
            ],
            result[
                "2ad520f7-99d1-4f22-8b97-f5bafd9606d8",
                "39b93470-1aef-4962-aef2-13088fd498ac",
            ],
        )
        self.assertEqual(
            result[
                "00175141-f0a8-4031-8fa2-2a47593dfb01",
                "2ad520f7-99d1-4f22-8b97-f5bafd9606d8",
            ],
            result[
                "2ad520f7-99d1-4f22-8b97-f5bafd9606d8",
                "00175141-f0a8-4031-8fa2-2a47593dfb01",
            ],
        )

        # Print the actual distances for manual verification
        print("\nDistance Matrix Results:")
        print(f"Similar1 <-> Similar2: {similar_distance:.2f}%")
        print(f"Similar1 <-> Different: {diff_to_similar1:.2f}%")
        print(f"Similar2 <-> Different: {diff_to_similar2:.2f}%")
