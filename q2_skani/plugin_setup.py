# ----------------------------------------------------------------------------
# Copyright (c) 2025, Bokulich Lab.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
from q2_types.per_sample_sequences import MAGs
from q2_types.sample_data import SampleData

import q2_skani
from q2_types.feature_data import FeatureData
from q2_types.feature_data_mag import MAG
from qiime2.core.type import Int, Float, Range, Bool, Str, Choices
from qiime2.plugin import Citations, Plugin
from q2_types.distance_matrix import DistanceMatrix

from q2_skani.skani import compare_seqs

citations = Citations.load("citations.bib", package="q2_skani")

plugin = Plugin(
    name="skani",
    version=q2_skani.__version__,
    website="https://github.com/bokulich-lab/q2-skani",
    package="q2_skani",
    description=(
        "This QIIME 2 plugin wraps skani for computing "
        "Average Nucleotide Identity (ANI) between genomes."
    ),
    short_description="Plugin for computing ANI between genomes.",
)

plugin.methods.register_function(
    function=compare_seqs,
    inputs={"genomes": FeatureData[MAG] | SampleData[MAGs]},
    parameters={
        "threads": Int % Range(1, None),
        "min_af": Float % Range(0.0, 100.0, inclusive_end=True),
        "compression": Int % Range(1, None),
        "marker_c": Int % Range(1, None),
        "screen": Float % Range(0.0, 100.0, inclusive_end=True),
        "ci": Bool,
        "detailed": Bool,
        "diagonal": Bool,
        "sparse": Bool,
        "full_matrix": Bool,
        "median": Bool,
        "no_learned_ani": Bool,
        "robust": Bool,
        "faster_small": Bool,
        "preset": Str % Choices(["fast", "medium", "slow", "small-genomes"]),
    },
    outputs=[("distance_matrix", DistanceMatrix)],
    input_descriptions={"genomes": "The genomes to compare."},
    parameter_descriptions={
        "threads": "Number of threads to use (default: 3).",
        "min_af": "Minimum aligned fraction to report (default: 15.0).",
        "compression": "Compression factor (k-mer subsampling rate) (default: 125).",
        "marker_c": "Marker k-mer compression factor (default: 1000).",
        "screen": (
            "Screen out pairs with approximately less than this "
            "% identity (default: 80.0)."
        ),
        "ci": "Output confidence intervals (default: False).",
        "detailed": "Print additional info (default: False).",
        "diagonal": "Output self-self comparisons (default: False).",
        "sparse": "Output sparse matrix format (default: False).",
        "full_matrix": (
            "Output full matrix instead of lower-triangular (default: True)."
        ),
        "median": "Estimate median identity instead of mean (default: False).",
        "no_learned_ani": (
            "Disable regression model for ANI prediction (default: False)."
        ),
        "robust": (
            "Estimate mean after trimming off 10%/90% quantiles (default: False)."
        ),
        "faster_small": "Filter small genomes more aggressively (default: False).",
        "preset": (
            "Preset mode to use: 'fast' (2x faster, less accurate), "
            "'medium' (2x slower, more accurate), 'slow' (4x slower, most accurate), "
            "or 'small-genomes' (for genomes < 20 kb)."
        ),
    },
    output_descriptions={"distance_matrix": "The distance matrix."},
    name="Compare sequences using skani triangle mode",
    description=(
        "Compare sequences using skani to compute Average Nucleotide "
        "Identity (ANI). This method performs a many-to-many comparison "
        "of genomes using the triangle method, which is more efficient for "
        "all-vs-all comparisons. skani is generally faster and more robust "
        "than FastANI, especially for metagenome-assembled genomes (MAGs)."
    ),
    citations=[],
)
