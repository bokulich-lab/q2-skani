# ----------------------------------------------------------------------------
# Copyright (c) 2025, Bokulich Lab.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import glob
from pathlib import Path

import pandas as pd
import subprocess
import tempfile
import os
from typing import List, Dict, Any, Literal, Optional, Union

import skbio
from q2_types.feature_data_mag import MAGSequencesDirFmt
from q2_types.per_sample_sequences import MultiMAGSequencesDirFmt


def _construct_triangle_cmd(
    fasta_list: str,
    output_file: str,
    skani_args: Dict[str, Any],
) -> List[str]:
    """Construct the Skani command with all parameters.

    Parameters
    ----------
    fasta_list : str
        Path to the file containing genome paths
    output_file : str
        Path to the output file
    skani_args : Dict[str, Any]
        Dictionary containing all Skani parameters

    Returns
    -------
    List[str]
        The constructed Skani command as a list of strings
    """
    cmd = ["skani", "triangle", "-v", "--distance"]

    # Add required arguments
    cmd.extend(["-l", fasta_list, "-o", output_file])

    # Define parameter mappings
    param_with_values = {
        "threads": "-t",
        "min_af": "--min-af",
        "compression": "-c",
        "marker_c": "-m",
        "screen": "-s",
    }

    boolean_flags = {
        "ci": "--ci",
        "detailed": "--detailed",
        "diagonal": "--diagonal",
        "sparse": "--sparse",
        "full_matrix": "--full-matrix",
        "median": "--median",
        "no_learned_ani": "--no-learned-ani",
        "robust": "--robust",
        "faster_small": "--faster-small",
    }

    # Add parameters with values
    for param, flag in param_with_values.items():
        if param in skani_args:
            cmd.extend([flag, str(skani_args[param])])

    # Add boolean flags
    for param, flag in boolean_flags.items():
        if param in skani_args and skani_args[param]:
            cmd.append(flag)

    # Add preset if specified (handled differently because the flag is the value itself)
    if "preset" in skani_args and skani_args["preset"]:
        cmd.append(f"--{skani_args['preset']}")

    return cmd


def _run_skani(cmd: List[str]) -> None:
    """Run Skani with proper error handling.

    Parameters
    ----------
    cmd : List[str]
        The command to run as a list of strings

    Raises
    ------
    RuntimeError
        If Skani fails to run or returns a non-zero exit code
    """
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        error_msg = (
            f"Skani failed with exit code {e.returncode}.\n"
            f"Command: {' '.join(cmd)}\n"
        )
        if e.stdout:
            error_msg += f"stdout:\n{e.stdout}\n"
        if e.stderr:
            error_msg += f"stderr:\n{e.stderr}"
        raise RuntimeError(error_msg)


def _process_skani_matrix(matrix_file: str) -> pd.DataFrame:
    """Process the Skani matrix file into a DistanceMatrix.

    Parameters
    ----------
    matrix_file : str
        Path to the Skani matrix output file

    Returns
    -------
    pd.DataFrame
        The processed distance matrix
    """
    # Read the matrix file
    df = pd.read_csv(matrix_file, sep="\t", index_col=0, skiprows=1, header=None)
    df.index = df.index.map(lambda x: Path(x).stem)
    df.columns = df.index.tolist()
    df.index.name = "id"

    # Convert ANI to distance if needed (100 - ANI)
    # if not df.values[0, 0] > 1:  # If values are ANI (0-100)
    #     df = 100 - df

    return df


def compare_seqs(
    genomes: Union[MAGSequencesDirFmt | MultiMAGSequencesDirFmt],
    threads: int = 3,
    min_af: float = 15.0,
    compression: int = 125,
    marker_c: int = 1000,
    screen: float = 80.0,
    ci: bool = False,
    detailed: bool = False,
    diagonal: bool = False,
    sparse: bool = False,
    full_matrix: bool = True,
    median: bool = False,
    no_learned_ani: bool = False,
    robust: bool = False,
    faster_small: bool = False,
    preset: Optional[Literal["fast", "medium", "slow", "small-genomes"]] = None,
) -> skbio.DistanceMatrix:
    """Compare sequences using Skani.

    Parameters
    ----------
    genomes : MAGSequencesDirFmt | MultiMAGSequencesDirFmt
        The genomes to compare
    threads : int, optional
        Number of threads to use, by default 3
    min_af : float, optional
        Minimum aligned fraction to report, by default 15.0
    compression : int, optional
        Compression factor (k-mer subsampling rate), by default 125
    marker_c : int, optional
        Marker k-mer compression factor, by default 1000
    screen : float, optional
        Screen out pairs with approximately less than this % identity, by default 80.0
    ci : bool, optional
        Output confidence intervals, by default False
    detailed : bool, optional
        Print additional info, by default False
    diagonal : bool, optional
        Output self-self comparisons, by default False
    sparse : bool, optional
        Output sparse matrix format, by default False
    full_matrix : bool, optional
        Output full matrix instead of lower-triangular, by default True
    median : bool, optional
        Estimate median identity instead of mean, by default False
    no_learned_ani : bool, optional
        Disable regression model for ANI prediction, by default False
    robust : bool, optional
        Estimate mean after trimming off 10%/90% quantiles, by default False
    faster_small : bool, optional
        Filter small genomes more aggressively, by default False
    preset : Literal["fast", "medium", "slow", "small-genomes"] | None, optional
        Preset mode to use, by default None

    Returns
    -------
    skbio.DistanceMatrix
        The distance matrix
    """
    # Filter locals() to only include function parameters, excluding 'genomes'
    skani_args = {
        k: v
        for k, v in locals().items()
        if k
        not in {
            "genomes",
        }
    }

    # Create a temporary directory for Skani input/output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a list file containing paths to all genomes
        list_file = os.path.join(temp_dir, "genome_list.txt")
        with open(list_file, "w") as f:
            f.writelines(
                f"{str(p)}\n"
                for p in glob.glob(str(genomes.path / "**" / "*.fasta"), recursive=True)
            )

        # Run Skani in triangle mode
        output_file = os.path.join(temp_dir, "skani_output.tsv")

        # Construct and run the Skani command
        cmd = _construct_triangle_cmd(
            fasta_list=list_file,
            output_file=output_file,
            skani_args=skani_args,
        )

        try:
            _run_skani(cmd)
        except Exception as e:
            raise RuntimeError(
                f"Failed to run Skani comparison: {str(e)}\n"
                f"Command: {' '.join(cmd)}"
            )

        # Process the matrix file into a DistanceMatrix
        df = _process_skani_matrix(output_file)
        return skbio.DistanceMatrix(df.values, ids=df.index.tolist())
