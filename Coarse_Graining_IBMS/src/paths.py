"""Central path constants for Coarse_Graining_IBMS.

All modules should import paths from here rather than constructing them locally,
so that output locations are consistent regardless of the working directory from
which scripts are invoked.
"""

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Coarse_Graining_IBMS/

IBM_OUTPUT_DIR = _PROJECT_ROOT / "data" / "ibm_output"
IMAGES_DIR = _PROJECT_ROOT / "data" / "images"
SURROGATE_MODELS_DIR = _PROJECT_ROOT / "data" / "models" / "surrogates"
OPTIMISATION_MODELS_DIR = _PROJECT_ROOT / "data" / "models" / "optimisation"

#: Stage-count output from Ziqian's R IBM (``S*/csv/<RUN_ID>/stage_df.csv``),
#: used by the adult-vs-larval density-relationship conference figure.  This is
#: committed *input* data, not a generated output, so it is not auto-created.
ZIQIAN_RUNS_DIR = _PROJECT_ROOT / "data" / "ziqian_ibm_runs"

#: Raw per-generation stage counts from the single-location release experiments
#: (``simulate_single_location_release.py``).  Its own subdirectory because one
#: run writes one CSV and they would otherwise crowd out the training data.
SINGLE_LOCATION_RELEASE_DIR = IBM_OUTPUT_DIR / "single_location_releases"

#: Figures from those experiments, kept together for the same reason.
SINGLE_LOCATION_RELEASE_IMAGES_DIR = IMAGES_DIR / "single_location_releases"

for _d in [IBM_OUTPUT_DIR, IMAGES_DIR, SURROGATE_MODELS_DIR, OPTIMISATION_MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


def input_output_pairs_path(
    seed: int | None,
    with_genetics: bool = False,
    test: bool = False,
    param_tag: str | None = None,
    homozygotes_only: bool = False,
) -> Path:
    """Path to an IBM input-output pair CSV for a given data seed.

    A single naming scheme is shared by the data generator and the dataloaders
    so they never disagree on where a file lives.

    Parameters
    ----------
    seed : int | None
        Seed used when the data was generated; encoded in the filename.
    with_genetics : bool
        Select the genetics-aware (per-genotype) file rather than the
        aggregated-counts file.
    test : bool
        Select the held-out test set, which carries a ``test=True`` marker in
        its filename so it is never confused with the training data.
    param_tag : str | None
        Marker identifying a non-default gene-drive parameter set (e.g. a
        Pareto-frontier point, tagged ``low0p50_high0p70_20260612_084001_row0``).
        When ``None`` — the default-parameter case — the filename is exactly
        what it was before custom parameter sets existed.
    homozygotes_only : bool
        Select the data whose input adults were sampled on the WW/GG face only,
        with no heterozygotes (``generate_training_data.py --homozygotes-only``).
        Such data describes a different region of the input space from the
        full three-genotype data, so it carries its own marker and a surrogate
        trained on one can never be matched to the other. ``False`` — the
        default — is the filename used before the heterozygote-free mode
        existed.
    """
    genetics = "_with_genetics" if with_genetics else ""
    test_marker = "_test=True" if test else ""
    params = f"_params={param_tag}" if param_tag else ""
    homozygotes = "_homozygotes=True" if homozygotes_only else ""
    return (
        IBM_OUTPUT_DIR
        / f"input_output_pairs{genetics}_seed={seed}{params}{homozygotes}{test_marker}.csv"
    )


def single_location_release_path(
    seed: int | None,
    param_tag: str | None = None,
) -> Path:
    """Path to the raw stage counts from a single-location release experiment.

    ``simulate_single_location_release.py`` runs the IBM for one location over
    many generations, which costs roughly a quarter of a minute per generation
    per replicate.  Re-running that to adjust a figure is not worth anyone's
    time, so every run writes its raw per-generation counts here and the script
    can replot from the file instead (``--from-csv``).

    Parameters
    ----------
    seed : int | None
        Master seed of the run, encoded in the filename so a figure and the
        data behind it can always be matched up.
    param_tag : str | None
        Marker identifying the gene-drive parameter set, as returned by
        ``Gene_Drive_Optimisation.gene_drive_objective.load_gene_drive_parameters``
        (e.g. ``low0p30_high0p50_20260601_101508_row0``).  ``None`` marks the
        defaults from ``initialize_simulation()``.
    """
    params = f"_params={param_tag}" if param_tag else "_params=default"
    SINGLE_LOCATION_RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    return (
        SINGLE_LOCATION_RELEASE_DIR / f"single_location_release{params}_seed={seed}.csv"
    )
