from dataclasses import replace
from pathlib import Path

import numpy as np

from bn_pah_fes.config import Parameters
from bn_pah_fes.data import load_data
from bn_pah_fes.fitting import fit_free_energy


FIT_PADDING_FACTORS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
RESULTS_DIR = Path("results")


params = Parameters(model="harmonic")
data = load_data(Path("data"), params.n_samples, params.n_acf)

for fit_padding_factor in FIT_PADDING_FACTORS:
    params_run = replace(
        params,
        fit_padding_factor=fit_padding_factor,
    )

    print(f"\nfit_padding_factor = {fit_padding_factor:g}")

    fit_result = fit_free_energy(data, params_run)

    output_data = np.column_stack(
        [data.idx, data.q[:, 0], data.q[:, 1], data.q[:, 2], fit_result.delta_g_pbe]
    )
    np.savetxt(
        RESULTS_DIR
        / f"DeltaG_PBE_fit_padding_factor_{fit_padding_factor:g}.txt",
        output_data,
        fmt=["%d", "%.10e", "%.10e", "%.10e", "%.10e"],
        header="index q0 qS qT DeltaG_PBE_Hartree",
    )
