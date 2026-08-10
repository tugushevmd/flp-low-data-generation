# Large artifacts

Model weights and large external raw runs are not tracked by Git. Put downloaded files in `artifacts/raw/` when a full recalculation is needed.

## Controlled-model weights

| file | size | SHA-256 |
|---|---:|---|
| `controlled_prior_pretraining_weights.zip` | 67.85 MB | `2228356a0ba1dddf18d37d6c8afd008fa0c13c74156947c94e2e86639131def6` |
| `controlled_prior_learning_curves_weights.zip` | 203.36 MB | `ba2ec3ce40b8136ce196fb7e6511585e4671e2743572a9d0935961312d99b6f3` |
| `controlled_prior_confirmatory_weights.zip` | 101.73 MB | `39b36934485ce3c6815b94714c3d10ebf339aa45b28f8d5d53eea35703b88256` |
| `controlled_prior_fixed_checkpoint_weights.zip` | 67.81 MB | `61d7e2c82e1245700307be2d142e1476af8a6d822dabcc3d1efe456af31e89a4` |

## External raw runs

| file | size | SHA-256 |
|---|---:|---|
| `external_gpmolformer_25_results.zip` | 0.16 MB | `b068f0aab7951e8c72b7cc606aeb944826b1b42dc84ee52a2ef4fc179eb199d2` |
| `external_molgpt_results.zip` | 0.26 MB | `8c6703b1a50992462a53b94cf49da63e01bb17a98e39113de7f25e1377117e09` |
| `external_reinvent_results.zip` | 37.37 MB | `94127ef43b3da4365750ddbb82f67910a751f839cede9fa8b2dad73e0aee2f09` |

The small external archives may later be added directly. The REINVENT archive is better distributed as a GitHub Release asset together with the model weights.
