# Local changes to `tusen-ai/SimpleTrack`

Upstream checkout: `third_party/SimpleTrack`, commit
`05c96bb7ed98fc179856f327544612a66c839b5e`.

The tracking algorithm and `configs/nu_configs/giou.yaml` were not changed.
The local patch is intentionally limited to the following compatibility and
debugging changes:

1. `preprocessing/nuscenes_data/detection.py` reads `sample['velocity']` only
   when upstream's `--velo` flag is active. This makes the documented no-velocity
   preprocessing path actually work and never synthesizes velocity.
2. The six official nuScenes metadata preprocessors accept `--scene`. Without
   this option their behavior is unchanged; with it they process one validation
   scene for the requested smoke test.
3. `mot_3d/preprocessing/bbox_coarse_hash.py` uses builtin `int` instead of the
   removed NumPy alias `np.int`. This is a type-only compatibility fix.

All orchestration, validation, evaluation export, and comparison logic lives
outside the upstream checkout.
