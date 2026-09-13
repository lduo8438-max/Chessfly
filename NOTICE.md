# Data and inspiration notice

Chessfly is designed to use the MaleCNS v1.0 connectome, made available under
CC BY 4.0 by the FlyEM Project at HHMI Janelia in collaboration with the
University of Cambridge, MRC Laboratory of Molecular Biology, and Google
Research. Source: <https://male-cns.janelia.org/>.

The project concept and its emphasis on explicit engineered interfaces were
inspired by Alex Wormuth's Stonkfly project:
<https://github.com/nftechie/stonkfly>. Chessfly is an independent implementation
and does not copy Stonkfly media or imply endorsement.

## Fly body model

The FlyJack-style renders use the NeuroMechFly body model as distributed in
FlyGym 1.1.0 under the Apache License 2.0: meshes, kinematic tree, the tripod
standing pose and part colours. NeuroMechFly: Lobato-Rios et al., *Nature
Methods* (2022); FlyGym / NeuroMechFly v2: Wang-Chen et al., *Nature Methods*
(2024). Source: <https://github.com/NeLy-EPFL/flygym>. The assets are not
redistributed with Chessfly; `chessfly prepare-fly-model` downloads the pinned
wheel, verifies its SHA-256, and copies FlyGym's LICENSE next to the extracted
files.

## Render style

The table-under-one-lamp staging, the spiking brain hologram and the palette of
the FlyJack-style video follow FlyJack
(<https://fanpu.io/games/flyjack/>). Chessfly re-implements that look in its own
Blender and compositing code and does not copy FlyJack media or code.
