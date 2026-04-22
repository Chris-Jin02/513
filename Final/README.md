# CS 513 Final Project Framework

This folder is now organized around one shared foundation plus three parallel model versions.

Shared foundation:
- [Proposal](./Proposal/proposal.md)
- [Data](./Data/README.md)

Parallel model versions:
- [Version_A](./Version_A/README.md)
- [Version_B](./Version_B/README.md)
- [Version_C](./Version_C/README.md)

Recommended working model:
1. Keep `Data` as the only shared technical baseline.
2. Let each team member build one full recommendation version using the same processed data.
3. Let each version own its own EDA, experiments, and results without redoing data cleaning.
4. Compare the three versions under the same temporal evaluation setup.
5. Use the comparison to decide the best final model, or build a hybrid if that is stronger.

Folder roles:
- `Data/`: canonical shared data pipeline and cleaned artifacts
- `Version_A/`: one complete modeling track
- `Version_B/`: one complete modeling track
- `Version_C/`: one complete modeling track
- `Proposal/`: proposal and planning notes

Each version folder now follows the same internal structure:
- `EDA/`
- `Experiments/`
- `Results/`
- `Notes/`

Recommended assignment:
- `Version_A`: your version
- `Version_B`: teammate version
- `Version_C`: teammate version
