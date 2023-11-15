# SSSP workflow

[![Documentation Status](https://readthedocs.org/projects/aiida-sssp-workflow/badge/?version=latest)](https://aiida-sssp-workflow.readthedocs.io/en/latest/?badge=latest)

## For maintainers

To create a new release, clone the repository, install development dependencies with `pip install '.[dev]'`, and then execute `bumpver update`.
This will:

  1. Create a tagged release with bumped version and push it to the repository.
  2. Trigger a GitHub actions workflow that creates a GitHub release.

Additional notes:

  - Use the `--dry` option to preview the release change.
  - The release tag (e.g. a/b/rc) is determined from the last release.
    Use the `--tag` option to switch the release tag.

### Logger level

The aiida-core logger level is recommened to set to `REPORT` as default.
In workflow, process related messages are pop to daemon logger in REPORT level.
If process finished with none-zero exit_code, log a waring message.
While for debug purpose, the INFO level is for showing the parameters infomations when processes are launched.

### Issues of version

- v3.0.1:
    - nowf not turned on for saving inode purpose, fixed in next version
    - the dual value of NC/SL is 8 for precision measure and bands for norm-conserving pseudopotentials, better to be 4. fixed in next version
    - the dual value for high dual elements is 8 for non-NC pseudopotentials, better to be 16. fixed in next version

## License

MIT


## Contact

📧 email: jusong.yu@psi.ch
