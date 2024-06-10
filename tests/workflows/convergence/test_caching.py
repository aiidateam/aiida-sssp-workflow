import pytest

from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import ProcessBuilder, run_get_node
from aiida.manage.caching import enable_caching


UpfData = DataFactory("pseudo.upf")


@pytest.mark.slow
@pytest.mark.usefixtures("aiida_profile_clean")
def test_caching_bands(
    pseudo_path,
    code_generator,
):
    """Test caching is working for the first pw calculation of bands.
    Test if the remote_path was empty bands and phonon_frequencies workflow will manage
    to rerun the first preparing pw.x calculation."""
    _ConvergenceBandsWorkChain = WorkflowFactory("sssp_workflow.convergence.bands")

    # The caching should turned on also for the first prepareing run
    # Otherwise, the scf calculation inside band workflow has two duplicate nodes which has same uuid
    # but are both valid cached source. This cause caching race condition.
    with enable_caching(identifier="aiida.calculations:quantumespresso.*"):
        bands_builder: ProcessBuilder = _ConvergenceBandsWorkChain.get_builder(
            pseudo=pseudo_path("Al"),
            protocol="test",
            cutoff_list=[(20, 80), (30, 120)],
            configuration="DC",
            code=code_generator("pw"),
            clean_workdir=True,
        )

        # Running a bands convergence workflow first and check that SCF is not from cached calcjob
        _, source_node = run_get_node(bands_builder)

        # check the first scf of reference
        # The pw calculation
        source_ref_wf = [
            p
            for p in source_node.called
            if p.base.extras.get("wavefunction_cutoff", None) == 30
        ][0]
        source_scf_calcjob_node = source_ref_wf.called[1].called[0].called[1]
        assert (
            source_scf_calcjob_node.base.extras.get("_aiida_cached_from", None) is None
        )

        # check the first bands of reference was cached
        # The pw calculation
        source_band_calcjob_node = source_ref_wf.called[1].called[1].called[0]
        assert (
            source_band_calcjob_node.base.extras.get("_aiida_cached_from", None) is None
        )

        # Run again and check it is using caching
        _, cached_node = run_get_node(bands_builder)
        cached_ref_wf = [
            p
            for p in cached_node.called
            if p.base.extras.get("wavefunction_cutoff", None) == 30
        ][0]
        cached_scf_calcjob_node = cached_ref_wf.called[1].called[0].called[1]

        assert (
            cached_scf_calcjob_node.base.extras.get("_aiida_cached_from", None)
            == source_scf_calcjob_node.uuid
        )
        assert not cached_scf_calcjob_node.base.caching.is_valid_cache

        cached_band_calcjob_node = cached_ref_wf.called[1].called[1].called[0]

        assert (
            cached_band_calcjob_node.base.extras.get("_aiida_cached_from", None)
            == source_band_calcjob_node.uuid
        )


@pytest.mark.slow
@pytest.mark.usefixtures("aiida_profile_clean")
def test_caching_phonon_frequencies(
    pseudo_path,
    code_generator,
):
    """Test caching is working for the first pw calculation of phonon_frequencies.
    Test if the remote_path was empty phonon_frequencies workflow will manage
    to rerun the first preparing pw.x calculation."""
    _ConvergencePhononFrequenciessWorkChain = WorkflowFactory(
        "sssp_workflow.convergence.phonon_frequencies"
    )

    # The caching should turned on also for the first prepareing run
    # Otherwise, the scf calculation inside band workflow has two duplicate nodes which has same uuid
    # but are both valid cached source. This cause caching race condition.
    with enable_caching(identifier="aiida.calculations:quantumespresso.*"):
        phonon_frequencies_builder: ProcessBuilder = (
            _ConvergencePhononFrequenciessWorkChain.get_builder(
                pseudo=pseudo_path("Al"),
                protocol="test",
                cutoff_list=[(20, 80), (30, 120)],
                configuration="DC",
                pw_code=code_generator("pw"),
                ph_code=code_generator("ph"),
                clean_workdir=True,
            )
        )

        # Running a phonon frequencies convergence workflow first and check that SCF is not from cached calcjob
        _, source_node = run_get_node(phonon_frequencies_builder)

        # check the first scf of reference
        # The pw calculation
        source_ref_wf = [
            p
            for p in source_node.called
            if p.base.extras.get("wavefunction_cutoff", None) == 30
        ][0]
        source_scf_calcjob_node = source_ref_wf.called[0].called[1]
        assert (
            source_scf_calcjob_node.base.extras.get("_aiida_cached_from", None) is None
        )

        # The ph calculation
        source_ph_calcjob_node = source_ref_wf.called[1].called[0]
        assert (
            source_ph_calcjob_node.base.extras.get("_aiida_cached_from", None) is None
        )

        # Run again and check it is using caching
        _, cached_node = run_get_node(phonon_frequencies_builder)
        cached_ref_wf = [
            p
            for p in cached_node.called
            if p.base.extras.get("wavefunction_cutoff", None) == 30
        ][0]
        cached_scf_calcjob_node = cached_ref_wf.called[0].called[1]

        assert (
            cached_scf_calcjob_node.base.extras.get("_aiida_cached_from", None)
            == source_scf_calcjob_node.uuid
        )
        assert not cached_scf_calcjob_node.base.caching.is_valid_cache

        # Run again and check it is using caching
        cached_ph_calcjob_node = cached_ref_wf.called[1].called[0]
        assert (
            cached_ph_calcjob_node.base.extras.get("_aiida_cached_from", None)
            == source_ph_calcjob_node.uuid
        )


@pytest.mark.slow
@pytest.mark.usefixtures("aiida_profile_clean")
def test_caching_bands_rerun_pw_prepare(
    pseudo_path,
    code_generator,
):
    """Test caching is working for the first pw calculation of bands.
    After the first run, I manually make the bands calculation invalid cache so it can rerun with an empty remote
    The test check that the scf will rerun if the remote not exist."""
    _ConvergenceBandsWorkChain = WorkflowFactory("sssp_workflow.convergence.bands")

    # The caching should turned on also for the first prepareing run
    # Otherwise, the scf calculation inside band workflow has two duplicate nodes which has same uuid
    # but are both valid cached source. This cause caching race condition.
    with enable_caching(identifier="aiida.calculations:quantumespresso.*"):
        bands_builder: ProcessBuilder = _ConvergenceBandsWorkChain.get_builder(
            pseudo=pseudo_path("Al"),
            protocol="test",
            cutoff_list=[(20, 80), (30, 120)],
            configuration="DC",
            code=code_generator("pw"),
            clean_workdir=True,
        )

        # Running a bands convergence workflow first and check that SCF is not from cached calcjob
        _, source_node = run_get_node(bands_builder)

        # Make the source band calculation invalid cache
        source_ref_wf = [
            p
            for p in source_node.called
            if p.base.extras.get("wavefunction_cutoff", None) == 30
        ][0]
        source_band_calcjob_node = source_ref_wf.called[1].called[1].called[0]
        assert source_band_calcjob_node.is_valid_cache

        source_band_calcjob_node.is_valid_cache = False

        # Run again and check it is using caching
        _, cached_node = run_get_node(bands_builder)

        # Check the band work chain finished okay
        cached_ref_wf = [
            p
            for p in cached_node.called
            if p.base.extras.get("wavefunction_cutoff", None) == 30
        ][0]
        assert cached_ref_wf.called[1].is_finished_ok

        cached_band_calcjob_node = cached_ref_wf.called[1].called[1].called[0]
        assert (
            cached_band_calcjob_node.base.extras.get("_aiida_cached_from", None) is None
        )
        assert cached_band_calcjob_node.exit_code.status == 305


@pytest.mark.slow
@pytest.mark.usefixtures("aiida_profile_clean")
def test_caching_phonon_frequencies_rerun_pw_prepare(
    pseudo_path,
    code_generator,
):
    """Test caching is working for the first pw calculation of phonon_frequencies.
    Test if the remote_path was empty phonon_frequencies workflow will manage
    to rerun the first preparing pw.x calculation."""
    _ConvergencePhononFrequenciessWorkChain = WorkflowFactory(
        "sssp_workflow.convergence.phonon_frequencies"
    )

    # The caching should turned on also for the first prepareing run
    # Otherwise, the scf calculation inside band workflow has two duplicate nodes which has same uuid
    # but are both valid cached source. This cause caching race condition.
    with enable_caching(identifier="aiida.calculations:quantumespresso.*"):
        phonon_frequencies_builder: ProcessBuilder = (
            _ConvergencePhononFrequenciessWorkChain.get_builder(
                pseudo=pseudo_path("Al"),
                protocol="test",
                cutoff_list=[(20, 80), (30, 120)],
                configuration="DC",
                pw_code=code_generator("pw"),
                ph_code=code_generator("ph"),
                clean_workdir=True,
            )
        )

        # Running a phonon_frequencies convergence workflow first and check that SCF is not from cached calcjob
        _, source_node = run_get_node(phonon_frequencies_builder)

        # Make the source ph calculation invalid cache
        source_ref_wf = [
            p
            for p in source_node.called
            if p.base.extras.get("wavefunction_cutoff", None) == 30
        ][0]
        source_ph_calcjob_node = source_ref_wf.called[1].called[0]
        assert source_ph_calcjob_node.is_valid_cache

        source_ph_calcjob_node.is_valid_cache = False

        # Run again and check it is using caching
        _, cached_node = run_get_node(phonon_frequencies_builder)

        cached_ref_wf = [
            p
            for p in cached_node.called
            if p.base.extras.get("wavefunction_cutoff", None) == 30
        ][0]
        # Check the ph from rerun pw is finished okay
        assert cached_ref_wf.is_finished_ok

        cached_ph_calcjob_node = cached_ref_wf.called[1].called[0]
        assert (
            cached_ph_calcjob_node.base.extras.get("_aiida_cached_from", None) is None
        )
        assert cached_ph_calcjob_node.exit_code.status == 312
