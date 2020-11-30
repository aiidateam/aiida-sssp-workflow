# -*- coding: utf-8 -*-
"""
WorkChain calculate the bands for certain pseudopotential
"""
from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, calcfunction, ToContext
from aiida.plugins import WorkflowFactory, CalculationFactory

PwBandsWorkChain = WorkflowFactory('quantumespresso.pw.bands')
create_kpoints_from_distance = CalculationFactory(
    'quantumespresso.create_kpoints_from_distance')


@calcfunction
def helper_parse_upf(upf):
    return orm.Str(upf.element)


PW_PARAS = lambda: orm.Dict(
    dict={
        'SYSTEM': {
            'ecutrho': 800,
            'ecutwfc': 200,
        },
        'ELECTRONS': {
            'conv_thr': 1e-10,
        },
    })


class BandsWorkChain(WorkChain):
    """WorkChain calculate the bands for certain pseudopotential"""
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('code',
                   valid_type=orm.Code,
                   help='The `pw.x` code use for the `PwCalculation`.')
        spec.input('pseudo',
                   valid_type=orm.UpfData,
                   required=True,
                   help='Pseudopotential to be verified')
        spec.input(
            'structure',
            valid_type=orm.StructureData,
            required=True,
            help='Ground state structure which the verification perform')
        spec.input('options',
                   valid_type=orm.Dict,
                   required=False,
                   help='Optional `options` to use for the `PwCalculations`.')
        spec.input_namespace('parameters', help='Para')
        spec.input('parameters.pw',
                   valid_type=orm.Dict,
                   default=PW_PARAS,
                   help='parameters for pw.x.')
        spec.input('parameters.scf_kpoints_distance',
                   valid_type=orm.Float,
                   default=lambda: orm.Float(0.1),
                   help='Kpoints distance setting for scf calculation.')
        spec.input('parameters.bands_kpoints_distance',
                   valid_type=orm.Float,
                   default=lambda: orm.Float(0.15),
                   help='Kpoints distance setting for bands nscf calculation.')
        spec.outline(
            cls.setup,
            cls.validate_structure,
            cls.run_bands,
            cls.inspect_bands,
            cls.results,
        )
        spec.output('scf_parameters',
                    valid_type=orm.Dict,
                    help='The output parameters of the SCF `PwBaseWorkChain`.')
        spec.output(
            'band_parameters',
            valid_type=orm.Dict,
            help='The output parameters of the BANDS `PwBaseWorkChain`.')
        spec.output('band_structure',
                    valid_type=orm.BandsData,
                    help='The computed band structure.')
        spec.exit_code(201,
                       'ERROR_SUB_PROCESS_FAILED_BANDS',
                       message='The `PwBandsWorkChain` sub process failed.')

    def setup(self):
        """Input validation"""
        # TODO set ecutwfc and ecutrho according to certain protocol
        import collections.abc

        def update(d, u):
            # pylint: disable=invalid-name
            for k, v in u.items():
                if isinstance(v, collections.abc.Mapping):
                    d[k] = update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        scf_parameters = {
            'SYSTEM': {
                'degauss': 0.02,
                'occupations': 'smearing',
                'smearing': 'marzari-vanderbilt',
            },
            'ELECTRONS': {
                'conv_thr': 1e-10,
            },
        }
        bands_parameters = {
            'SYSTEM': {
                # ############## NOT REALLY TAKE EFFECT ##########
                # 'degauss': 0.02,
                # 'occupations': 'smearing',
                # 'smearing': 'marzari-vanderbilt',
                # #############
                "noinv": True,
                "nosym": True,
            },
            'ELECTRONS': {
                'conv_thr': 1e-10,
            },
        }
        pw_scf_parameters = update(scf_parameters,
                                   self.inputs.parameters.pw.get_dict())
        pw_bands_parameters = update(bands_parameters,
                                     self.inputs.parameters.pw.get_dict())
        self.ctx.pw_scf_parameters = orm.Dict(dict=pw_scf_parameters)
        self.ctx.pw_bands_parameters = orm.Dict(dict=pw_bands_parameters)

        self.ctx.scf_kpoints_distance = self.inputs.parameters.scf_kpoints_distance

        bands_kpoints_distance = self.inputs.parameters.bands_kpoints_distance
        self.ctx.bands_kpoints = create_kpoints_from_distance(
            self.inputs.structure, bands_kpoints_distance, orm.Bool(False))

    def validate_structure(self):
        """Create isolate atom and validate structure"""
        self.ctx.element = helper_parse_upf(self.inputs.pseudo)
        self.ctx.pseudos = {self.ctx.element.value: self.inputs.pseudo}

    def run_bands(self):
        if 'options' in self.inputs:
            options = self.inputs.options.get_dict()
        else:
            from aiida_quantumespresso.utils.resources import get_default_options

            # Too many kpoints may go beyond 1800s
            options = get_default_options(with_mpi=True,
                                          max_wallclock_seconds=1800 * 3)

        inputs = AttributeDict({
            'structure': self.inputs.structure,
            'scf': {
                'pw': {
                    'code': self.inputs.code,
                    'pseudos': self.ctx.pseudos,
                    'parameters': self.ctx.pw_scf_parameters,
                    'settings': orm.Dict(dict={'CMDLINE': ['-ndiag', '1']}),
                    'metadata': {
                        'options': options
                    },
                },
                'kpoints_distance': self.ctx.scf_kpoints_distance,
            },
            'bands': {
                'pw': {
                    'code': self.inputs.code,
                    'pseudos': self.ctx.pseudos,
                    'parameters': self.ctx.pw_bands_parameters,
                    'settings': orm.Dict(dict={'CMDLINE': ['-nk', '4']}),
                    'metadata': {
                        'options': options
                    },
                },
            },
            'nbands_factor': orm.Float(2.0),
            'bands_kpoints': self.ctx.bands_kpoints,
        })
        running = self.submit(PwBandsWorkChain, **inputs)
        self.report(f'Running pw bands calculation pk={running.pk}')
        return ToContext(workchain_bands=running)

    def inspect_bands(self):
        """inspect the result of bands calculation."""
        workchain = self.ctx.workchain_bands

        if not workchain.is_finished_ok:
            self.report(
                f'PwBandsWorkChain for bands evaluation failed with exit status {workchain.exit_status}'
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_BANDS

        self.report('pw bands workchain succesfully completed')
        self.out('scf_parameters', workchain.outputs.scf_parameters)
        self.out('band_parameters', workchain.outputs.band_parameters)
        self.out('band_structure', workchain.outputs.band_structure)

    def results(self):
        pass
