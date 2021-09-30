# -*- coding: utf-8 -*-
"""
Convergence workchain based on aiida-optimize by
inherit `aiida_optimize.engines._convergence::_ConvergenceImpl`
"""
import typing as ty
import itertools

import numpy as np

from aiida_optimize.engines.base import OptimizationEngineWrapper
from aiida_optimize.engines._convergence import _ConvergenceImpl
from aiida_optimize.engines._result_mapping import Result
from aiida_optimize.helpers import get_nested_result

from aiida import orm
from aiida.orm.nodes.data.base import to_aiida_type

__all__ = ['TwoInputsTwoFactorsConvergence']


class _TwoInputsTwoFactorsConvergenceImpl(_ConvergenceImpl):
    def __init__(
        self,
        *,
        input_values: ty.List[ty.Any],
        tol: float,
        conv_thr: float,
        input_key: str,
        extra_input_key: str,
        result_key: str,
        convergence_window: int,
        array_name: ty.Optional[str],
        current_index: int,
        result_values: ty.List[ty.Any],
        initialized: bool,
        logger: ty.Optional[ty.Any],
        result_state: ty.Optional[ty.Dict[int, Result]] = None,
    ):
        super().__init__(
            input_values=input_values,
            tol=tol,
            input_key=input_key,
            result_key=result_key,
            convergence_window=convergence_window,
            array_name=array_name,
            current_index=current_index,
            result_values=result_values,
            initialized=initialized,
            logger=logger,
            result_state=result_state,
        )

        self.extra_input_key = extra_input_key
        self.conv_thr = conv_thr

    @property
    def _num_new_iters(self) -> int:
        """
        Determine the minimum number of additional outputs to have a hope
        of converging in the next step.
        """
        distance_triangle = self._distance_triangle
        results_in_window = self._result_window
        # Find location of the last calculation which creates out-of-tolerance
        # roughness, and do enough calculations so that it is no longer in the
        # next convergence window
        num_new_iters_tol = 0
        for i, row in enumerate(distance_triangle):
            if np.any(np.array(row) > self.tol):
                num_new_iters_tol = i + 1

        # count the number of results that not satisfy conv_thr
        # the more not satisfied the more add to the next iteration
        # if all under the conv_thr add (zero) new iteration
        num_new_iters_conv_thr = (np.array(results_in_window) >
                                  self.conv_thr).sum()

        # Use the max(greedy) number of the two types of criterion
        num_new_iters = max(num_new_iters_tol, num_new_iters_conv_thr)

        # Check that we don't go past the end of the input_values when trying
        # to remove the calculation that is too rough from the window
        # If we do, return -1 as an indication that convergence will not be
        # possible
        if self.current_index + num_new_iters > len(self.input_values):
            # num_new_iters = len(self.input_values) - self.current_index
            num_new_iters = -1

        return num_new_iters

    @property
    def is_converged(self) -> bool:
        """
        Check if convergence has been reached by calculating the Frobenius
        or 2-norm of the difference between all the result values / arrays
        and checking that the maximum distance between points in the
        convergence window is less than the tolerance.
        """
        if not self.initialized:
            return False

        # calculate pair distances between results
        distance_triangle = self._distance_triangle
        # flatten all the distances into a 1D list
        distances = list(itertools.chain(*distance_triangle))

        # check if the maximum distance is less than the tolerance
        # import ipdb; ipdb.set_trace()
        is_tol_converge = bool(np.max(distances) < self.tol)
        is_conv_thr_converge = bool(
            np.max(self._result_window) < self.conv_thr)

        res = is_tol_converge and is_conv_thr_converge

        return res

    def _create_inputs(self) -> ty.List[ty.Dict[str, orm.Float]]:
        """
        Create the inputs for the evaluation function.
        If the work chain is not initialized, the appropriate number of
        inputs to fill the convergence window are generated.
        Otherwise, one more input will be generated
        """
        if not self.initialized:
            # Do enough calculations to fill the initial convergence window
            num_new_iters = self.convergence_window
            self.initialized = True
        else:
            num_new_iters = self._num_new_iters

        self.current_index += num_new_iters
        inputs = [{
            self.input_key: to_aiida_type(self.input_values[i][0]),
            self.extra_input_key: to_aiida_type(self.input_values[i][1]),
        } for i in range(self.current_index -
                         num_new_iters, self.current_index)]

        return inputs

    def _get_optimal_result(self) -> ty.Tuple[int, orm.Node, orm.Node]:
        """
        Retrieve the converged index and result value (output value, _not_ max
        distance within convergence window)
        """
        opt_index = len(self.result_values) - self.convergence_window
        opt_input = self._result_mapping[opt_index].input[self.input_key]
        opt_output = get_nested_result(self._result_mapping[opt_index].output,
                                       self.result_key)

        return opt_index, opt_input, opt_output


class TwoInputsTwoFactorsConvergence(OptimizationEngineWrapper):
    """
    Wrapper class for convergence engine
    """

    _IMPL_CLASS = _TwoInputsTwoFactorsConvergenceImpl

    def __new__(  # pylint: disable=too-many-arguments,arguments-differ
        cls,
        input_values: ty.List[ty.Any],
        tol: float,
        conv_thr: float,
        input_key: str,
        extra_input_key: str,
        result_key: str,
        convergence_window: int = 2,
        array_name: ty.Optional[str] = None,
        logger: ty.Optional[ty.Any] = None,
    ) -> _TwoInputsTwoFactorsConvergenceImpl:
        return cls._IMPL_CLASS(  # pylint: disable=no-member
            input_values=input_values,
            tol=tol,
            conv_thr=conv_thr,
            input_key=input_key,
            extra_input_key=extra_input_key,
            result_key=result_key,
            convergence_window=convergence_window,
            array_name=array_name,
            current_index=0,
            result_values=[],
            initialized=False,
            logger=logger)
