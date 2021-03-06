# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvec import BaseAdvectionElements


class BaseFluidElements(object):
    _dynvarmap = {2: ['rho', 'u', 'v', 'p'],
                  3: ['rho', 'u', 'v', 'w', 'p']}

    def _process_ics(self, ics):
        rho, p = ics[0], ics[-1]

        # Multiply velocity components by rho
        rhovs = [rho*c for c in ics[1:-1]]

        # Compute the energy
        gamma = self._cfg.getfloat('constants', 'gamma')
        E = p/(gamma - 1) + 0.5*rho*sum(c*c for c in ics[1:-1])

        return [rho] + rhovs + [E]


class EulerElements(BaseFluidElements, BaseAdvectionElements):
    def set_backend(self, backend, nscalupts):
        super(EulerElements, self).set_backend(backend, nscalupts)

        # Register our flux kernel
        backend.pointwise.register('pyfr.solvers.euler.kernels.tflux')

    def get_tdisf_upts_kern(self):
        tplargs = dict(ndims=self.ndims, nvars=self.nvars,
                       c=self._cfg.items_as('constants', float))

        return self._be.kernel('tflux', tplargs, dims=[self.nupts, self.neles],
                               u=self.scal_upts_inb, smats=self._smat_upts,
                               f=self._vect_upts[0])
