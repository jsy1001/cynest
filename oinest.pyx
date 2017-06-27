from math import *
import numpy as Np

from nestbase cimport NestSample, NestBase, rand_uniform
import oifits


cdef class OiSample(NestSample):

    def __str__(self):
        self.like.model.set_vars(self.vars)
        return "%s logL=%lf chi^2=%lf" % (self.like.model, self.logL,
                                          -2.*self.logL/self.like.deg_freedom())

    def _checkLogL(self):
        diff = self._trialLogLhood(self.uvars) - self.logL
        if abs(diff) > 0.1:
            print "%s %s %lf %lf\n" % (
                self, self.uvars, self._trialLogLhood(self.uvars), self.logL)

    cdef double _trialLogLhood(self, ndarray uvars):
        # :TODO: optimize
        cdef ndarray vars
        cdef int i
        vars = Np.zeros(uvars.shape, Np.float)
        for 0 <= i < uvars.shape[0]:
            vars[i] = self.varScale[i]*uvars[i] + self.varOffset[i]
        return -self.like.nl_posterior(vars)

    cdef double evolve(self, double logLstar):
        # :TODO: optimize
        cdef double step = 0.1
        cdef int m = 20
        cdef int accept = 0
        cdef int reject = 0
        cdef ndarray tryVars
        cdef int i
        tryVars = Np.zeros(self.uvars.shape, Np.float)
        while m > 0:
            # Trial parameters
            for 0 <= i < self.uvars.shape[0]:
                tryVars[i] = self.uvars[i] + step * (2.*rand_uniform() - 1.)
                tryVars[i] -= floor(tryVars[i])
            # Accept if and only if within hard likelihood constraint
            tryLogL = self._trialLogLhood(tryVars)
            if tryLogL > logLstar:
                ##self.uvars = tryVars # :BUG: reference
                for 0 <= i < self.uvars.shape[0]:
                    self.uvars[i] = tryVars[i]
                self.vars = self.varScale*self.uvars + self.varOffset
                self.logL = tryLogL
                accept += 1
            else:
                reject += 1
            # Refine step-size to let acceptance ratio converge around 50%
            if accept > reject:
                step *= exp(1.0/accept)
            if accept < reject:
                step /= exp(1.0/reject)
            m -= 1


cdef class OiNest(NestBase):

    def __init__(self, data, model, nObj=100, maxIter=2000):
        """Read data, define model."""
        NestBase.__init__(self, nObj, maxIter)
        self.like = oifits.Likelihood(data, model)
        self.varScale = Np.zeros((model.nvar,), Np.float)
        self.varOffset = Np.zeros((model.nvar,), Np.float)
        for 0 <= i < model.nvar:
            pmin, pmax = model.get_prior(i)[1:]
            self.varScale[i] = pmax - pmin
            self.varOffset[i] = pmin
        print self.varScale
        print self.varOffset

    def results(self, samples, logZ):
        cdef ndarray v, vv
        cdef int nvar, i
        cdef OiSample obj
        print samples[-1]
        nvar = self.like.model.nvar
        v = Np.zeros((nvar,), Np.float)
        vv = Np.zeros((nvar,), Np.float)
        for obj in samples:
            w = exp(obj.logWt - logZ)
            v += w * obj.uvars
            vv += w * obj.uvars * obj.uvars
        for 0 <= i < nvar:
            mean = self.varScale[i]*v[i] + self.varOffset[i]
            stddev = self.varScale[i]*sqrt(vv[i]-v[i]*v[i])
            print '[%d] %15s = %10.3f +_ %9.3f' % (
                i, self.like.model.varNames[i], mean, stddev)
        # :TODO: uncertainties due to sampling
        

    cdef NestSample new_sample(self):
        cdef OiSample obj
        cdef int nvar, i
        nvar = self.like.model.nvar
        obj = OiSample()
        obj.like = self.like
        obj.varScale = self.varScale
        obj.varOffset = self.varOffset
        obj.uvars = Np.zeros((nvar,), Np.float)
        for 0 <= i < nvar:
            obj.uvars[i] = rand_uniform()
        obj.vars = obj.varScale*obj.uvars + obj.varOffset
        # :TODO: obj.setvars() ?
        obj.logL = obj._trialLogLhood(obj.uvars)
        return obj

    cdef NestSample copy_sample(self, NestSample obj):
        cdef OiSample src, dest
        cdef int nvar, i
        nvar = self.like.model.nvar
        src = obj
        dest = OiSample()
        dest.logWt = src.logWt
        dest.logL = src.logL
        dest.like = src.like
        #dest.uvars = src.uvars # :BUG: reference not copy
        dest.uvars = Np.zeros((nvar,), Np.float)
        dest.vars = Np.zeros((nvar,), Np.float)
        for 0 <= i < nvar:
            dest.uvars[i] = src.uvars[i]
            dest.vars[i] = src.vars[i]
        dest.varScale = src.varScale
        dest.varOffset = src.varOffset
        return dest


# Local Variables:
# mode: python
# End:
