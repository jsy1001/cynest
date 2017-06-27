from math import *

from nestbase cimport NestSample, NestBase, rand_uniform
#from nestbase import *


cdef class LightHouseSample(NestSample):

    def __str__(self):
        return "u=%lf v=%lf logL=%lf" % (self.u, self.v, self.logL)

    cdef double _trialLogLhood(self, double u, double v):
        cdef int k  # data index
        cdef double logL = 0  # logLikelihood accumulator
        cdef double x, y
        x = 4.0 * u - 2.0
        y = 2.0 * v
        for 0 <= k < self.ndata:
            logL += log((y/3.1416) / ((self.data[k]-x)*(self.data[k]-x) + y*y))
        return logL

    cdef double evolve(self, double logLstar):
        cdef double step = 0.1
        cdef int m = 20
        cdef int accept = 0
        cdef int reject = 0
        while m > 0:
            # Trial parameters
            tryu = self.u + step * (2.*rand_uniform() - 1.)
            tryv = self.v + step * (2.*rand_uniform() - 1.)
            tryu -= floor(tryu)
            tryv -= floor(tryv)
            # Accept if and only if within hard likelihood constraint
            tryLogL = self._trialLogLhood(tryu, tryv)
            if tryLogL > logLstar:
                self.u = tryu
                self.v = tryv
                self.logL = tryLogL
                # logWt now incorrect, but will be re-evaluated before use
                accept += 1
            else:
                reject += 1
            # Refine step-size to let acceptance ratio converge around 50%
            if accept > reject:
                step *= exp(1.0/accept)
            if accept < reject:
                step /= exp(1.0/reject)
            m -= 1


cdef class LightHouse(NestBase):

    def __init__(self, nObj=100, maxIter=1000):
        cdef int k
        NestBase.__init__(self, nObj, maxIter)
        D = [ 4.73,  0.45, -1.73,  1.09,  2.19,  0.12,
              1.31,  1.00,  1.32,  1.07,  0.86, -0.49, -2.59,  1.73,  2.11,
              1.61,  4.98,  1.71,  2.23,-57.20,  0.96,  1.25, -1.56,  2.45,
              1.19,  2.17,-10.66,  1.91, -4.16,  1.92,  0.10,  1.98, -2.51,
              5.55, -0.47,  1.91,  0.95, -0.78, -0.84,  1.72, -0.01,  1.48,
              2.70,  1.21,  4.41, -4.79,  1.33,  0.81,  0.20,  1.58,  1.29,
              16.19,  2.75, -2.38, -1.79,  6.50,-18.53,  0.72,  0.94,  3.64,
              1.94, -0.11,  1.57,  0.57]
        self.ndata = len(D)
        for 0 <= k < self.ndata:
            self.data[k] = D[k]

    def results(self, samples, logZ):
        cdef double x, y
        cdef double mx = 0.0, mxx = 0.0  # 1st and 2nd moments of x
        cdef double my = 0.0, myy = 0.0  # 1st and 2nd moments of y
        cdef double w                    # proportional weight
        cdef LightHouseSample obj
        for obj in samples:
            x = 4.0 * obj.u - 2.0
            y = 2.0 * obj.v
            w = exp(obj.logWt - logZ)
            mx  += w * x
            mxx += w * x * x
            my  += w * y
            myy += w * y * y
        print "mean(x) = %g, stddev(x) = %g" % (mx, sqrt(mxx-mx*mx))
        print "mean(y) = %g, stddev(y) = %g" % (my, sqrt(myy-my*my))
         
    cdef NestSample new_sample(self):
        cdef LightHouseSample obj
        obj = LightHouseSample()
        obj.ndata = self.ndata
        obj.data = self.data
        obj.u = rand_uniform()
        obj.v = rand_uniform()
        # :TODO: obj.setuv() ?
        obj.logL = obj._trialLogLhood(obj.u, obj.v)
        return obj

    cdef NestSample copy_sample(self, NestSample obj):
        cdef LightHouseSample src, dest
        src = obj
        dest = LightHouseSample()
        dest.logWt = src.logWt
        dest.logL = src.logL
        dest.ndata = src.ndata
        dest.data = src.data
        dest.u = src.u
        dest.v = src.v
        return dest
        

# Local Variables:
# mode: python
# End:
