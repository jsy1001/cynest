include "numpy.pxi"

from nestbase cimport NestSample, NestBase

cdef class OiSample(NestSample):

    cdef object like
    cdef ndarray uvars
    cdef readonly ndarray vars
    cdef ndarray varScale
    cdef ndarray varOffset

    cdef double _trialLogLhood(self, ndarray uvars)

    cdef double evolve(self, double logLstar)


cdef class OiNest(NestBase):

    cdef object like
    cdef ndarray varScale
    cdef ndarray varOffset

    cdef NestSample new_sample(self)
    
    cdef NestSample copy_sample(self, NestSample obj)


# Local Variables:
# mode: python
# End:
