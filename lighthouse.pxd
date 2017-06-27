from nestbase cimport NestSample, NestBase

cdef class LightHouseSample(NestSample):

    cdef int ndata
    cdef double *data
    cdef double u
    cdef double v

    cdef double _trialLogLhood(self, double u, double v)

    cdef double evolve(self, double logLstar)


cdef class LightHouse(NestBase):

    cdef int ndata
    cdef double data[64]

    cdef NestSample new_sample(self)
    
    cdef NestSample copy_sample(self, NestSample obj)

# Local Variables:
# mode: python
# End:
