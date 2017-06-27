cdef double rand_uniform()


cdef class NestSample:

    # Read-only access from python for post-processing of posterior samples
    cdef readonly double logWt
    cdef readonly double logL

    cdef double evolve(self, double logLstar)

    # :TODO:
    #cdef copy(self) ??


cdef class NestBase:

    cdef int nObj     # number of objects
    cdef int maxIter  # maximum number of iterations

    cdef NestSample new_sample(self)
    
    cdef NestSample copy_sample(self, NestSample obj)

# Local Variables:
# mode: python
# End:
