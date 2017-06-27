from math import *

cdef extern from "stdlib.h":
    cdef enum dummy2:
        RAND_MAX
    int rand()

cdef double rand_uniform():
    """Return random value from uniform distribution inside (0,1)."""
    return ((rand()+0.5)/(RAND_MAX+1.0))

cdef double logplus(double x, double y):
    """logarithmic addition log(exp(x)+exp(y))."""
    if x>y:
        return x+log(1+exp(y-x))
    else:
        return y+log(1+exp(x-y))


cdef class NestSample:

    cdef double evolve(self, double logLstar):
        pass


cdef class NestBase:

    def __init__(self, nObj=100, maxIter=1000):
        self.nObj = nObj
        self.maxIter = maxIter

    def run(self):
        # :TODO: expose samples
        cdef list samples
        cdef list Objects           # Collection of n objects
        cdef double logwidth        # ln(width in prior mass)
        cdef double logLstar        # ln(Likelihood constraint)
        cdef double H    = 0.0      # Information, initially 0
        # :TODO:
        #cdef double logZ =-DBL_MAX  # ln(Evidence Z, initially 0)
        cdef double logZ = -1e99
        cdef double logZnew         # Updated logZ
        cdef int n
        cdef int i                  # Object counter
        cdef NestSample obj
        cdef NestSample worst       # Worst object
        cdef int iworst             # Index of worst object
        cdef int icopy              # Index of duplicated object
        cdef int nest               # Nested sampling iteration count

        # :TODO: option to seed random number generator
        # Initialize lists
        n = self.nObj
        Objects = []
        samples = []
        for 0 <= i < n:
            Objects.append(self.new_sample())  # random sample
        # Outermost interval of prior mass
        logwidth = log(1.0 - exp(-1.0 / n))

        # NESTED SAMPLING LOOP ______________________________________________
        for 0 <= nest < self.maxIter:
            
            for 0 <= i < n:
                obj = Objects[i]

            # Worst object in collection, with Weight = width * Likelihood
            worst = Objects[0]
            for obj in Objects[1:]:
                if obj.logL < worst.logL:
                    worst = obj;
            iworst = Objects.index(worst)
            print "%4d  %3d  %lf" % (nest, iworst, worst.logL)
            #worst._checkLogL()
                
            worst.logWt = logwidth + worst.logL
            # Update Evidence Z and Information H
            logZnew = logplus(logZ, worst.logWt)
            H = (exp(worst.logWt - logZnew) * worst.logL
                 + exp(logZ - logZnew) * (H + logZ) - logZnew)
            logZ = logZnew
            # Posterior Samples
            samples.append(self.copy_sample(worst))
            # Kill worst object in favour of copy of different survivor
            while True:
                icopy = <int>(n * rand_uniform()) % n
                if not (icopy == iworst and n > 1 ):  # don't kill if n only 1
                    break
            logLstar = worst.logL  # new likelihood constraint
            Objects[iworst] = self.copy_sample(Objects[icopy])  # overwrite worst object
            # Evolve copied object within constraint
            obj = Objects[iworst]
            obj.evolve(logLstar)
            # Shrink interval
            logwidth -= 1.0 / n
        # _______ NESTED SAMPLING LOOP (might be ok to terminate early)
        
        # :TODO: final correction

        # Exit with evidence Z, information H, and optional posterior Samples
        print "# iterates = %d" % (nest,)
        print "Evidence: ln(Z) = %g +- %g" % (logZ, sqrt(H/n))
        print "Information: H = %g nats = %g bits" % (H, H/log(2.))
        self.results(samples, logZ)
        return samples

    def results(self, samples, logZ):
        pass

    cdef NestSample new_sample(self):
        raise NotImplementedError

    cdef NestSample copy_sample(self, NestSample obj):
        raise NotImplementedError

# Local Variables:
# mode: python
# End:
