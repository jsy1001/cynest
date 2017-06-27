include "math.pxi"
include "Python.pxi"
include "complexobject.pxi"
include "numpy.pxi"

# Initialize numpy - this MUST be done before any other code is executed.
import_array()

import numpy as Np
import scipy.special

pi = 3.1415926535897931
mas = pi/180./3600./1000.
degree = pi/180.


# we use this enum to avoid string comparisons in Component.complex_vis()
ctypedef enum VMOD_LD_TYPE:
    VMOD_NONE
    VMOD_UNIFORM
    VMOD_GAUSSIAN
    VMOD_HESTROFFER

cdef class Component:

    """Single component of a model."""

    #cdef readonly char *name
    cdef readonly object name
    cdef readonly char *shapeType
    cdef readonly char *ldType
    cdef VMOD_LD_TYPE ld
    cdef public double ldP1
    cdef public double flux
    cdef public double pos_radius
    cdef public double pos_theta
    cdef public double diam
    cdef public double ratio
    cdef public double pa

    def __init__(self, name, flux=1.0, pos=(0.,0.),
                 shapeType='disc', ldType='gaussian', ldP1=1., diam=10.,
                 ratio=None, pa=None):
        # :TODO: base class that implements shift/rotate/scale?
        # rotate entire model?
        self.name = name
        # flux
        self.flux = flux
        # position
        self.pos_radius = pos[0]
        self.pos_theta = pos[1]
        # shapeType
        validShapes = ['point', 'disc', 'ellipse']
        if shapeType not in validShapes:
            raise ValueError, 'invalid shapeType'
        self.shapeType = shapeType
        # ldType
        if shapeType == 'point':
            self.ldType = None
            self.ld = VMOD_NONE
        else:
            validLd = ['uniform', 'gaussian', 'hestroffer']
            if ldType not in validLd:
                raise ValueError, 'invalid ldType'
            else:
                self.ldType = ldType
                self.ld = VMOD_UNIFORM + validLd.index(ldType)
                if self.ld == VMOD_HESTROFFER:
                    self.ldP1 = ldP1
                else:
                    self.ldP1 = -1.  # ignored
        # shape parameters
        self.diam = diam # ignored for 'point'
        if shapeType == 'ellipse':
            if ratio is None:
                raise ValueError, 'ratio not specified'
            if pa is None:
                raise ValueError, 'pa not specified'
            self.ratio = ratio
            self.pa = pa
        else:
            self.ratio = 1.0
            self.pa = 0.0

    def __repr__(self):
        """Return string that can be passed to eval() to recreate self."""
        s = """Component('%s', flux=%f, pos=(%f,%f),
                 shapeType='%s', ldType='%s', ldP1=%f, diam=%f""" % (
            self.name, self.flux, self.pos_radius, self.pos_theta,
            self.shapeType, self.ldType, self.ldP1, self.diam)
        if self.shapeType == 'ellipse': # :BUG: use enum?
            print "__repr__ '%s'" % self.shapeType ##
        s += ", ratio=%f, pa=%f" % (self.ratio, self.pa)
        s += ")"
        return s

    def __getattr__(self, paramSpec):
        """Get component parameter."""
        if paramSpec == 'pos':
            return (self.pos_radius, self.pos_theta)
        else:
            raise AttributeError, paramSpec
        
    #def __setattr__(self, paramSpec, val):
    #    """Set component parameter."""
    #    if paramSpec == 'pos':
    #        self.pos_radius, self.pos_theta = val
    #    else:
    #        print self, paramSpec, val
    #        object.__setattr__(self, paramSpec, val) # :BUG:

    def copy(self):
        """Return copy of self."""
        return eval(repr(self))

    cpdef complex_vis(self, double u0, double v0, double mjd,
                      float wavelen, float bandwidth):
        """Return complex visibility."""
        cdef double u, v
        cdef double paRad, diamRad
        cdef double x1, x2, rho, vv
        cdef complex vis
        # reverse baseline convention so we can use mfit formulae
        u = -u0
        v = -v0
        paRad = self.pa*degree
        diamRad = self.diam*mas
        x1 = (self.ratio**2) * (((u * cos(paRad)) - (v * sin(paRad)))**2)
        x2 = (((v * cos(paRad)) + (u * sin(paRad)))**2)
        rho = (sqrt(x1 + x2))/wavelen
        xx = pi * fabs(diamRad) * rho
        if self.ld == VMOD_GAUSSIAN:
            vv = exp(-xx*xx/(4.0 * log(2.0)))
        elif self.ld == VMOD_UNIFORM:
            vv = 2*scipy.special.j1(xx)/xx
        elif self.ld == VMOD_HESTROFFER:
            order = self.ldP1/2. + 1.
            v1 = scipy.special.gamma(self.ldP1/2. + 2.)
            v2 = scipy.special.jv(order, xx)
            v3 = (xx/2.)**order
            vv = v1*v2/v3
        else:
            raise NotImplementedError
        thetaRad = self.pos_theta*degree
        radiusRad = self.pos_radius*mas
        arg = 2*pi*radiusRad*(u*sin(thetaRad) + v*cos(thetaRad))/wavelen
        vis = complex(self.flux*vv*cos(arg), self.flux*vv*sin(arg))
        return vis


cdef class Model:

    """Sum of multiple Component instances."""

    cdef list cptList
    cdef readonly dict components
    cdef readonly int nvar
    cdef readonly list varNames
    cdef readonly dict priors

    def __init__(self, *args):
        self.cptList = []
        self.components = {}
        for a in args:
            if not isinstance(a, Component):
                raise TypeError, 'not a Component instance'
            self.cptList.append(a)
            self.components[a.name] = a
        self.fix_all()

    def __len__(self):
        """Get number of components."""
        return len(self.cptList)

    def __getitem__(self, key):
        """Get model component by name or index."""
        if isinstance(key, int):
            return self.cptList[key]
        else:
            return self.components[key]

    def __getattr__(self, paramSpec):
        """Get model parameter.
        
        Parameter should be specified as <component name>_<parameter name>.
        """
        name, param = paramSpec.split('_', 1)
        return self.components[name].__getattribute__(param)
        
    def __setattr__(self, paramSpec, val):
        """Set model parameter.
        
        Parameter should be specified as <component name>_<parameter name>.
        """
        name, param = paramSpec.split('_', 1)
        self.components[name].__setattr__(param, val)

    def add(self, cpt):
        """Add component to model.

        Parameters of new component will initially be fixed.

        """
        self.cptList.append(cpt)
        self.components[cpt.name] = cpt

    def __repr__(self):
        s = "Model("+repr(self.cptList[0])
        for c in self.cptList[1:]:
            s += ",\n      "+repr(c)
        s += ")"
        return s

    def __str__(self):
        return repr(self)

    def save(self, fileName):
        """Save model to file."""
        f = open(fileName, 'w')
        f.write(repr(self))
        f.close()

    def spectrum(self, func, u, v, mjd, wavelens, bandwidths):
        """Call specified function to obtain array of spectral values."""
        nwave = len(wavelens)
        val = func(u, v, mjd, wavelens[0], bandwidths[0])
        spec = Np.zeros((nwave,), type(val))
        spec[0] = val
        for i in range(1, nwave):
            spec[i] = func(u, v, mjd, wavelens[i], bandwidths[i])
        return spec
            
    cpdef complex_vis(self, double u, double v, double mjd,
                      float wavelen, float bandwidth):
        """Return complex visibility."""
        cdef complex vis
        cdef double totFlux
        vis = 0j
        totFlux = 0.
        # :TODO: bandwidth smearing
        for cpt in self.cptList:
            vis += cpt.complex_vis(u, v, mjd, wavelen, bandwidth)
            totFlux += cpt.flux
        # renormalize
        vis = vis/totFlux
        return vis

    cpdef squared_vis(self, double u, double v, double mjd,
                    float wavelen, float bandwidth):
        """Return squared visibility."""
        return abs(self.complex_vis(u, v, mjd, wavelen, bandwidth))**2.

    cpdef bispectrum(self, double u1, double v1, double u2, double v2,
                     double mjd, float wavelen, float bandwidth):
        """Return bispectrum."""
        cdef complex vis1, vis2, vis3, bis
        vis1 = self.complex_vis(u1, v1, mjd, wavelen, bandwidth)
        vis2 = self.complex_vis(u2, v2, mjd, wavelen, bandwidth)
        vis3 = self.complex_vis(u1+u2, v1+v2, mjd, wavelen, bandwidth)
        bis = vis1*vis2*(vis3.conjugate())
        return bis

    def fix_all(self):
        """Make all parameters non-variable."""
        self.nvar = 0
        self.varNames = []
        self.priors = {}

    def def_vars(self, **kw):
        """Specify (further) parameter(s) to vary and corresponding priors.

        Parameters are specified as keyword arguments i.e.
        <component name>_<parameter name>=<prior spec>

        Priors are specified in one of the following ways:
        None -- No prior (maximum likelihood fitting)
        ('uniform', min, max) -- Uniform prior
        ('gaussian', mean, sigma) -- Gaussian prior
        
        """
        keys = kw.keys()
        keys.sort()
        for key in keys:
            pr = kw[key]
            if not isinstance(pr, (type(None), tuple)):
                raise ValueError, \
                    'invalid prior: %s (expecting None or tuple)' % pr
            if (isinstance(pr, tuple) and 
                (len(pr) != 3 or pr[0] not in ['uniform', 'gaussian'])):
                raise ValueError, 'invalid prior tuple: %s' % pr
            self.varNames.append(key)
            self.priors[key] = pr
        self.nvar = len(self.varNames)
        # :TODO: check for illegal combinations

    cpdef nl_prior(self, vars):
        """Return negative log prior."""
        cdef int i
        cdef double sum
        sum = 0.
        for 0 <= i < self.nvar:
            key = self.varNames[i]
            p = self.priors[key]
            # :TODO: use enum instead of string?
            if p is not None:
                if p[0] == 'uniform' and (vars[i] < p[1] or vars[i] > p[2]):
                    sum += 1e9
                if p[0] == 'gaussian':
                    sum += (vars[i] - p[1])**2./(2*p[2]**2.)
        return sum

    def get_prior(self, ivar):
        """Return prior for variable with specified index."""
        key = self.varNames[ivar]
        return self.priors[key]

    def get_vars(self):
        """Return array of variable values."""
        vars = []
        for key in self.varNames:
            name, param = key.split('_', 1)
            vars.append(self.components[name].__getattribute__(param))
        return Np.array(vars)

    cpdef set_vars(self, ndarray vars):
        """Set variable parameters."""
        cdef int i
        for 0 <= i <self.nvar:
            name, param = self.varNames[i].split('_', 1)
            self.components[name].__setattr__(param, vars[i])

    def copy(self, vars=None):
        """Return copy of self, with new variable values."""
        newMod = Model()
        for c in self.cptList:
            newMod.add(c.copy())
        if vars is not None:
            newMod.set_vars(vars)
        return newMod


def read_model(fileName):
    """Read file written by Model.save()."""
    f = open(fileName)
    str = f.read()
    f.close()
    mod = eval(str)
    return mod

# Local Variables:
# mode: python
# End:
