include "stdlib.pxi"
include "math.pxi"
include "glib.pxi"
include "Python.pxi"
include "complexobject.pxi"
include "numpy.pxi"


# Initialize numpy - this MUST be done before any other code is executed.
import_array()

import numpy as Np
import os.path
import glob

pi = 3.1415926535897931
degree = pi/180.

cdef extern from "exchange.h":

    ctypedef int STATUS
    ctypedef char BOOL
    ctypedef double DATA

    ctypedef struct oi_vis2_record # need forward reference

    ctypedef struct oi_vis2:
        char *date_obs
        char *arrname
        char *insname
        long numrec
        int nwave
        oi_vis2_record *record

    ctypedef struct oi_vis2_record:
        int target_id
        double time
        double mjd
        DATA *vis2data
        DATA *vis2err
        double ucoord
        double vcoord
        int sta_index[2]
        BOOL *flag

    ctypedef struct oi_t3_record # need forward reference

    ctypedef struct oi_t3:
        char *date_obs
        char *arrname
        char *insname
        long numrec
        int nwave
        oi_t3_record *record

    ctypedef struct oi_t3_record:
        int target_id
        double time
        double mjd
        DATA *t3amp
        DATA *t3amperr
        DATA *t3phi
        DATA *t3phierr
        double u1coord
        double v1coord
        double u2coord
        double v2coord
        int sta_index[3]
        BOOL *flag

    ctypedef struct oi_wavelength:
        char *insname
        int nwave
        float *eff_wave
        float *eff_band

cdef extern from "oifile.h":

    ctypedef struct oi_fits:
        GList *vis2List
        GList *t3List
        GHashTable *wavelengthHash
        # no need to declare struct members we aren't interested in

    void init_oi_fits(oi_fits *)
    STATUS write_oi_fits(char *, oi_fits, STATUS *)
    STATUS read_oi_fits(char *, oi_fits *, STATUS *)
    char * format_oi_fits_summary(oi_fits *)

cdef extern from "oimerge.h":

    void merge_oi_fits_list(GList *, oi_fits *)


cdef class OiFits:

    """Wrapper class for OIFITSlib file-level API."""

    cdef oi_fits *this

    def __cinit__(self):
        self.this = <oi_fits *>malloc(sizeof(oi_fits))

    def __init__(self, fileName=None):
        if fileName is None:
            init_oi_fits(self.this)
        else:
            self.read(fileName)

    def __dealloc__(self):
        free(self.this)

    def __str__(self):
        return format_oi_fits_summary(self.this)

    def write(self, fileName):
        """Write object data to new FITS file."""
        cdef int status = 0
        write_oi_fits(fileName, self.this[0], &status)
        if status != 0:
            raise IOError, "Error writing OIFITS file %s" % fileName

    def read(self, fileName):
        """Read all OIFITS tables from FITS file."""
        cdef int status = 0
        read_oi_fits(fileName, self.this, &status)
        if status != 0:
            raise IOError, "Error reading OIFITS file %s" % fileName

def merge(*args):
    """Merge OiFits objects supplied as arguments."""
    return mergeList(args)

def mergeList(inList):
    """Merge OiFits objects in inList."""
    cdef OiFits inOi, outOi = OiFits()
    cdef GList *inGList = NULL # empty list
    for inOi in inList:
        inGList = g_list_append(inGList, inOi.this)
    merge_oi_fits_list(inGList, outOi.this)
    g_list_free(inGList)
    return outOi

def mergeFiles(inSpec):
    """Merge OiFits files matching pathname pattern using wildcards."""
    oiList = map(OiFits, glob.glob(inSpec))
    return mergeList(oiList)
    

# :TODO: move OiFlat, Likelihood to separate module
# :TODO: use cimport not include

def _pad(val):
    if isinstance(val, float):
        return '%9.3f' % val
    else:
        return '%9s' % val

def _columnize(seq):
    return ' '.join(map(_pad, seq))+'\n'


cdef class OiFlat:

    """Interface to OI data, using numpy structured arrays."""

    cdef readonly ndarray vis2
    cdef readonly ndarray t3

    def __init__(self, source):
        """Constructor.

        source may be OiFits instance or sequence of structured arrays
        (vis2, t3) or OIFITS filename.

        """
        if isinstance(source, OiFits):
            self.vis2 = self._vis2_from_oifits(source)
            self.t3 = self._t3_from_oifits(source)
        elif isinstance(source, str) and os.path.exists(source):
            o = OiFits(source)
            self.vis2 = self._vis2_from_oifits(o)
            self.t3 = self._t3_from_oifits(o)
        elif hasattr(source, '__iter__') and len(source) == 2:
            # is sequence of length 2
            self.vis2, self.t3 = source
        else:
            raise TypeError, "cannot construct from %s" % type(source)

    def __str__(self):
        s = _columnize(self.vis2.dtype.names)
        s += ''.join(map(_columnize, self.vis2))
        s += _columnize(self.t3.dtype.names)
        s += ''.join(map(_columnize, self.t3))
        return s

    def vis2_field(self, field):
        return self._sa_field(self.vis2, field)

    def t3_field(self, field):
        return self._sa_field(self.t3, field)

    def unique(self, field):
        all = Np.array([])
        for sa in [self.vis2, self.t3]:
            try:
                col = self._sa_field(sa, field)
            except ValueError:
                continue
            all = Np.concatenate((all, col))
        return Np.unique(all)

    def nights(self):
        """Return list of nights as MJDs."""
        nights = Np.unique((self.unique('mjd') + 2400000.50).astype(Np.int))
        return nights - 2400000.50

    def select(self, values=[], ranges=[]):
        """Return new instance containing view of selected records."""
        try:
            vis2 = self._sa_select(self.vis2, values, ranges)
        except ValueError:
            vis2 = self.vis2[0:0]
        try:
            t3 = self._sa_select(self.t3, values, ranges)
        except ValueError:
            t3 = self.t3[0:0]
        return OiFlat((vis2, t3))

    # :TODO: select data type(s)

    def non_null(self):
        """Return new instance containing view that excludes NULL t3amp."""
        condition = Np.logical_not(Np.isnan(self.t3['t3amp']))
        t3 = self.t3.compress(condition, axis=0)
        return OiFlat((self.vis2, t3))

    def _sa_field(self, sa, field):
        """Return specified field of structured array.

        Provides derived fields: 'uvradius', 'uvpa'
        :TODO: mjdmid ?

        """
        if field == 'uvradius':
            if 'ucoord' in sa.dtype.names:
                x = Np.sqrt(Np.square(sa['ucoord']) + Np.square(sa['vcoord']))
            elif 'u1coord' in sa.dtype.names:
                x1 = Np.sqrt(Np.square(sa['u1coord']) +
                             Np.square(sa['v1coord']))
                x2 = Np.sqrt(Np.square(sa['u2coord']) +
                             Np.square(sa['v2coord']))
                x3 = Np.sqrt(Np.square(sa['u1coord'] + sa['u2coord']) +
                             Np.square(sa['v1coord'] + sa['v2coord']))
                x = Np.array([x1, x2, x3]).max(axis=0)
        elif field == 'uvpa':
            if 'ucoord' in sa.dtype.names:
                x = Np.arctan2(sa['vcoord'], sa['ucoord'])/degree
            elif 'u1coord' in sa.dtype.names:
                raise ValueError
        else:
            x = sa[field]
        return x

    def _sa_select(self, ndarray sa, list values, list ranges):
        """Return view of selected records in structured array."""
        if len(sa) == 0: return sa
        condition = Np.ones((sa.shape[0],), Np.bool)
        for v in values:
            field, val = v
            condition &= (self._sa_field(sa, field) == val)
        for r in ranges:
            field, minval, maxval = r
            condition &= ((self._sa_field(sa, field) > minval) &
                          (self._sa_field(sa, field) < maxval))
        return sa.compress(condition, axis=0)

    def _vis2_from_oifits(self, OiFits source):
        """Convert OI_VIS2 tables to single structured array."""
        # :TODO: add insname etc.
        # :TODO: units
        cdef:
            oi_fits *oi
            oi_vis2 *vis2
            oi_vis2_record *v2Rec
            oi_wavelength *wv
            GList *curr
            int iRec, iWave
        vis2Desc = {'names': ('ucoord', 'vcoord', 'mjd', 'eff_wave', 'eff_band',
                              'vis2data', 'vis2err', 'flag',
                              'waveband', 'baseline'),
                    'formats': ('f8', 'f8', 'f8', 'f4', 'f4',
                                'f8', 'f8', 'bool',
                                'a10', 'a20')}
        recList = []
        oi = source.this
        curr = oi.vis2List
        while curr != NULL:
            vis2 = <oi_vis2 *>curr.data
            wv = <oi_wavelength *>g_hash_table_lookup(oi.wavelengthHash,
                                                      vis2.insname)
            for 0 <= iRec < vis2.numrec:
                v2Rec = &vis2.record[iRec]
                sta = [v2Rec.sta_index[0], v2Rec.sta_index[1]]
                sta.sort()
                bas = '%s:%d/%d' % (vis2.arrname, sta[0], sta[1])
                for 0 <= iWave < wv.nwave:
                    wb = '%.0f/%.0f' % (wv.eff_wave[iWave]/1e-9,
                                        wv.eff_band[iWave]/1e-9)
                    rec = (v2Rec.ucoord, v2Rec.vcoord, v2Rec.mjd,
                           wv.eff_wave[iWave], wv.eff_band[iWave],
                           v2Rec.vis2data[iWave], v2Rec.vis2err[iWave],
                           v2Rec.flag[iWave], wb, bas)
                    recList.append(rec)
            curr = curr.next
        a = Np.array(recList, dtype=vis2Desc)
        return a

    def _t3_from_oifits(self, OiFits source):
        """Convert OI_T3 tables to single structured array."""
        cdef:
            oi_fits *oi
            oi_t3 *t3
            oi_t3_record *t3Rec
            oi_wavelength *wv
            GList *curr
            int iRec, iWave
        t3Desc = {'names': ('u1coord', 'v1coord', 'u2coord', 'v2coord',
                            'mjd', 'eff_wave', 'eff_band',
                            't3amp', 't3amperr', 't3phi', 't3phierr', 'flag',
                            'waveband', 'triangle'),
                  'formats': ('f8', 'f8', 'f8', 'f8',
                              'f8', 'f4', 'f4',
                              'f8', 'f8', 'f8', 'f8', 'bool',
                              'a10', 'a20')}
        recList = []
        oi = source.this
        curr = oi.t3List
        while curr != NULL:
            t3 = <oi_t3 *>curr.data
            wv = <oi_wavelength *>g_hash_table_lookup(oi.wavelengthHash,
                                                      t3.insname)
            for 0 <= iRec < t3.numrec:
                t3Rec = &t3.record[iRec]
                tri = '%s:%d/%d/%d' % (t3.arrname, t3Rec.sta_index[0],
                                       t3Rec.sta_index[1], t3Rec.sta_index[2])
                for 0 <= iWave < wv.nwave:
                    wb = '%d/%d' % (wv.eff_wave[iWave]/1e-9,
                                    wv.eff_band[iWave]/1e-9)
                    rec = (t3Rec.u1coord, t3Rec.v1coord,
                           t3Rec.u2coord, t3Rec.v2coord, t3Rec.mjd,
                           wv.eff_wave[iWave], wv.eff_band[iWave],
                           t3Rec.t3amp[iWave], t3Rec.t3amperr[iWave],
                           t3Rec.t3phi[iWave], t3Rec.t3phierr[iWave],
                           t3Rec.flag[iWave], wb, tri)
                    recList.append(rec)
            curr = curr.next
        a = Np.array(recList, dtype=t3Desc)
        return a
    

cdef class Likelihood:

    """Class for calculating likelihood for OIFITS data."""

    cdef readonly OiFlat data
    cdef readonly int ndata
    cdef readonly OiFlat validData  # unflagged data
    cdef readonly object model

    def __init__(self, data, model):
        """Constructor."""
        self.data = data
        self.validData = data.select([('flag', False)])
        self.model = model
        self.ndata = (len(self.validData.vis2) + len(self.validData.non_null().t3) +
                      len(self.validData.t3)) # vis2 + t3amp + t3phi

    def model_vis2(self, data=None):
        """Return array of model V**2 corresponding to data points."""
        if data is None:
            data = self.data
        vis2 = data.vis2
        vsq = Np.vectorize(self.model.squared_vis)(
            vis2['ucoord'], vis2['vcoord'],
            vis2['mjd'], vis2['eff_wave'], vis2['eff_band'])
        return vsq

    def model_t3(self, data=None):
        """Return array of model bispectra corresponding to data points."""
        if data is None:
            data = self.data
        t3 = data.t3
        bis = Np.vectorize(self.model.bispectrum)(
            t3['u1coord'], t3['v1coord'], t3['u2coord'], t3['v2coord'],
            t3['mjd'], t3['eff_wave'], t3['eff_band'])
        return bis

    def model_t3amp(self, data=None):
        """Return array of model bis. amplitude corresponding to data points."""
        return abs(self.model_t3(data))

    def model_t3phi(self, data=None):
        """Return array of model closure phase corresponding to data points."""
        return Np.angle(self.model_t3(data), deg=True)

    def deg_freedom(self):
        """Return number of degrees of freedom."""
        return (self.ndata - self.model.nvar)

    def reduced_chisq(self, vars=None):
        """Return chi-squared per degree of freedom."""
        if vars is None:
            vars = self.model.get_vars()
        chisq = 2*self.nl_likelihood(vars)
        return chisq/self.deg_freedom()

    cpdef nl_posterior(self, ndarray vars):
        """Return neg. log posterior of data given model."""
        pr = self.model.nl_prior(vars)
        like = self.nl_likelihood(vars)
        return (pr + like)

    cpdef nl_likelihood(self, ndarray vars):
        """Return neg. log likelihood (chi-squared/2) of data given model."""
        cdef:
            int i
            ndarray tab
            double amp, phi, diffsq, sum
            complex bis
        self.model.set_vars(vars)
        sum = 0.

        # Squared visibilities
        tab = self.validData.vis2
        for 0 <= i < len(tab):
            amp = self.model.squared_vis(
                tab['ucoord'][i], tab['vcoord'][i],
                tab['mjd'][i], tab['eff_wave'][i], tab['eff_band'][i])
            diffsq = (tab['vis2data'][i] - amp)**2.
            sum += diffsq/(2 * tab['vis2err'][i]**2.)

        # Bispectra
        tab = self.validData.t3
        for 0 <= i < len(tab):
            bis = self.model.bispectrum(
                tab['u1coord'][i], tab['v1coord'][i],
                tab['u2coord'][i], tab['v2coord'][i],
                tab['mjd'][i], tab['eff_wave'][i], tab['eff_band'][i])
            if not Np.isnan(tab['t3amp'][i]):
                amp = abs(bis)
                diffsq = (tab['t3amp'][i] - amp)**2.
                sum += diffsq/(2 * tab['t3amperr'][i]**2.)
            # NB phase calculations modulo 360 degrees
            phi = Np.angle(bis, deg=True)
            diffsq = self.diffmod360(tab['t3phi'][i], phi)**2.
            sum += diffsq/(2 * tab['t3phierr'][i]**2.)

        return sum

    cpdef diffmod360(self, double a, double b):
        """Return modulo 360 degree difference of a and b."""
        cdef double c, diff
        #diff = fmod(a - b, 360.) # :BUG:
        c = a - b
        diff = c - floor(c/360.)*360.
        if diff > 180.:
            diff = 360. - diff
        return diff

# Local Variables:
# mode: python
# End:
