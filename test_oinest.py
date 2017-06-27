import oifits
from vmodel import *
import oinest

import oiopt


def test():
    """Test function."""
    # Read data
    data = oifits.OiFlat("test.oifits")
    print data

    # Define model
    mod = Model(Component('Ab', flux=0.89, ldType='uniform', diam=6.9),
                Component('Aa', flux=1.00, ldType='uniform', diam=8.5,
                          pos=(54.9, 26.6)))
    mod.def_vars(Ab_flux=("uniform", 0.7, 1.1),
                 Aa_pos_radius=("uniform", 40., 70.),
                 Aa_pos_theta=("uniform", 10, 40.))
    print mod
    
    # Run nested sampling
    nest = oinest.OiNest(data, mod, maxIter=500)
    samples = nest.run()
    # Save posterior samples
    f = open("nest.dat", "w")
    for s in samples:
        f.write("%f %f %f\n" % (s.vars[0], s.vars[1], s.logL))
    f.close()

    # Compare with BFGS
    l = oifits.Likelihood(data, mod)
    print "Initial reduced chi^2 = %.1lf" % (l.reduced_chisq(),)
    mod, sol, err = oiopt.optimize(l, verbose=True)


if __name__ == '__main__':
    test()
