import oifits
from vmodel import *
import oinest

import oiopt
from pylab import *
import oiplot


if __name__ == '__main__':
    # Read data
    data = oifits.OiFlat("alp_ori_05_sel.oifits")
    # Note COAST software writes OIFITS with lots of flagged dummy values
    data = data.select([('flag', False)],
                       [('eff_wave', 880e-9, 940e-9)])
    print data

    # Define model
    mod = Model(Component('star', flux=1.0,
                          ldType='hestroffer', ldP1=1., diam=71.6)
                , Component('spot1', flux=0.4, ldType='gaussian', diam=5.,
                            pos=(9.3, 98))
                , Component('spot2', flux=0.4, ldType='gaussian', diam=5.,
                            pos=(7.4, 272))
                )
    mod.def_vars(star_diam=("uniform", 40., 80.)
                 , spot1_flux=("uniform", 0., 0.5),
                 spot1_pos_radius=("uniform", 0., 40),
                 spot1_pos_theta=("uniform", 0., 360.)
                 , spot2_flux=("uniform", 0., 0.5),
                 spot2_pos_radius=("uniform", 0., 40),
                 spot2_pos_theta=("uniform", 0., 360.)
                 )
    print mod
    
    # Run nested sampling
    nest = oinest.OiNest(data, mod, maxIter=1000)
    nest.run()

    # Compare with BFGS
    l = oifits.Likelihood(data, mod)
    print "Initial reduced chi^2 = %.1lf" % (l.reduced_chisq(),)
    mod, sol, err = oiopt.optimize(l, verbose=True)

    #clf()
    #oiplot.plot_vis2(l, 'uvradius')
    #oiplot.plot_t3phi(l, 'mjd')
    #legend()
    #draw()
    #show()
