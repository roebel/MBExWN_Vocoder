# AUTHOR:  G.Degottex/A.Roebel
# COPYRIGHT: Copyright(c) 2008 - 2022 IRCAM/CNRS - Degottex/Roebel

from __future__ import division, print_function, absolute_import
import numpy as np
try:
    from . import FglotLFsynthparams as gpsp
except ImportError:
    from glottis import FglotLFsynthparams as gpsp

def _exp_imag(x):
    return np.cos(x) + 1j*np.sin(x)

#@profile
def FglotspecLF(f, oq, am, ta,  Ee=1, alpha=-1, epar=-1, orig=0, get_derivative=True, dtype=np.float64):
    """
    Spectrum of the Glottal Flow derivative (or glottal flow) 
    according to the LF model (Using Roebel&Degottex formulas)
    
    USAGE
       function [spec, spec1, spec2, alpha, epar] = FglotspecILF(f, oq, am, ta, orig, Ee, alpha, epar)
       
    INPUT
    f        : frequency vector normalized by fundamental frequency
               (value k correspond to the harmonic k)
               
    oq       : open phase coefficient in terms of the fundamental period T0
               in terms of the LF model parameters te and ta the open phase
               it depends on the convention you use:
                  usually: te/T0  (and there)
                  by Fant: (te+ta)/T0
               in ]0;1[
  
    am       : asymmetry coefficient expressed as coefficient relative to the
               open phase of the glottal puls (tp=te*am).
               in ]0.5;1[
               should be kept > 0.53 to avoid numerical instability
  
    ta       : effective closing time in terms of the fundamental period
               in [0;1-oq[
  
    [Ee]     : maximum negative excitation energy
               (default to 1)
 
    [alpha]  : precomputed synthesis parameter (like in [1])
               (default compute it internaly)
    [epar]   : precomputed synthesis parameter (NOT epsilon in glotspecLFsimple ! epar=epsilon*ta) (epsilon in [1])
               (default compute it internaly)
 
    [orig]   : time origin for Fourier transform (0=start of puls; oq=GCI)
               set to oq to have it at maximum negative excitation
               (default to 0)
  
    [get_derivative] : output glottal flow without addiding derivative related to radiation

    OUTPUT
    spec     : spectrum related to the glottal puls covering open and closeing phase (t=0..tE=TA)
    spec1    : spectrum related to the first part of the pulse form up to GCI
    spec2    : spectrum related to the relaxation phase after the GCI
    alpha    : synthesis parameter (like in [1])
    epar     : synthesis parameter (NOT epsilon in glotspecLFsimple ! epar=epsilon*ta) (epsilon in [1])
    ta       : the ta parameter can be potentially adapted during the process of finding alpha and epar
    REFERENCES
    [1] Fant, G. and Liljencrants, J. and Lin, Q., "A four-parameter model of glottal flow", STL-QPSR, vol. 4, pp. 1-13, 1985.
    
    AUTHORS
      G. Degottex 2008-2012 (matlab version)
      A. Roebel   2013-2022 (python version)

    COPYRIGHT
      Copyright (c) 2008-2022  IRCAM/CNRS - Degottex, Roebel

    """

    # set default values
    #if nargin<5; orig=0; end
        
    # check parameters
    if ta>0 and alpha > 0 and epar < 0 :
        raise RuntimeError('if ta>0 and alpha is given, epar has to be present too')
    eps = np.finfo(np.float64).eps
    if(oq <= eps or oq >=(1-eps)):
      raise RuntimeError('open quotient {0:f} out of range'.format(oq))
    if(am <= 0.5 or am >=(1-eps)):
        raise RuntimeError('asymetry {0:f} is out of range'.format(am))
    if(ta < 0 or ta >(1-oq)):
        raise RuntimeError('return phase length(ta) {0:f} is out of range'.format(ta))
    
    # time parameters
    # tc = 1;	# implicitly, and so for a normalized period ..
    te = dtype(oq)

    # create/solve for synthesis parameters	wg, epar, alpha, E0
    wg = dtype(np.pi/(oq*am))

    if alpha <= 0 :
        # alpha and epar have to be computed
        [alpha, epar, ta] = gpsp.FglotLFsynthparams(oq, am, ta)
    alpha =dtype(alpha)
    ta =  dtype(ta)
    epar = dtype(epar)

    # construct spectrum for both L-model (opening phase) and return phase
    w = (f*2*np.pi).astype(dtype, copy=False)

    E0_2    = dtype(-0.5*Ee/(np.exp(alpha*te)*np.sin(wg*te)))
    logE0_2 = np.log(E0_2)

    # Original result
    # spec1 = E0*(1/2*i*(exp(alpha*te+i*wg*te-i*w*te)-1)./(-alpha-i*wg+i*w)-1/2*i*(exp(alpha*te-i*wg*te-i*w*te)-1)./(-alpha+i*wg+i*w));
    # changed to reduce vector arithmetic especially vector
    # arithmetic with complex numbers
    #
    #spec1o = ((exp((alpha*te+logE0_2)+i*te*(wg-w))-E0_2)./(i*alpha+(w-wg))...
    #         -(exp((alpha*te+logE0_2)-i*te*(wg+w))-E0_2)./(i*alpha+(wg+w)));
    #
    # new version avoiding some exponential functions in the related C version
    expalphatel=dtype(np.exp((alpha*te+logE0_2)))
    # for some pairs of oq am wg may be equal to an individual frequency point, if this is the case we add
    # a small offset to the potentially zero denominator below
    eps = np.finfo(dtype).eps
    if np.abs(alpha) < eps and np.min(np.abs(w-wg)) < eps:
        wg_eps = eps
    else:
        wg_eps = 0

    spec1 =  ((expalphatel * _exp_imag(te * (wg - w)) - E0_2) /  (1j * alpha + (w - wg + wg_eps))
              - (expalphatel * _exp_imag(-te * (w +wg)) - E0_2) /  (1j * alpha + (w+wg )))

    # if(max(abs(spec1-spec1o))> eps)
    #   figure (555)
    #   subplot(211)
    #   plot(Flp(spec1-spec1b))
    #   subplot(212)
    #   plot(angle(spec1-spec1b))
    # end
    spec = spec1
    
    if ta==0:
        spec2 = dtype(0)
    else:
        bb      = np.flatnonzero(w>np.finfo(w.dtype).eps)
        if epar > 0:
            # again, the original function
            # spec2 = -Ee/epar*(exp(epar*te/ta)*(-exp(-epar/ta-i*w)+exp(-epar*te/ta-i*w*te))./(epar/ta+i*w)-i*exp(epar/ta*(-1+te))*(exp(-i*w)-exp(-i*w*te))./w);

            # that results from the fourier
            # transform is replaced by a transformed version that
            # centralizes as much as possible the calls to exp(w) and
            # separates scalar and vectorial arithmetic as much as possible
            expte1ta= np.exp(epar*(te-1)/ta)
            hh      = np.ones(w.shape, dtype=dtype) * (-1j*(te-1))
            Efte    = _exp_imag(-te*w)
            hh[bb]  = (Efte[bb]-_exp_imag(-w[bb]))/w[bb];
            # use limit according to hopital for  w->0, better do not
            # replace that by a simple spec(0) = 0 because the version here
            # allows us to test whether the spectrum is  zero for w=0.
            spec2   = ((Ee*ta*(1-expte1ta))*Efte +(1j*Ee*epar*expte1ta)*hh)/(w*(1j*ta*(expte1ta-1))+epar*(expte1ta-1));
        else:
            # for epar == 0 (ta ~= 1-oq) we simplify the glottal pulse return phase
            # by the line connecting -Ee to 0 over the time segment from oq to oq + ta = T
            #
            # We solve the Fourier integral in sympy
            # import sympy as sy
            # t, ta, w =sy.symbols("t, ta, w", real=True, positive=True)
            # sy.simplify(sy.integrate((t-ta)/ta*sy.exp(-1j*w*t), [t, 0, ta]))
            #
            # to get the result
            # -> 1.0*I/w - 1.0/(ta*w**2) + 1.0*exp(-1.0*I*ta*w)/(ta*w**2)
            #
            # For the point w = 0 this result is undefined but we can simply
            # calculate the integral for the special case w= 0
            # sy.simplify(sy.integrate((t-ta)/ta, [t, 0, ta]))
            # -> -ta/2
            #
            # finally we take into account that the function is displaced to t=oq
            # by means of adding a spectral domain delay.
            spec2     = Ee * ta*0.5*np.ones(w.shape, dtype=dtype) + ta*0j
            spec2[bb] = Ee *(1j*ta*w[bb] - 1 + np.exp(-1j*w[bb]*ta))/(ta*w[bb]**2)
            spec2    *= np.exp(-1j*oq*w)
            
        spec    = spec + spec2


    if get_derivative:
        if(w[0] == 0) :
            spec[0] = 0;
    else:
        if(w[0] != 0) :
            spec /= (1j*w)
        else:
            spec[1:] /= (1j*w[1:])

            E0 = -Ee/(np.exp(alpha*oq)*np.sin(wg*oq));
            openingside = E0*(-2*alpha*np.exp(alpha*te)*wg*np.cos(wg*te)+alpha**2*np.exp(alpha*te)*np.sin(wg*te)
                                  -wg**2*np.exp(alpha*te)*np.sin(wg*te)
                                  +wg*te*alpha**2+wg**3*te+2*alpha*wg)/(alpha**2+wg**2)**2;

            if( ta > 0) :
                epsilon = epar/ta
                expete  = np.exp(epsilon*(-1+te))
                # original version kept for reference
                # closingside = -1/2*Ee*(2*expete+expete*epsilon^2+2*expete*epsilon-2+expete*epsilon^2*te^2-2*expete*epsilon*te-2*expete*epsilon^2*te)/epsilon^3/ta;
                #
                # compact version of the equation above
                closingside=-1/2*Ee*ta**2*(expete*(2+epsilon**2+2*epsilon
                                                       +(epsilon*te)**2-2*epsilon*te-2*epsilon**2*te)-2)/(epar**3)
            else:
                closingside = 0
        
            spec[0] = openingside+closingside
        
    # apply time shift
    if abs(orig)>0:
        spec = spec* _exp_imag(w* dtype(orig))

    return spec, spec1, spec2, alpha, epar, ta

