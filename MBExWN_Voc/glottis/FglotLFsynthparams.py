# AUTHOR:  G.Degottex/A.Roebel
# COPYRIGHT: Copyright(c) 2008 - 2022 IRCAM/CNRS - Degottex/Roebel
import numpy as np
import opt.fzero as fz
import scipy.optimize as sopt

#@profile
def FglotLFsynthparams(oq, am, ta, old=False) :
    '''% Solve alpha and epar parameters for LF glottis model
    %
    % USAGE
    %  function [alpha, epar] = FglotLFsynthparams(oq, am, ta)
    %
    % INPUT PARAMETERS
    %
    % oq : open phase coefficient in terms of the fundamental period
    %      in terms of the LF model parameters te
    %      is given by te/T0 with T0 being the fundamental period
    %      in ]0;1[
    %
    % am : asymmetry coefficient expressed as coefficient relative to the
    %      open phase of the glottal puls
    %      in [0.5;1[
    %
    % ta:  effective closing time in terms of the fundamental period
    %      in [0;1-oq[
    %
    % OUTPUT PARAMETERS
    %
    %    alpha	: synthesis parameter following LF definition
    %    epar	: synthesis parameter following LF definition
    %             (NOT epsilon in glotspecLFori ! epar=epsilon*ta)
    %    ta     : suggested ta parameter avoiding instabilities during integration of 
    %             pulse equations.
    %
    % AUTHORS
    %  G. Degottex 2008-2012 (matlab version)
    %  A. Roebel   2013-2022 (python version)
    %
    % COPYRIGHT
    %  Copyright (c) 2008-2022  IRCAM/CNRS - Degottex, Roebel
    %
    % $Id: FglotLFsynthparams.m,v 1.2 2009/03/23 15:46:29 degottex Exp $
    '''

    target_type_realization = oq+am+ta;
    if(oq <= np.finfo(target_type_realization).eps or oq >=(1-np.finfo(target_type_realization).eps)) :
        raise RuntimeError('open quotient out of range')
    if(am < 0.5 or am >=(1-np.finfo(target_type_realization).eps)):
        raise RuntimeError('asymetry is out of range'); 
    if(ta < 0 or ta >(1-oq)):
        raise RuntimeError('return phase length(ta) is out of range');

    # time parameters
    # tc = 1;	% implicitly, and so for a normalized period ..
    te = oq;
    tp = am*oq;

    # solve synthesis parameters	wg, epar, alpha, E0
    wg = np.pi/(oq*am);

    # precalculate some constants
    cos_wgte =np.cos(wg*te)
    wg_cos_wgte = wg * cos_wgte
    sin_wgte =np.sin(wg*te)
    wgh2 = wg**2

    if ta<= np.finfo(np.float32).eps :

        if old:
            # in case of abrupt closure
            def afun(a):
                return np.exp(a*oq)*(wg*np.cos(wg*oq)-a*np.sin(wg*oq))-wg
            (alpha, tmp, tmp, tmp) = fz.fzero(afun, 0.)
        else:
            def eq_alpha(a) :
                return np.exp(a*oq)*(wg_cos_wgte-a*sin_wgte)-wg

            alphal  = 0
            alphar  = 0.1
            valpha0 = eq_alpha(0)
            if(np.abs(valpha0) >np.finfo(oq).eps):
                while( (valpha0 * eq_alpha(alphar) > 0) and (valpha0 * eq_alpha(-alphar) > 0)) :
                    alphal  = alphar
                    alphar += 1
                if(eq_alpha(-alphar)*valpha0<0):
                    alphal = -alphal
                    alphar = -alphar
            else:
                alphal = -0.1
                alphar = 0.1

            alpha = sopt.brentq(eq_alpha, alphal, alphar)

            if(alpha>np.fmax(alphar,alphal)):
                raise RuntimeError("glotspecLFC: GetLFModelAlpha: alpha estimate did not converge.\n")
        epar = 0
        ta = 0
    else:
        
        # we search a parameterisation of the function
        # E_2(t) in eq 11 of Fant 4 parameter model of the glottal flow, 1985
        #
        # E_2(t) = -Ee/(\epsilon ta)*(exp(-\epsilon*(t-te)) - exp(-\epsilon*(T-te))
        # using a transformed function with
        # E_2(t) = -Ee/epar*(exp(-epar*(t-te)/ta) - exp(-epar*(T-te)/ta)
        # ensuring the following limits
        # E_2(0) = -Ee, E_2(1) = 0! 
        #
        # Due to the difference in the second product the second limit is independent of epar
        # proper selection of epar is required to ensure the E_2(0)
        
        if oq >0.999 :
            # if 1-oq is very small the change of ta does hardly change the overall puls form
            # and we can skip the complex (and in that case probably unstable optimisation)
            # all together and force ta and epar to intermediate values
            epar = 0.5
            ta   = 0.5*(1-oq)
        elif ta > 0.99*(1-oq) :
            # here the return phase is cut very early approaching a line  that we can simply
            # approximate by a linear function
            epar = 0
            ta   = 1-oq
        else:
            # for the parameter ranges 0< ta < 1-te
            # the function efun(epar) has a single minimum with value <= zero
            # that is located in the range 0<=epar<=1  and can calculated directly from the derivative
            # 1 - (1-te)/ta *exp(-epar/ta*(1-te)) = 0
            # ->
            # epar_min = -(1-te)/ta*np.log(ta/(1-te))
            #
            # we can start the minimization using fzero limitting the range
            # to [epar_min, 1.1]
            te_m_1_d_ta = (te-1)/ta
            if old:
                def efun(epar) :
                    return epar-1+ np.exp(-epar/ta*(1-te));
                eleft  = - np.log(- te_m_1_d_ta)/te_m_1_d_ta
                eright = 1.1
                (epar, tmp, tmp, tmp) = fz.fzero(efun, (eleft,eright))
            else:
                def eq_epar(epar) :
                    return epar - 1 + np.exp(epar*te_m_1_d_ta)
                eleft  = - np.log(- te_m_1_d_ta)/te_m_1_d_ta
                eright = 1.1
                # solve e=epar
                epar = sopt.brentq(eq_epar, eleft, eright)
             
            
        # the original integral using epar
        # E2I2 = -(-exp(epar*(te-1)/ta)-exp(epar*(te-1)/ta)*epar/ta+exp(epar*(te-1)/ta)*epar/ta*te+1)/(epsilon^2/ta)
        
        # a transformed version using a number of arithmetic
        # transformations to reduce the number of exp functions
        if epar == 0:
            E2I = -ta/2
        else:
            E2I = (-np.exp(epar/ta*(te-1))*(ta+epar-te*epar)+ta) /(epar*(-1+np.exp(epar/ta*(te-1))));


        if old:
            def afun(a) :
                return -(-wg*np.cos(wg*te)+a*np.sin(wg*te)+wg*np.exp(-a*te))/(a**2+wg**2)/np.sin(wg*te) + E2I
            (alpha, tmp, tmp, tmp) = fz.fzero(afun, 0.)
        else:
            def eq_alpha(a) :
                return -(-wg_cos_wgte+a*sin_wgte+wg*np.exp(-a*te))/(a**2+wgh2)/sin_wgte + E2I

            alphal  = 0
            alphar  = 0.1
            valpha0 = eq_alpha(0)
            if(np.abs(valpha0) >np.finfo(oq).eps) :
                while( (valpha0 * eq_alpha(alphar) > 0)
                           and (valpha0 * eq_alpha(-alphar) > 0)) :
                    alphal = alphar
                    alphar += 1
                if(eq_alpha(-alphar)*valpha0<0):
                    alphal = -alphal
                    alphar = -alphar
            else:
                alphal = -0.1
                alphar = 0.1
            alpha = sopt.brentq(eq_alpha, alphal, alphar)
            if(alpha>np.fmax(alphar,alphal)):
                raise RuntimeError("glotspecLFC: GetLFModelAlpha: alpha estimate did not converge with ta={0:f} .".format(ta))
        
    return alpha, epar, ta
