# AUTHORS
#    A.Roebel
# COPYRIGHT
#    Copyright(c) 2016 - 2022 IRCAM, Roebel
#
# window generator

from __future__ import division, absolute_import

import numpy as np

# from libfft
# Max error acceptable in Izero 
_IzeroEPSILON=1E-21            

def Izero(x) :
    sum = u = n = 1.;
    halfx = (x/2.0);
    while u >= IzeroEPSILON*sum:
      temp  = halfx/n
      n    += 1
      temp *= temp
      u    *= temp
      sum  += u

    return sum


def window(win_type, winlen, para=None) :
    """
    Return libfft/supervp compatible windows as specified by win_type and winlen.

    Supported windows types ::

    hanning:   Hanning window 
    mhanning:  Hanning window including the zeros at the zero borders (Matlab variant of Hanning window)
    hamming:            Hamming window
    exactblackman:
    blackman:           Blackman window
    blackmanharris_3_1: 
    blackmanharris_3_2:
    blackmanharris_4_1:
    blackmanharris_4_2:
    hanning2/nuttall3_30db: Squared Hanning window == 3 coeff Nutall window with 30db sidelobe decay
    nuttall3_18db :  3 param nuttall window with 18db/octave side lobe decay
    nuttall4_6db :   4 param nuttall window with 6db/octave side lobe decay
                     corresponds to scipy.sigfnall.nuttall
    nuttall4_30db :  4 param nuttall window with 30db/octave side lobe decay
                     corresponds to scipy.sigfnall.nuttall    
    nuttall4_42db:   4 param nuttall window with 42db/octave side lobe decay
    triang/bartlett: triangular window   
    gauss: Gaussian window with parameter being std deviation relative to window length
    kaiser: kaiser window ith param beta 
    """

    a1 = 0.0
    a2 = 0.0
    a3 = 0.0
    a4 = 0.0

    win = np.zeros((winlen,))
    mid = (winlen-1)//2
    win_type = win_type.lower()
    
    if win_type == "hanning" or win_type == "hann":
        a1 = 0.5
        a2 = -0.5
    elif win_type == "mhanning":
        return Mhanning(winlen)
    elif win_type == "hamming":
        a1= 0.54
        a2= -0.46
    elif win_type.startswith("rect") or win_type.startswith("box"):
        a1 = 1.
    elif  win_type == "exactblackman":
        a1 = 0.42659 
        a2 = -0.49656 
        a3 = 0.07685
    elif  win_type == "blackman" :
        a1 = 0.42 
        a2 = -0.5 
        a3 = 0.08
    elif  win_type == "blackmanharris_3_1":
        # from matlab function blackmanharris
        a1 = 0.42323 
        a2 = -0.49755 
        a3 = 0.07922
    elif  win_type == "blackmanharris_3_2":
        # from matlab function blackmanharris

        a1 = 0.44959 
        a2 = -0.49364 
        a3 = 0.05677   
    elif  win_type == "blackmanharris_4_1":
        # from matlab function blackmanharris
        a1 = 0.35875 
        a2 = -0.48829 
        a3 = 0.14128
        a4 = -0.01168
    elif  win_type == "blackmanharris_4_2":
        # from matlab function blackmanharris
        a1 = 0.40217 
        a2 = -0.49703 
        a3 = 0.09392 
        a4 = -0.001830   
    elif win_type == "hanning2" or win_type == "nuttall3_30db":
        # Squared hanning window is equivalent to 3param nuttall window
        # with 30db/octave side lobe decay
        a1 = 0.375
        a2 = -0.5
        a3 = 0.125
    elif win_type == "nuttall3_18db":      
        # nuttall window with 18db/octave side lobe decay
        a1 = 0.40897
        a2 = -0.5
        a3 = 0.09103
    elif win_type == "nuttall4_6db":      
        # 4 param nuttall window with 6db/octave side lobe decay
        # corresponds to scipy.sigfnall.nuttall
        a1 = 0.3635819
        a2 = -0.4891775
        a3 = 0.1365995
        a4 = -0.0106411
    elif win_type == "nuttall4_30db":      
        # 4 param nuttall window with 30db/octave side lobe decay
        a1 = 0.338946
        a2 = -0.481973
        a3 = 0.161054
        a4 = -0.018027
    elif win_type == "nuttall4_42db":      
      # 4 param nuttall window with 42db/octave side lobe decay
      a1 = 10./32.
      a2 = -15./32.
      a3 = 6./32.
      a4 = -1./32.

    elif win_type.startswith("triang") or win_type == "bartlett" :   
        rmid = (winlen-1.)/2.
        slope= 1.0/rmid	
        win[winlen-1:winlen-2-mid:-1] = win[:mid+1] = slope*np.arange(mid+1)
        return win
    
    elif win_type == "gauss" :
        
        if not para :
            raise RuntimeError("window::cannot calculate Gauss window without a width parameter")
	
        sigma  = np.float(winlen)/para;      	
        offset = -winlen/2.-0.5
        x=offset+np.arange(winlen);
        win[:] = np.exp(-(x*x)/(2*sigma*sigma))
        return win

    elif win_type == "kaiser" :   

        beta = para
        beta2 = beta*beta
        ibeta = Izero(beta)
        ibeta = 1.0/ibeta
	
        alpha  = (winlen-1.)/2. 
        alpha2 = alpha*alpha

        for  N in np.arange(1, winlen) :

            temp1 = ((N-alpha)/alpha)
            temp4 = 1.-temp1 *temp1
            temp  = np.sqrt(temp4)

            iz = Izero(beta*temp)
            win[N]  = ibeta * iz

        iz = Izero(0.0)
        win[0]=win[winlen-1] = ibeta * iz;
        return win
    else :
        raise RuntimeError("window::unsupported window type {0}".format(win_type))
        
        
    # calculate for all the cosine windows
    Nmax = winlen-1
    rmid = (winlen-1.)/2.
    x = np.arange(mid+1)
    win[winlen-1:winlen-2-mid:-1] = win[:mid+1] =  a1 + a2*np.cos(2.*np.pi*x/Nmax)+ a3*np.cos(4.*np.pi*x/Nmax) +a4*np.cos(6.*np.pi*x/Nmax)  
    return win


def Mhanning(M):
    """
    Return the Hanning window excluding the zeros at the window boundaries
    This window handling is compatible with matlab hanning window
    """

    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1, float)
    n = np.arange(1,M+1)

    return 0.5-0.5*np.cos(2.0*np.pi*n/(M+1))

