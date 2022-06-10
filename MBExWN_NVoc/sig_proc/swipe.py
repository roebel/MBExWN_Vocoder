from __future__ import absolute_import, print_function, division

import sys
import numpy as np

from matplotlib import mlab

from scipy.interpolate import interp1d


def swipe(x,fs,plim=np.array([30., 5000.]),
            dt=0.01, sTHR=-np.Inf, freq_limit=12500,
            spline_interp=False, music_mode=False,
              verbose=False, ana_zero_frames=True) :
    """
    run f0 analysis using the swipe algorithm 

    Based on Camacho, Arturo. A sawtooth waveform inspired pitch estimator for
    speech and music. Doctoral dissertation, University of Florida. 2007.

    Inputs:
    ============
    x   : signal waveform (1D-array)
    fs  : samplerate in Hz (scalar)
    plim: 1d array of size 2 specifying the pitch range in Hz  (def: np.array([30, 5000]))
    dt  : time step in seconds (def: 0.01)
    sTHR: threshold for picth salience below the given value the 
          results will be discared (def: -Inf)
    freq_limit : frequency limit to be used for spectral analysis in Hz (def: 12500)
    use_spline_interp: interpolate spectral magnitude using 3 order splines (def: use zero padding 
                          with linear interpolation)
                       Note that the original swipe algorithm used splines, but these are rather costly
                       to produce especially for long windows (dense frequency sampling)
                       without producing a significant advantage
    music_mode: Use linear frequency grid and constant spectral weight which according to 
                 Camacho, JASA 2008, is prefereable for music (Def: speech)

    ana_zero_frames : Do not apply pitch strength calculation to STFT frames that contain only zeros,
                      settings this to False reduces slightly the amount of computation that is 
                      required but adds some overhead (Def: True)


    Outputs: tuple containing
    ====================
    p: pitch sequence in Hz
    t: time positions in sec
    s: score for each pitch 
    """


    dlog2p = 1/96
    dERBs = 0.1 
    dHzs  = 5 

    # Times    
    t = np.arange(0,  len(x)/fs, dt)
    # Hop size (in cycles)
    dc = 4
    # Parameter k for Hann window    
    K = 2
    # Define pitch candidates
    log2pc = np.arange(np.log2(plim[0]), np.log2(plim[-1]), dlog2p)
    pc = 2.**log2pc
    # Pitch strength matrix    
    S = np.zeros( (len(pc), len(t) ))
    
    # Determine P2-WSs
    logWs = np.array(np.round(np.log2( 4*K * fs / plim ) ), dtype=np.int)
    # P2-WSs
    ws = 2**np.arange( logWs[0], logWs[1], -1)
    
    # Optimal pitches for P2-WSs
    pO = 4*K * fs / ws
    # Determine window sizes used by each pitch candidate
    # d will be used for indexing eso we adapt the start value by subtracting 1
    # compared to trhe matlab version
    d = log2pc - np.log2( 4 * K * fs / ws[0] )
    # Create ERBs spaced frequencies (in Hertz)
    if music_mode:
        fSamps = np.arange(pc[0]/4, np.fmin(fs/2, freq_limit), dHzs)
    else:
        fSamps = erbs2hz(np.arange(hz2erbs(pc[0]/4), hz2erbs(np.fmin(fs/2, freq_limit)), dERBs))

    #import time
    #sumX = 0
    #sumP = 0
    #sumI = 0
    stft_obj = None
    for i in np.arange( len(ws), dtype=np.int) :
        # Hop size (in samples)        
        dn = np.int(np.round( dc * fs / pO[i] ))
        if verbose:
            print("ws {0:.0f} step {1:.0f}".format(ws[i]*fs, dn*fs), file=sys.stderr)
        # Zero pad signal
        xzp = np.concatenate((np.zeros( ws[i]//2 ), x, np.zeros(dn + ws[i]//2)))
        # Compute spectrum
        # Hann window (matlab variante not including the 0s at the boundaries        
        w = np.hanning( ws[i]+2 )[1:-1]
        # Window overlap        
        o = np.fmax( 0, np.round( ws[i] - dn ) )

        #startX = time.process_time()
        #[ X, f, ti ] = specgram( xzp, ws(i), fs, w, o );
        if spline_interp:
            X, f, ti = mlab.specgram(xzp, NFFT=ws[i], window=w, Fs=fs, noverlap=o,
                                        mode='magnitude')

            ind = np.flatnonzero(f <= freq_limit)
            f = f[ind]
            X=X[ind, :]
            # simulate matlab ti
            ti-=ti[0]
            # Interpolate at equidistant ERBs steps
            fi = interp1d(f, X.T, bounds_error=False, fill_value=0,
                              kind="cubic", assume_sorted=True )
        else:
            # do interpolation by means of zero padding and
            # with subsequent linear interpolation to avoid excessive runtime
            # of the cubic interpolator
            X, f, ti = mlab.specgram(xzp, pad_to=ws[i]*4,
                                        NFFT=ws[i], window=w, Fs=fs, noverlap=o,
                                        mode='magnitude')

            ind = np.flatnonzero(f <= freq_limit)
            f = f[ind]
            X=X[ind, :]
            # simulate matlab ti
            ti-=ti[0]
            # Interpolate at equidistant ERBs steps
            fi = interp1d(f, X.T, bounds_error=False, fill_value=0,
                              kind="linear", assume_sorted=True )
                    
        # Magnitude in erb grid
        M  = np.fmax( 0, fi(fSamps).T)
        # Loudness            
        L = np.sqrt( M );

        if not ana_zero_frames:
            Lnz_bb = np.sum(L, axis=0)>0
            Lnz = L[:, Lnz_bb]
        #startP = time.process_time() 
        #sumX += startP - startX

        # Select candidates that use this window size
        if i==len(ws)-1:
            j = np.flatnonzero(d - i > -1)  
            k = np.flatnonzero(d[j] - i < 0)
        elif i==0:
            j = np.flatnonzero(d - i < 1)
            k = np.flatnonzero(d[j] - i > 0)
        else :
            j = np.flatnonzero(np.abs(d - i) < 1)
            k = np.arange(len(j)) 

        # create pitch salience pc[j].shape x L.shape[1]
        if ana_zero_frames:
            Si_ = pitchStrengthAllCandidates( fSamps, L, pc[j], music_mode)
        else:
            Si_nz = pitchStrengthAllCandidates( fSamps, Lnz, pc[j], music_mode)
            Si_ = np.zeros((Si_nz.shape[0], L.shape[1]), dtype=Si_nz.dtype)
            Si_[:, Lnz_bb] = Si_nz

        #startI = time.process_time() 
        #sumP += startI - startP
            
        # Interpolate at desired times
        if Si_.shape[1] > 1:
            fi = interp1d(ti, Si_, bounds_error=False, fill_value=np.nan,
                              kind="linear", assume_sorted=True )
                # Magnitude in erb grid
            Si  = fi(t)
        else:
            Si    = np.zeros((Si_.shape[0], t.shape[0]))
            Si[:] = np.nan

        #startE = time.process_time() 
        #sumI += startE - startI
            
        lam = d[j[k]] - i;
        mu  = np.ones( (j.shape[0], 1) )
        mu[k,:] = 1 - np.abs( lam )[:,np.newaxis]
        S[j,:] = S[j,:] + mu * Si;

    # Fine-tune the pitch using parabolic interpolation
    #p = repmat( NaN, size(S,2), 1 );
    p = np.zeros((S.shape[1]))
    p[:] = np.nan
    s = p.copy()
    for j in range(S.shape[1]):
        i  = np.argmax( S[:,j] )
        s[j] = S[i,j]
        if s[j] < sTHR:
            continue
        if i== 0 :
            p[j] = pc[0]
        elif i==len(pc) - 1 :
            p[j] = pc[-1] 
        else :
            I = np.arange(i-1, i+2)
            tc = 1 / pc[I]
            ntc = ( tc/tc[1] - 1 ) * 2 * np.pi;
            c   = np.polyfit( ntc, S[I,j], 2 )
            ftc = 2**-np.arange(np.log2(pc[I[0]]),np.log2(pc[I[-1]]), 1/(12*64))
            nftc = ( ftc/tc[1] - 1 ) * 2*np.pi;
            pp = np.polyval( c, nftc )
            k = np.argmax(pp)
            s[j] = pp[k]  
            p[j] = 2 ** ( np.log2(pc[I[0]]) + (k-1)/(12*64) )

    #sumE = time.process_time() - startE
    #print("time: {0} {1} {2} {3}".format(sumX, sumP, sumI, sumE))sr
    return p, t, s

def pitchStrengthAllCandidates( f, L, pc, music_mode ) :
    # Normalize loudness
    #warning off MATLAB:divideByZero
    #L = L ./ repmat( sqrt( sum(L.*L) ), size(L,1), 1 );
    LS = np.sqrt( np.sum(L*L, axis=0) )
    L = L / np.fmax(LS, np.finfo(LS.dtype).eps)[np.newaxis,:]
    #warning on MATLAB:divideByZero
    # Create pitch salience matrix
    S = np.zeros( (len(pc), L.shape[1] ))
    for j in np.arange(len(pc)):
        S[j,:] = pitchStrengthOneCandidate( f, L, pc[j], music_mode)
    return S

def primesfrom2to(n) :
    """ 
    # http://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
    Input n>=6, Returns a array of primes, 2 <= p < n 
    """
    sieve = np.ones(np.int(n)//3 + (np.int(n)%6==2), dtype=np.bool)
    sieve[0] = False
    for i in range(int(n**0.5)//3+1):
        if sieve[i]:
            k=3*i+1|1
            sieve[      ((k*k)//3)      ::2*k] = False
            sieve[(k*k+4*k-2*k*(i&1))//3::2*k] = False
    return np.r_[2,3,((3*np.nonzero(sieve)[0]+1)|1)]

def pitchStrengthOneCandidate( f, L, pc, music_mode ) :

    # Number of harmonics
    n = np.fix( f[-1]/pc - 0.75 )
    # Kernel
    k = np.zeros( f.shape ) 
    # Normalize frequency w.r.t. candidate
    q = f / pc 
    for i in np.concatenate(([1], primesfrom2to(n+1))):
        a = np.abs( q - i )
        # Peak's weigth
        p = a < .25
        k[p] = np.cos( 2*np.pi * q[p] )
        # Valleys' weights
        v = np.logical_and(.25 < a, a < .75)
        k[v] = k[v] + np.cos( 2*np.pi * q[v] ) / 2

    # Apply envelope, according to camacho for music constant envelop is better
    if not music_mode:
        k = k * np.sqrt( 1. / f );
    # K+-normalize kernel
    k = k / np.linalg.norm( k[k>0] )
    # Compute pitch strength
    #S = k' * L;
    S = np.dot(k[np.newaxis,:], L)
    return S

def hz2erbs(hz) :
    return 21.4 * np.log10( 1 + hz/229. )

def erbs2hz(erbs) :
    return ( 10 ** (erbs/21.4) - 1. ) * 229;
