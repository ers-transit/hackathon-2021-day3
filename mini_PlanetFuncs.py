"""
Simple transit model without limb darkening taken from PlanetFuncs
https://github.com/nealegibson/PlanetFuncs

"""

import numpy as np

def transit(par,t):
  """
  Simple transit model for cicular orbit and no limb darkening.
  
  par is a parameter vector:
    par = [T0,P,aRs,rho,b,foot,Tgrad]
    where:
    T0 - central transit time
    P - orbital period
    aRs - semi-major axis in units of stellar radius (system scale) or a/R_star
    rho - planet-to-star radius ratio - the parameter of interest for trasmission spectroscopy
    b - impact parameter
    foot - out-of-transit flux
    Tgrad - linear gradient of basis function
  t is time (in same units as T0 and P)
  
  """

  T0,P,a_Rstar,p,b,foot,Tgrad = par
    
  #calculate phase angle
  theta = (2*np.pi/P) * (t - T0)
  
  #normalised separation z
  z = np.sqrt( (a_Rstar*np.sin(theta))**2 + (b*np.cos(theta))**2 );

  #calculate flux
  f = np.ones(t.size) #out of transit
  f[z<=1-p] = 1-p**2 #fully in transit
  ind = (z > np.abs(1-p)) * (z<=1+p) #and finally for ingress/egress
  k0 = np.arccos((p**2+z[ind]**2-1)/2/p/z[ind]) # kappa0
  k1 = np.arccos((1-p**2+z[ind]**2)/2/z[ind]) # kappa1
  f[ind] = 1-(k0*p**2 + k1 - np.sqrt( 0.25*(4*z[ind]**2-(1+z[ind]**2-p**2)**2) ))/np.pi
  
  #modify according to linear basis function
  f = f * (foot + (t - T0) * Tgrad)
    
  #return flux
  return f
