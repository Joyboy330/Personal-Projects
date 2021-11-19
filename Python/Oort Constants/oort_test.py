'''
This code reads in a Gaia .fits file which must include:
Galactic Coordinates (l,b,Parallax), proper motions (mu_ra, mu_dec), and radial velocities

then outputs the oort constants + their log-likelihood value for the entire sample.
Mostly analagous to the radial bin code.
'''
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pp
import galpy.potential as pot
import astropy.coordinates as coord
import astropy.units as u
import scipy.interpolate as interpol
import scipy.optimize as opt
import astropy.io.fits as fits
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as colors
import astropy
import multiprocessing
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
from astropy.units import Quantity

out = multiprocessing.Queue()
 
def Q(g,g_err,dat):
	"""
	An intermediate method for queueing the data for each multiprocessing process
	"""
	while len(dat) != 0:
		LogLike(g,g_err,dat.pop())

def LogLike(G, G_err, params):
    '''
    Log-Likelihood function from Bovy(2017)

    G is an astropy coordinate frame object in galactic coordinates with all of the Gaia 6D
    phase space information.

    G_err is an analagous object that propagates the uncertainties in the kinematic
    and positional data
    '''
    assert type(G) == (astropy.coordinates.builtin_frames.galactic.Galactic)
    
    A = params[0]
    B = params[1]
    C = params[2]
    K = params[3]
    u_0 = params[4]
    v_0 = params[5]
    w_0 = params[6]
    sig_l = params[7]
    sig_b = params[8]


    #Conversion factor from mas/yr to km/(s*kpc)
    conv = 4.74047

    ll = (((G.pm_l_cosb*(u.yr/u.mas)*conv - mu_l(G.l/u.deg, G.b/u.deg, G.distance/u.pc, A, B, C, u_0, v_0)) ** 2 /
           ((G_err.pm_l_cosb*(u.yr/u.mas)*conv) ** 2 + sig_l ** 2)) +
          np.log((G_err.pm_l_cosb*(u.yr/u.mas)*conv) ** 2 + sig_l ** 2) +
          ((G.pm_b*(u.yr/u.mas)*conv - mu_b(G.l/u.deg, G.b/u.deg, G.distance/u.pc, A, C, K, u_0, v_0, w_0)) ** 2 / ((G_err.pm_b*(u.yr/u.mas)*conv) ** 2 + sig_b ** 2)) +
          np.log((G_err.pm_b*(u.yr/u.mas)*conv) ** 2 + sig_b ** 2))
	
    #Returning the output to a multiprocessing queue for retrieval later
    out.put((-0.5 * np.nansum(ll, axis=0),params))
    return (-0.5 * np.nansum(ll, axis=0),params)


def mu_l(l, b, d, A, B, C, u_0, v_0):
    d = d/1000 #in kpc
    l = l * (np.pi / 180)*u.rad
    b = b * (np.pi / 180)*u.rad
    return (A * np.cos(2 * l) - C * np.sin(2 * l) + B) * np.cos(b) + d**-1 * (u_0 * np.sin(l) - v_0 * np.cos(l))


def mu_b(l, b, d, A, C, K, u_0, v_0, w_0):
    d = d/1000 #in kpc
    l = l * (np.pi / 180)*u.rad
    b = b * (np.pi / 180)*u.rad
    return -(A * np.sin(2 * l) + C * np.cos(2 * l) + K) * np.sin(b) * np.cos(b) + d**-1 * (
            (u_0 * np.cos(l) + v_0 * np.sin(l)) * np.sin(b) - w_0 * np.cos(b))
#
# Set the filename here 
#  vvvvvvvvvvvvvvvvvv
#  
#   
#
fn = '/data/dsw59/EilersDat.fits'
dat = fits.open(fn,memap=False)
locA = dat[1].data
#
# ^^^^^^^^^^^^^^^^^^^^
#
#
#
#

#Doing the Coordinate transforms for the stellar velocities and their errors

star_loc_GAL = coord.Galactic(l = locA['l']*u.deg, b = locA['b']*u.deg,
                        distance = 1/(locA['parallax']/1000)*u.pc)

star_loc_EQU = star_loc_GAL.transform_to(coord.ICRS)

star_loc_EQU = coord.ICRS(ra = star_loc_EQU.ra, dec = star_loc_EQU.dec, distance = star_loc_EQU.distance,
                         pm_ra = locA['pmra']*(u.mas/u.yr), pm_dec = locA['pmdec']*(u.mas/u.yr),
                         radial_velocity =locA['radial_velocity']*(u.km/u.s),
                         differential_type=coord.SphericalDifferential)

#Creating a cylinder with radius 2000pc
want = np.where(((1/(locA['parallax']/1000)*np.cos(locA['b']*(np.pi/180))) < 2000)
                & (np.abs(1/(locA['parallax']/1000)*np.sin(locA['b']*(np.pi/180))) < 400)) #Limiting distance from disk plane

star_loc_GCN = star_loc_EQU.transform_to(coord.Galactocentric)[want]

Galactic_coords = star_loc_EQU.transform_to(coord.Galactic)[want]

#Now the errors

star_loc_GAL_err = coord.Galactic(l = locA['l'][want]*u.deg, b = locA['b'][want]*u.deg,
                        distance = 1/(locA['parallax_error'][want]/1000)*u.pc)

star_loc_EQU_err = star_loc_GAL_err.transform_to(coord.ICRS)

star_loc_EQU_err = coord.ICRS(ra = star_loc_EQU_err.ra, dec = star_loc_EQU_err.dec, distance = star_loc_EQU_err.distance,
                         pm_ra = locA['pmra_error'][want]*(u.mas/u.yr), pm_dec = locA['pmdec_error'][want]*(u.mas/u.yr),
                         radial_velocity =locA['radial_velocity_error'][want]*(u.km/u.s),
                         differential_type=coord.SphericalDifferential)

star_loc_GCN_err = star_loc_EQU_err.transform_to(coord.Galactocentric)

Galactic_coords_err = star_loc_EQU_err.transform_to(coord.Galactic)


#Parallelization routine below here

#Set the parameters we want to vary

num_nodes = multiprocessing.cpu_count()-3 #Number of cpu cores to use
proc_per_core = 15625                     #Number of jobs per core; higher is better for convergence, worse for runtime + memory
n = num_nodes*proc_per_core               #The fineness of the grid
dim = n // num_nodes
n = dim * num_nodes
param_set = []
jobs = []


for i in range(n):
    # A,B,C,K in km/(s*kpc)
    A = np.random.random()*(18-12)+10
    B = np.random.random()*(-15+9)-9
    C = np.random.random()*(10)+0
    K = np.random.random()*(10)+0

    # u,v,w in km/s
    u_0 = np.random.random()*(15-5)+5
    v_0 = np.random.random()*(10)+0
    w_0 = np.random.random()*(15-5)+5

    # proper motions in km/(s*kpc)
    sig_ul = 1
    sig_ub = 1

    param_set.append(np.array([A, B, C, K, u_0, v_0, w_0, sig_ul, sig_ub]))



start = time.time()
ans = []


while len(param_set) != 0:
	#for i in range(num_nodes):
	temp = []
	for d in range(proc_per_core):

		temp.append(param_set.pop())

	t = multiprocessing.Process(target=Q,args=(Galactic_coords,Galactic_coords_err,temp,))
	jobs.append(t)

	while True:

		try:

			t.start()

		except OSError:

			continue

		break 

for i in range(n):

	ans.append(out.get())

end = time.time()
elapsed_min = (end - start)/60


print("The runtime was ", np.round(elapsed_min,2)," minutes.")

sorted = np.argsort([i[0] for i in ans])

filename = 'oortparams'

for i in time.localtime()[1:5]:

    filename += str(i).zfill(2)

filename += '.dat'

with open(filename,'w') as f:

    f.write('# File Used: '+fn+'\n')
    f.write('# (Log Likelihood, [A,B,C,K,u_0,v_0,w_0,sig_ul,sig_ub])\n')

    for i in sorted:

        f.write(str(ans[i])+'\n')
