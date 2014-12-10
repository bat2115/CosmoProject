import numpy as np
import matplotlib.pyplot as plt 
from pylab import *
from scipy import integrate
from scipy import misc
from numpy.linalg import inv
import datetime




#Upload SN data
#columns: name, z, magnitudes, errors
data = np.genfromtxt(fname='SNdata.txt')

z = data[:,1] #SN redshifts
DM_data = data[:,2] #SN measured distance modulus
error = data[:,3] #SN errors

M = -19.3146267582 #absolute magnitude of the SN standard candle (h=0.7, with systematics)
mag_data = DM_data + M #SN measured apparent magnitudes

c = 3*10**5 #km/s       #Definition of c
scriptM = M - 5.0*log10(70.0 / c) + 25.0      #Definition of script M (H is normalized)

omegaM = 0.3      #Omega matter
omegaDE = 0.7
w = -1.0          #Equation of State




#Defining a function to calculate the magnitudes of each SNe as a function of omega_M, w, and script_M:
def mag(omega_M, omega_DE, script_M):
    mag = zeros_like(z)
    #Loop over all SNe:
    for j in range(len(z)):
        #beginning = datetime.datetime.now()
        integrand = lambda x: 1.0 / np.sqrt( (omega_M*(1.0+x)**3.0) + (omega_DE*((1.0+x)**(3.0*(1.0+w)))) )
        H0comovD = integrate.quad( integrand, 0, z[j] )[0]
        #end = datetime.datetime.now()
        #print end-beginning
        #Calculate the luminosity distance:
        H0dL = (1.0 + z[j]) * H0comovD
        #Calculate the magnitude:
        mag[j] = 5.0 * log10( H0dL ) + script_M
    return mag

#m_array = arange(35, 45, 0.1)
#for i in range(len(m_array)):
#    plt.plot(m_array[i], mag(omegaM, omegaDE, m_array[i])[0], 'k.')
#    plt.plot(m_array[i], mag(omegaM, omegaDE, m_array[i])[5], 'r.')
#    plt.plot(m_array[i], mag(omegaM, omegaDE, m_array[i])[230], 'g.')
#    plt.plot(m_array[i], mag(omegaM, omegaDE, m_array[i])[70], 'b.')
#plt.show()

#for i in range(len(m_array)):
#    Likelihood_ = np.exp(-(np.sum( ( (mag_data - mag(omegaM, omegaDE, m_array[i])) / error )**2.0 )) / 2.0)
#    plt.plot(m_array[i], Likelihood_, 'k.')
#    print Likelihood_
#plt.show()


#Varying both omega_M and omega_DE, marginalized over Mscript
omegaM_array = np.arange(0.0, 0.5, 0.01) #dx = 0.0002
omegaDE_array = np.arange(0.2, 1.1, 0.02) #dx = 0.0003

def Likelihood(omega_M, omega_DE, script_M):
    integrand = lambda script__M: np.exp(-(np.sum( ( (mag_data - mag(omega_M, omega_DE, script__M)) / error )**2.0 )) / 2.0)
    beginning = datetime.datetime.now()
    L = integrate.quad(integrand, script_M-40.0, script_M+40.0)[0]
    end = datetime.datetime.now()
    print end-beginning
    print L
    return L

scriptL = np.zeros( (len(omegaDE_array), len(omegaM_array)) )
print np.shape(scriptL), np.shape(omegaM_array), np.shape(omegaDE_array)
for i in range(len(omegaDE_array)):
    print "row", i, "of", len(omegaDE_array)
    for j in range(len(omegaM_array)):
        scriptL[i, j] = Likelihood(omegaM_array[j], omegaDE_array[i], scriptM)

print omegaM_array[np.where(scriptL == np.amax(scriptL))[1]], omegaDE_array[np.where(scriptL == np.amax(scriptL))[0]]

np.savetxt('L_array_Case1.txt', scriptL)

#plt.figure(0)
#x, y = meshgrid(omegaM_array, omegaDE_array)
#print shape(x), shape(y), shape(scriptL)
#plt.contour(x, y, scriptL)#, [1.0, 1.0/4.0] )


plt.imshow(scriptL,origin='lower',interpolation='none',extent=(np.min(omegaM_array),np.max(omegaM_array),np.min(omegaDE_array),np.max(omegaDE_array)),aspect='auto')


font = {'family' : 'normal',
    'weight' : 'normal',
    'size'   : 26}
matplotlib.rc('font', **font)


plt.show()
