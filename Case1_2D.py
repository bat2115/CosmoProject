from numpy import *
import matplotlib.pyplot as plt 
from pylab import *
from scipy import integrate
from scipy import misc
from numpy.linalg import inv




#Upload SN data
#columns: name, z, magnitudes, errors
data = genfromtxt(fname='SNdata.txt')

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
        integrand = lambda x: 1.0 / np.sqrt( (omega_M*(1.0+x)**3.0) + (omega_DE*((1.0+x)**(3.0*(1.0+w)))) )
        H0comovD = integrate.quad( integrand, 0, z[j] )[0]
        #Calculate the luminosity distance:
        H0dL = (1.0 + z[j]) * H0comovD
        #Calculate the magnitude:
        mag[j] = 5.0 * log10( H0dL ) + script_M
    return mag




#Varying both omega_M and omega_DE, marginalized over Mscript

omegaM_array = arange(0.25, 0.28, 0.0005) #dx = 0.0002
omegaDE_array = arange(0.71, 0.74, 0.0005) #dx = 0.0003

def Likelihood(omega_M, omega_DE, script_M):
    chisqd = sum( ( (mag_data - mag(omega_M, omega_DE, script_M)) / error )**2.0 )
    integrand = lambda script_M: exp(-chisqd / 2.0)
    L = integrate.quad(integrand, script_M-5.0, script_M+5.0)[0]
    return L

scriptL = zeros( (len(omegaM_array), len(omegaDE_array)) )
print shape(scriptL), shape(omegaM_array), shape(omegaDE_array)
for i in range(len(omegaM_array)):
    for j in range(len(omegaDE_array)):
        scriptL[i,j] = Likelihood(omegaM_array[i], omegaDE_array[j], scriptM)

print omegaM_array[where(scriptL == amax(scriptL))[0]], omegaDE_array[where(scriptL == amax(scriptL))[1]]



plt.figure(0)
x, y = meshgrid(omegaDE_array, omegaM_array)
print shape(x), shape(y), shape(scriptL)
plt.contour(x, y, scriptL) #, [1.0, 1.0/4.0] )





font = {'family' : 'normal',
    'weight' : 'normal',
    'size'   : 26}
matplotlib.rc('font', **font)


plt.show()
