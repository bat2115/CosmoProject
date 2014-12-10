from numpy import *
import matplotlib.pyplot as plt 
from pylab import *
from scipy import integrate
from scipy import misc
from numpy.linalg import inv
import scipy.optimize as so




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
def mag(omega_M, w_, script_M):
    mag = zeros_like(z)
    #Loop over all SNe:
    for j in range(len(z)):
        integrand = lambda x: 1.0 / np.sqrt( (omega_M*(1.0+x)**3.0) + ((1.0-omega_M)*((1.0+x)**(3.0*(1.0+w_)))) )
        H0comovD = integrate.quad( integrand, 0, z[j] )[0]
        #Calculate the luminosity distance:
        H0dL = (1.0 + z[j]) * H0comovD
        #Calculate the magnitude:
        mag[j] = 5.0 * log10( H0dL ) + script_M
    return mag




#FISHER MATRIX
#####################################################################################

#Find dm/dp for each parameter (omega_M, w, and script_M) for each SNe
#In other words, find the slope of mag(omega_M, w, and script_M) at
#these values: omega_M = 0.3, w = -1.0, and script_M = const, for 
#each SNe
dm_domegaM  = misc.derivative(lambda x: mag(x, w, scriptM), omegaM, dx=1e-5) 
dm_dw       = misc.derivative(lambda x: mag(omegaM, x, scriptM), w, dx=1e-5)
dm_dscriptM = misc.derivative(lambda x: mag(omegaM, w, x), scriptM, dx=1e-5)

#Define 1/sigma**2, where sigma = errors for each SNe:
one_over_var = 1.0 / (error**2.0)

#Evaluate each element of the Fisher Matrix:
F11 = sum(one_over_var * dm_domegaM * dm_domegaM)
F12 = sum(one_over_var * dm_domegaM * dm_dw)
F13 = sum(one_over_var * dm_domegaM * dm_dscriptM)
F23 = sum(one_over_var * dm_dw * dm_dscriptM)
F22 = sum(one_over_var * dm_dw * dm_dw)
F33 = sum(one_over_var * dm_dscriptM * dm_dscriptM)

#Inputting these elements into a numpy matrix:
FisherMatrix = zeros((3,3))
FisherMatrix[0,0] = F11
FisherMatrix[0,1] = F12
FisherMatrix[0,2] = F13
FisherMatrix[1,0] = F12
FisherMatrix[1,1] = F22
FisherMatrix[1,2] = F23
FisherMatrix[2,0] = F13
FisherMatrix[2,1] = F23
FisherMatrix[2,2] = F33

print "Fisher Matrix:"
print FisherMatrix

#Finding the inverse of the Fisher matrix (covariance matrix):
CovarMatrix = inv(FisherMatrix)

print "Covariance Matrix:"
print CovarMatrix

print "Errors (marginalized):"
print "error on Omega_M = ", sqrt(CovarMatrix[0,0]), "error on w = ", sqrt(CovarMatrix[1,1]), "error on M = ", sqrt(CovarMatrix[2,2])

print "Errors (unmarginalized):"
print "error on Omega_M = ", 1.0/(sqrt(FisherMatrix[0,0])), "error on w = ", 1.0/(sqrt(FisherMatrix[1,1])), "error on M = ", 1.0/(sqrt(FisherMatrix[2,2]))

#####################################################################################




#Varying both omega_M and w, marginalized over Mscript
omegaM_array = arange(0.1, 0.5, 0.03) #dx = 0.0005
w_array = arange(-1.5, -0.5, 0.05) #dx = 0.0008

def Likelihood(omega_M, w_, script_M):
    integrand = lambda script_M: exp(-(sum( ( (mag_data - mag(omega_M, w_, script_M)) / error )**2.0 )) / 2.0)
    #L = integrate.quad(integrand, script_M-40.0, script_M+40.0)[0]
    #print L
    beginning = datetime.datetime.now()
    L = integrate.quad(integrand, script_M-10.0, script_M+10.0)[0]
    end = datetime.datetime.now()
    print end-beginning
    #scriptm = arange(script_M-2.0, script_M+2.0, 0.5)
    #L_trapz = zeros_like(scriptm)
    #for i in range(len(scriptm)):
    #    print (-(sum( ( (mag_data - mag(omega_M, w_, scriptm[i])) / error )**2.0 ) - 6*10**4) / 2.0)
    #    L_trapz[i] = trapz( exp(-(sum( ( (mag_data - mag(omega_M, w_, scriptm[i])) / error )**2.0 ) - 10000) / 2.0) )
    #L_=sum(L_trapz)
    
    
    
    return L

scriptL = zeros( (len(w_array), len(omegaM_array)) )
print shape(scriptL), shape(omegaM_array), shape(w_array)
for i in range(len(w_array)):
    print "row", i, "of", len(w_array)
    for j in range(len(omegaM_array)):
        scriptL[i,j] = Likelihood(omegaM_array[j], w_array[i], scriptM)



print omegaM_array[where(scriptL == amax(scriptL))[1]], w_array[where(scriptL == amax(scriptL))[0]]



np.savetxt('L_array_Case2.txt', scriptL)


#plt.figure(0)
#x, y = meshgrid(omegaM_array, w_array)
#print shape(x), shape(y), shape(scriptL)
#plt.contour(x, y, scriptL)#, [1.0, 1.0/4.0] )


plt.imshow(scriptL,origin='lower',interpolation='none',extent=(min(omegaM_array),max(omegaM_array),min(w_array),max(w_array)),aspect='auto')





font = {'family' : 'normal',
    'weight' : 'normal',
    'size'   : 26}
matplotlib.rc('font', **font)


plt.show()
