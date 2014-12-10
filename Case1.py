from numpy import *
import matplotlib.pyplot as plt 
from pylab import *
from scipy import integrate
from scipy import misc
from numpy.linalg import inv




#Uploading SN data
#columns: name, z, magnitudes, errors
data = genfromtxt(fname='SNdata.txt')

z = data[:,1]             #SN redshifts
DM_data = data[:,2]       #SN measured distance modulus
error = data[:,3]         #SN errors

M = -19.3146267582        #absolute magnitude of the SN standard candle (h=0.7, with systematics)
mag_data = DM_data + M    #SN measured apparent magnitudes

c = 3*10**5 #km/s         #Definition of c
scriptM = M - 5.0*log10(70.0 / c) + 25.0      #Definition of script M (H is normalized)

omegaM = 0.3      #Omega matter
omegaDE = 0.7     #Omega_Lambda
w = -1.0          #Equation of State




#Defining a function to calculate the magnitudes of each SNe as a function of omega_M, omega_DE, and script_M:
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




#FISHER MATRIX
#####################################################################################

#Find dm/dp for each parameter (omega_M, w, and script_M) for each SNe
#In other words, find the slope of mag(omega_M, w, and script_M) at
#these values: omega_M = 0.3, w = -1.0, and script_M = const, for 
#each SNe
dm_domegaM  = misc.derivative(lambda x: mag(x, omegaDE, scriptM), omegaM, dx=3e-1) 
dm_domegaDE = misc.derivative(lambda x: mag(omegaM, x, scriptM), omegaDE, dx=5e-1)
dm_dscriptM = misc.derivative(lambda x: mag(omegaM, omegaDE, x), scriptM, dx=2e-1)

#Define 1/sigma**2, where sigma = errors for each SNe:
one_over_var = 1.0 / (error**2.0)

#Evaluate each element of the Fisher Matrix:
F11 = sum(one_over_var * dm_domegaM * dm_domegaM)
F12 = sum(one_over_var * dm_domegaM * dm_domegaDE)
F13 = sum(one_over_var * dm_domegaM * dm_dscriptM)
F23 = sum(one_over_var * dm_domegaDE * dm_dscriptM)
F22 = sum(one_over_var * dm_domegaDE * dm_domegaDE)
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
print "error on Omega_M = ", sqrt(CovarMatrix[0,0]), "error on Omega_DE = ", sqrt(CovarMatrix[1,1]), "error on M = ", sqrt(CovarMatrix[2,2])

print "Errors (unmarginalized):"
print "error on Omega_M = ", 1.0/(sqrt(FisherMatrix[0,0])), "error on Omega_DE = ", 1.0/(sqrt(FisherMatrix[1,1])), "error on M = ", 1.0/(sqrt(FisherMatrix[2,2]))

#Selecting the marginalized covariance matrix:
CovarMatrix_marg = CovarMatrix[0:2, 0:2]

print "Marginalized Covariance Matrix:"
print CovarMatrix_marg

#Finding the marginalized Fisher matrix by taking the inverse
#of the marginalized covariance matrix:
FisherMatrix_marg = inv(CovarMatrix_marg)

print "Marginalized Fisher Matrix:"
print FisherMatrix_marg




#Plotting the contours of predicted errors from the Fisher Matrix:
plt.figure(0)
X_omegaM = linspace(-3.0, 2.0, 5000)
Y_omegaDE = linspace(-3.0, 2.0, 5000)
X, Y = meshgrid(X_omegaM, Y_omegaDE)
#Determining the coefficients of the 2D ellipse equation:
#unmarginalized:
G11 = FisherMatrix[0,0]
G12 = FisherMatrix[0,1]
G22 = FisherMatrix[1,1]
#marginalized:
G11_m = FisherMatrix_marg[0,0]
G12_m = FisherMatrix_marg[0,1]
G22_m = FisherMatrix_marg[1,1]

#ellipse functions:
Z = G11*(X-0.3)**2 + 2.0*G12*(X-0.3)*(Y-0.7) + G22*(Y-0.7)**2           #unmarginalized
Z_m = G11_m*(X-0.3)**2 + 2.0*G12_m*(X-0.3)*(Y-0.7) + G22_m*(Y-0.7)**2   #marginalized

#Plotting the one- and two-sigma contours in the Omega_M-w plane for both cases
C = plt.contour(X, Y, Z, [1.0/0.434, 1.0/0.167], colors = 'b', linestyles = 'solid', 
                label = 'fixed $\mathcal{M}$')
C_m = plt.contour(X, Y, Z_m, [1.0/0.434, 1.0/0.167], colors = 'r', linestyles = 'solid', 
                label = 'marginalized over $\mathcal{M}$')

#Labels:
plt.xlabel('$\Omega_M$')
plt.ylabel('$\Omega_DE$')
plt.legend(loc='upper right')

labels = ['fixed $\mathcal{M}$', 'marginalized over $\mathcal{M}$']
C.collections[0].set_label(labels[0])
C_m.collections[0].set_label(labels[1])

plt.legend(loc='upper right', prop={'size':15})

#####################################################################################




#Varying omega_M
omegaM_array = arange(0.27, 0.31, 0.0002)

#Calculating ChiSquared values
ChiSqd_omegaM = zeros_like(omegaM_array)
for i in range(len(omegaM_array)):
    ChiSqd_omegaM[i] = sum( ( (mag_data - mag(omegaM_array[i], omegaDE, scriptM)) / error )**2.0 )

#Calculating the Likelihood
L_omegaM_ = exp(-(ChiSqd_omegaM) / 2.0)
L_omegaM = L_omegaM_/max(L_omegaM_) #normalizing


#Plotting Likelihood
plt.figure(1)
plt.plot(omegaM_array, L_omegaM, 'k-')

#Defining the 68th and 95th percentiles:
delta_ChiSqd = ChiSqd_omegaM - min(ChiSqd_omegaM)
_68_ = where( delta_ChiSqd < 1.0 )
_95_ = where( delta_ChiSqd < 4.0 )

#Checking whether they actually are the 68th and 95th percentiles
print trapz(L_omegaM[_68_])/trapz(L_omegaM)
print trapz(L_omegaM[_95_])/trapz(L_omegaM)

#Plotting errors
plt.plot(omegaM_array[_95_], L_omegaM[_95_], 'b-', linewidth=1.5, label='two sigma')
plt.plot(omegaM_array[_68_], L_omegaM[_68_], 'r-', linewidth=1.5, label='one sigma')
plt.axvline(omegaM_array[_95_][0], color='b')        
plt.axvline(omegaM_array[_95_][-1], color='b')
plt.axvline(omegaM_array[_68_][0], color='r')        
plt.axvline(omegaM_array[_68_][-1], color='r')

plt.xlabel('$\Omega_{M}$')
plt.ylabel('$\mathcal{L}$')
plt.legend(loc='upper right', prop={'size':15})

#Calculating the most likely value of omegaM with one- and two-sigma errors
omegaM_correct = omegaM_array[where(L_omegaM == max(L_omegaM))]
onesigma = ( (omegaM_array[where(L_omegaM == max(L_omegaM))] - omegaM_array[_68_][0]) + (omegaM_array[_68_][-1] - omegaM_array[where(L_omegaM == max(L_omegaM))]) ) / 2.0
twosigma = ( (omegaM_array[where(L_omegaM == max(L_omegaM))] - omegaM_array[_95_][0]) + (omegaM_array[_95_][-1] - omegaM_array[where(L_omegaM == max(L_omegaM))]) ) / 2.0

print "omega_M = ", omegaM_array[where(L_omegaM == max(L_omegaM))], '+-', onesigma, "(onesigma)", twosigma, "(twosigma)"







#Varying omegaDE
omegaDE_array = arange(0.66, 0.72, 0.0003)

#Calculating ChiSquared values
ChiSqd_omegaDE = zeros_like(omegaDE_array)
for i in range(len(omegaDE_array)):
    ChiSqd_omegaDE[i] = sum( ( (mag_data - mag(omegaM, omegaDE_array[i], scriptM)) / error )**2.0 )

#Calculating the Likelihood
L_omegaDE_ = exp(-(ChiSqd_omegaDE) / 2.0)
L_omegaDE = L_omegaDE_/max(L_omegaDE_) #normalizing


#Plotting Likelihood
plt.figure(2)
plt.plot(omegaDE_array, L_omegaDE, 'k-')

#Defining the 68th and 95th percentiles:
delta_ChiSqd = ChiSqd_omegaDE - min(ChiSqd_omegaDE)
_68_ = where( delta_ChiSqd < 1.0 )
_95_ = where( delta_ChiSqd < 4.0 )

#Checking whether they actually are the 68th and 95th percentiles
print trapz(L_omegaDE[_68_])/trapz(L_omegaDE)
print trapz(L_omegaDE[_95_])/trapz(L_omegaDE)

#Plotting errors
plt.plot(omegaDE_array[_95_], L_omegaDE[_95_], 'b-', linewidth=1.5, label='two sigma')
plt.plot(omegaDE_array[_68_], L_omegaDE[_68_], 'r-', linewidth=1.5, label='one sigma')
plt.axvline(omegaDE_array[_95_][0], color='b')
plt.axvline(omegaDE_array[_95_][-1], color='b')
plt.axvline(omegaDE_array[_68_][0], color='r')
plt.axvline(omegaDE_array[_68_][-1], color='r')

plt.xlabel('w')
plt.ylabel('$\mathcal{L}$')
plt.legend(loc='upper right', prop={'size':15})

#Calculating the most likely value of w with one- and two-sigma errors
w_correct = omegaDE_array[where(L_omegaDE == max(L_omegaDE))]
onesigma = ( (omegaDE_array[where(L_omegaDE == max(L_omegaDE))] - omegaDE_array[_68_][0]) + (omegaDE_array[_68_][-1] - omegaDE_array[where(L_omegaDE == max(L_omegaDE))]) ) / 2.0
twosigma = ( (omegaDE_array[where(L_omegaDE == max(L_omegaDE))] - omegaDE_array[_95_][0]) + (omegaDE_array[_95_][-1] - omegaDE_array[where(L_omegaDE == max(L_omegaDE))]) ) / 2.0

print "omega_DE = ", omegaDE_array[where(L_omegaDE == max(L_omegaDE))], '+-', onesigma, "(onesigma)", twosigma, "(twosigma)"





font = {'family' : 'normal',
    'weight' : 'normal',
    'size'   : 26}
matplotlib.rc('font', **font)


plt.show()
