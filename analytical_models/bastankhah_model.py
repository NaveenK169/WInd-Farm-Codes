import numpy as np
import math as mt

def bastankhah_isoturb(x, y, z, ny, nz, xturb_he, yturb, zturb, turb_diam, CT_value, sigma_0, k_star):
    # sig_he = np.zeros(len(x))
    # arg_he = np.zeros(len(x))
    uwc_model = np.zeros(len(x))
    cen_model = np.zeros(len(x))
    rad_model = np.zeros((len(x), len(y), len(z)))

    for j in range(len(x)):
        xdist = x[j] - xturb_he
        if xdist > 0:
            sig_he = sigma_0 + k_star * xdist
            arg_he = 1 - CT_value / (8 * (sig_he / turb_diam)**2)

            if arg_he > 0:
                uwc_model[j] = 1 - mt.sqrt(arg_he)

            cen_model[j] = uwc_model[j]

            for iz in range(nz):
                for iy in range(ny):
                    rsq = (z[iz] - zturb)**2 + (y[iy] - yturb)**2
                    ex = np.exp(-0.5 * rsq / (sig_he**2))
                    rad_model[j, iy, iz] = cen_model[j] * ex

    return uwc_model, rad_model


#####################---------------------------------------
def bastankhah_isoturb_limit(x,y,z,ny,nz,xturb_he,yturb,zturb,turb_diam,CT_value,sigma_0,k_star):
    ind_fact = np.zeros((len(x))); uwc_model_limit = np.zeros(len(x)); cen_model_limit = np.zeros((len(x)))
    rad_model_limit = np.zeros((len(x),len(y),len(z)))
    for j in range(len(x)):
        ind_fact[j] = 2 * 0.5*(1-mt.sqrt(1-CT_value))
        xdist = x[j] - xturb_he
        if xdist > 0:
            # print(j)
            sig_he = sigma_0 + k_star*xdist

            arg_he = 1-CT_value/(8*(sig_he/turb_diam)**2)
            if arg_he > 0:
                uwc_model_limit[j] = 1-mt.sqrt((arg_he))
         
            cen_model_limit[j] = np.minimum(ind_fact[j], uwc_model_limit[j])
            ###------------radial--------------------------------------------------------------------------------- 
            for iz in range(nz):
                for iy in range(ny):
                    rsq = (z[iz] - zturb)**2 + (y[iy] - yturb)**2
                    ex = np.exp((-1/2)*(rsq/sig_he**2))
                    
                    rad_model_limit[j,iy,iz] =  cen_model_limit[j] * ex 
    
    return cen_model_limit, rad_model_limit

        