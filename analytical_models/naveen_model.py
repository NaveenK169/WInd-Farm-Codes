import numpy as np
import math as mt 
from functions_all.masking import * ##making function


def naveen_isoturb(x,y,z,ny,nz,xturb_he,yturb,zturb,turb_diam,CT_value,sigma_0,k_star,f, slope_f,const_f,gamma,
                 u0_les_cent,uhubz_hnt_1d, u0mod_cent,uhub_model_1d,f_fit,gamma_const,extra_term):
    
    uwc_model_corr = np.zeros(len(x))
    cen_model_corr = np.zeros(len(x))
    rad_model_corr = np.zeros((len(x), len(y), len(z)))
    for j in range(len(x)):
        xdist = x[j] - xturb_he
        if xdist > 0:
            sig_he = sigma_0 + k_star*xdist

            if f_fit == 0:
                F = f[j]
            elif f_fit == 1:   
                F = slope_f*xdist + const_f
 
            if gamma_const == 0:
                arg_cor =  1- ((1-F)/((sig_he/turb_diam)**2)*(CT_value/8 + gamma/mt.pi))
            if gamma_const == 1:
                arg_cor =  1- ((1-F)*CT_value)/(8*(sig_he/turb_diam)**2)

            if arg_cor > 0:
                uwc_model_corr[j] = (1-mt.sqrt((arg_cor)))
                
            cen_model_corr[j] = uwc_model_corr[j]
            ###------------radial--------------------------------------------------------------------------------- 
            for iz in range(nz):
                for iy in range(ny):
                    rsq = (z[iz] - zturb)**2 + (y[iy] - yturb)**2
                    ex = np.exp((-1/2)*(rsq/sig_he**2))
                    rad_model_corr[j,iy,iz] = cen_model_corr[j]*ex

            if extra_term == 0:
                rad_model_corr[j,:,:] = rad_model_corr[j,:,:] + (u0_les_cent[j]/uhubz_hnt_1d-1)
            if extra_term == 1:
                rad_model_corr[j,:,:] = rad_model_corr[j,:,:] + (u0mod_cent[j]/uhub_model_1d-1)
    
    return cen_model_corr, rad_model_corr


def naveen_isoturb_limit(x,y,z,ny,nz,xturb_he,yturb,zturb,turb_diam,CT_value,sigma_0,k_star,f,slope_f,const_f,gamma,
                 u0_les_cent,uhubz_hnt_1d, u0mod_cent,uhub_model_1d,f_fit,gamma_const,extra_term):
    
    ind_fact = np.zeros((len(x))); uwc_model_corr = np.zeros(len(x))
    cen_model_corr = np.zeros(len(x)); rad_model_corr = np.zeros((len(x), len(y), len(z)))
    
    for j in range(len(x)):
        ind_fact[j] = 2 * 0.5*(1-mt.sqrt(1-CT_value))
        xdist = x[j] - xturb_he
        if xdist > 0:
            sig_he = sigma_0 + k_star*xdist

            if f_fit == 0:
                F = f[j]
            elif f_fit == 1:   
                F = slope_f*xdist + const_f
            # F = -0.006 *xdist  - 0.020

            if gamma_const == 0:
                arg_cor =  1- ((1-F)/((sig_he/turb_diam)**2)*(CT_value/8 + gamma[j]/mt.pi))
            if gamma_const == 1:
                arg_cor =  1- ((1-F)*CT_value)/(8*(sig_he/turb_diam)**2)

            if arg_cor > 0:
                uwc_model_corr[j] = (1-mt.sqrt((arg_cor))) #+ (u0_les_cent[j]/uhubz_hnt_1d-1)
                
            cen_model_corr[j] = np.minimum(ind_fact[j], uwc_model_corr[j])
            ###------------radial--------------------------------------------------------------------------------- 
            for iz in range(nz):
                for iy in range(ny):
                    rsq = (z[iz] - zturb)**2 + (y[iy] - yturb)**2
                    ex = np.exp((-1/2)*(rsq/sig_he**2))
                    rad_model_corr[j,iy,iz] = cen_model_corr[j]*ex

            if extra_term == 0:
                rad_model_corr[j,:,:] = rad_model_corr[j,:,:] + (u0_les_cent[j]/uhubz_hnt_1d-1) #+ gamma
            elif extra_term == 1:
                rad_model_corr[j,:,:] = rad_model_corr[j,:,:] + (u0mod_cent[j]/uhub_model_1d-1)
    
    return cen_model_corr, rad_model_corr


def naveen_isoturb_iterative(x,y,z,dy,dz,ny,nz,xturb_he,yturb,zturb,ifringe,turb_diam,CT_value,sigma_0,k_star,gamma,
                 u0_les_cent,uhubz_hnt_1d,u0_model,u0mod_cent,uhub_model_1d,gamma_const,extra_term):

    uwc_model_corr = np.zeros(len(x));  cen_model_corr = np.zeros(len(x))
    rad_model_corr = np.zeros((len(x), len(y), len(z)))
    fxx = np.zeros(len(x))

    count = 0; err = 1000
    while err>0.01:
        count += 1
        print(count)
        for j in range(len(x)):
            xdist = x[j] - xturb_he
            if xdist > 0:
                sig_he = sigma_0 + k_star*xdist

                # F = slope_f*xdist + cons_f

                if gamma_const == 0:
                    arg_cor =  1- ((1-fxx[j])/((sig_he/turb_diam)**2)*(CT_value/8 + gamma/mt.pi))
                elif gamma_const == 1:
                    arg_cor =  1- ((1-fxx[j])*CT_value)/(8*(sig_he/turb_diam)**2)

                if arg_cor > 0:
                    uwc_model_corr[j] = (1-mt.sqrt((arg_cor)))
                    
                cen_model_corr[j] = uwc_model_corr[j]
                ###------------radial--------------------------------------------------------------------------------- 
                for iz in range(nz):
                    for iy in range(ny):
                        rsq = (z[iz] - zturb)**2 + (y[iy] - yturb)**2
                        ex = np.exp((-1/2)*(rsq/sig_he**2))
                        rad_model_corr[j,iy,iz] = cen_model_corr[j]*ex

                if extra_term == 0:
                    rad_model_corr[j,:,:] = rad_model_corr[j,:,:] + (u0_les_cent[j]/uhubz_hnt_1d-1) #+ gamma
                elif extra_term == 1:
                    rad_model_corr[j,:,:] = rad_model_corr[j,:,:] + (u0mod_cent[j]/uhub_model_1d-1)
               

        ###----findind the velocity from the deficit ----
        vel_model = u0_model - (rad_model_corr*uhub_model_1d)

        left_term = (vel_model * (u0_model - uhub_model_1d))/uhub_model_1d**2

        rad_sq = (turb_diam/2)+0.001
        mask_radial = mask_single_turb(left_term, y, z, ny, nz, yturb, zturb,rad_sq)
        left_term_mask = np.sum(left_term * mask_radial, axis=(1, 2)) * dy * dz

        right_term = (vel_model * rad_model_corr*uhub_model_1d)/uhub_model_1d**2
        right_term_mask = np.sum(right_term * mask_radial, axis=(1, 2)) * dy * dz

        # fx = left_term_mask/right_term_mask
        fx = np.divide(left_term_mask, right_term_mask, out=np.zeros_like(left_term_mask, dtype=float), where=right_term_mask != 0)

        err = np.sqrt(np.sum(abs(fxx[:ifringe] - fx[:ifringe]))/len(fx[:ifringe]))
        # print(err)

        ###-----final f(x) -----
        fxx = fx    

    return fxx, cen_model_corr, rad_model_corr
    

def naveen_isoturb_limit_iterative(x,y,z,dy,dz,ny,nz,xturb_he,yturb,zturb,ifringe,turb_diam,CT_value,sigma_0,k_star,gamma,
                 u0_les_cent,uhubz_hnt_1d,u0_model,u0mod_cent,uhub_model_1d,gamma_const,extra_term):

    uwc_model_corr = np.zeros(len(x));  cen_model_corr = np.zeros(len(x))
    rad_model_corr = np.zeros((len(x), len(y), len(z)))
    ind_fact = np.zeros((len(x)))

    count = 0; err = 1000
    fxx = np.zeros(len(x))
    
    while err>0.01:
        count += 1
        print(count)
        for j in range(len(x)):
            ind_fact[j] = 2 * 0.5*(1-mt.sqrt(1-CT_value))
            xdist = x[j] - xturb_he
            if xdist > 0:
                sig_he = sigma_0 + k_star*xdist

                # F = slope_f*xdist + cons_f

                if gamma_const == 0:
                    arg_cor =  1- ((1-fxx[j])/((sig_he/turb_diam)**2)*(CT_value/8 + gamma/mt.pi))
                elif gamma_const == 1:
                    arg_cor =  1- ((1-fxx[j])*CT_value)/(8*(sig_he/turb_diam)**2)

                if arg_cor > 0:
                    uwc_model_corr[j] = (1-mt.sqrt((arg_cor)))
                    
                cen_model_corr[j] = np.minimum(ind_fact[j], uwc_model_corr[j])
                ###------------radial--------------------------------------------------------------------------------- 
                for iz in range(nz):
                    for iy in range(ny):
                        rsq = (z[iz] - zturb)**2 + (y[iy] - yturb)**2
                        ex = np.exp((-1/2)*(rsq/sig_he**2))
                        rad_model_corr[j,iy,iz] = cen_model_corr[j]*ex

                if extra_term == 0:
                    rad_model_corr[j,:,:] = rad_model_corr[j,:,:] + (u0_les_cent[j]/uhubz_hnt_1d-1) #+ gamma
                elif extra_term == 1:
                    rad_model_corr[j,:,:] = rad_model_corr[j,:,:] + (u0mod_cent[j]/uhub_model_1d-1)
               

        ###----findind the velocity from the deficit ----
        vel_model = u0_model - (rad_model_corr*uhub_model_1d)

        left_term = (vel_model * (u0_model - uhub_model_1d))/uhub_model_1d**2

        rad_sq = (turb_diam/2)+0.001
        mask_radial = mask_single_turb(left_term, y, z, ny, nz, yturb, zturb,rad_sq)
        left_term_mask = np.sum(left_term * mask_radial, axis=(1, 2)) * dy * dz

        right_term = (vel_model * rad_model_corr*uhub_model_1d)/uhub_model_1d**2
        right_term_mask = np.sum(right_term * mask_radial, axis=(1, 2)) * dy * dz

        # fx = left_term_mask/right_term_mask
        fx = np.divide(left_term_mask, right_term_mask, out=np.zeros_like(left_term_mask, dtype=float), where=right_term_mask != 0)

        err = np.sqrt(np.sum(abs(fxx[:ifringe] - fx[:ifringe]))/len(fx[:ifringe]))
        print(err)

        ###-----final f(x) -----
        fxx = fx    

    return fxx, cen_model_corr, rad_model_corr