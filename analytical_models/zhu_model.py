import numpy as np
import math as mt


def find_k1(radius,r_half):

    if radius < r_half:
        k1 = mt.sin(mt.pi/2)
    elif radius >= r_half:
        k1 = 1
    return k1

def ground_effect_correction(y,z,yturb,zturb,radius, r_half, sig_he_t, tihub_hnt_1d):
    k1 = find_k1(radius, r_half)
    ### Calculate azimuth angle in radians
    a_rad = mt.atan2(z - zturb, y - yturb)
 
    if z >= zturb:
        del_r = 0.23*np.sin(a_rad)*(k1*np.exp(-((radius - r_half)**2)/(2*(sig_he_t**2))))
    
    elif z < zturb:
        del_r = -1.23*np.sin(a_rad)*(k1*np.exp(-((radius - r_half)**2)/(2*(sig_he_t**2))))

    corr_factor = del_r*tihub_hnt_1d

    return corr_factor

def finding_added_ti_zhu(x,y,z,nx,ny,nz,xturb,yturb,zturb,CT_value,turb_diam,sigma_0,k_star,tihub_hnt_1d):
    added_ti = np.zeros((len(x),len(y),len(z)))
    radius = np.zeros((len(x),len(y),len(z)))

    for i in range(len(x)):
        xloc = x[i] - xturb
        if xloc > 0:
            sig_he = sigma_0 + (k_star*xloc)
            r_half = sig_he * mt.sqrt(2*mt.log(2))
            sig_he_t = sig_he/mt.sqrt(2*mt.log(2))

            val = (0.39*tihub_hnt_1d**(-0.18))/(1.5+((0.8/mt.sqrt(CT_value))*(xloc/turb_diam)**0.8))
            for iz in range(nz):
                for iy in range(ny):
                    radius = mt.sqrt((z[iz] - zturb)**2 + (y[iy] - yturb)**2)
    
                    if radius < r_half:
                        corr_factor = ground_effect_correction(y[iy],z[iz],yturb,zturb,radius, r_half, sig_he_t, tihub_hnt_1d)
                        added_ti[i,iy,iz] = val*(1-(0.15*(1+np.cos(mt.pi*radius/r_half)))) + corr_factor
                    
                    elif radius >= r_half:
                        corr_factor = ground_effect_correction(y[iy],z[iz],yturb,zturb,radius, r_half, sig_he_t, tihub_hnt_1d)
                        added_ti[i,iy,iz] = val*(np.exp(-(radius - r_half)**2/(2*sig_he_t**2))) + corr_factor

    return added_ti


def finding_ti_zhu(x,y,z,nx,ny,nz,xturb,yturb,zturb,CT_value,turb_diam,sigma_0,k_star,tihub_hnt_1d):
    added_ti = finding_added_ti_zhu(x,y,z,nx,ny,nz,xturb,yturb,zturb,CT_value,turb_diam,sigma_0,k_star,tihub_hnt_1d)
    ti_zhu = np.zeros((len(x),len(y),len(z)))
    for i in range((len(x))):
        for j in range((len(y))):
            for k in range((len(z))):

                if added_ti[i,j,k] >= 0:
                    ti_zhu[i,j,k] = np.sqrt(abs(tihub_hnt_1d**2 + added_ti[i,j,k]**2))

                elif added_ti[i,j,k] < 0:
                    ti_zhu[i,j,k] = np.sqrt(abs(tihub_hnt_1d**2 - added_ti[i,j,k]**2))
    return ti_zhu


def zhu_isoturb(x,y,z,ny,nz,dy,dz,xturb,yturb,zturb,CT_value,sigma_0,k_star,turb_diam,zhu_ct,u0_model,uhub_model_1d,mask_radial):
    
    cen_model = np.zeros((len(x)))
    cen_model_limit = np.zeros((len(x)))
    rad_model = np.zeros((len(x),len(y),len(z)))

    hxx = np.ones(len(x)); err = 100
    while err>0.01:
        for j in range(len(x)):
            xdist = x[j] - xturb
            if xdist > 0:
                sig_he = sigma_0 + k_star *xdist
                # sig_he = zhu_sigma_0 + zhu_kstar*xdist
    
                ct = (CT_value + zhu_ct[j])/8; val = hxx[j]**2 * (sig_he/turb_diam)**2
                aa = ct/val
                ###---- applying consition to ensure aa value not more than 1 ------
                if aa > 1:
                    aa = 0.999
                else:
                    aa = aa

                arg = 1 - (aa)
                if arg > 0:
                    cen_model[j] = hxx[j] * (1 - np.sqrt(arg))
                    
                cen_model_limit[j] = cen_model[j]
                ###------------radial--------------------------------------------------------------------------------- 
                for iz in range(nz):
                    for iy in range(ny):
                        rsq = (z[iz] - zturb)**2 + (y[iy] - yturb)**2
                        ex = np.exp((-1/2)*(rsq/sig_he**2))
                        rad_model[j,iy,iz] = cen_model_limit[j]*ex

        # numa = vel_hnt*rad_model
        numa = u0_model*rad_model
        numa_mask = np.sum(numa * mask_radial, axis=(1, 2)) * dy * dz

        # zhub= zturb; yhub = [yturb]
        # numa_mask = np.zeros(len(x))
        # for im in range(len(x)):
        #     numa_mask[im] = fun.mask_param(numa[im,:,:],y,z,zhub,yhub,0.5+0.001)

        dena_mask = np.sum(rad_model * mask_radial, axis=(1, 2)) * dy * dz
        # dena_mask = np.zeros(len(x))
        # for im in range(len(x)):
        #     dena_mask[im] = fun.mask_param(rad_model[im,:,:],y,z,zhub,yhub,0.5+0.001)

        dena = uhub_model_1d*dena_mask

        hx = np.divide(numa_mask, dena, out=np.zeros_like(numa_mask, dtype=float), where=dena != 0)

        err = np.sqrt(np.sum(abs(hxx - hx))/len(hx))  ### L2 err
        # err = np.sqrt(np.mean((hxx - hx)**2))  ### RMSE 
        
        hxx = hx


    return cen_model_limit, rad_model


def zhu_isoturb_limit(x,y,z,ny,nz,dy,dz,xturb,yturb,zturb,CT_value,sigma_0,k_star,turb_diam,zhu_ct,u0_model,uhub_model_1d,mask_radial):
    cen_model = np.zeros((len(x))); ind_fact = np.zeros((len(x)))
    cen_model_limit = np.zeros((len(x)))
    rad_model = np.zeros((len(x),len(y),len(z)))

    hxx = np.ones(len(x)); err = 100
    while err>0.01:
        for j in range(len(x)):
            ind_fact[j] = 2 * 0.5*(1-mt.sqrt(1-CT_value))
            xdist = x[j] - xturb
            if xdist > 0:
                sig_he = sigma_0 + k_star *xdist
                # sig_he = zhu_sigma_0 + zhu_kstar*xdist
                if hxx[j] > 0:
                    ct = (CT_value + zhu_ct[j])/8; val = hxx[j]**2 * (sig_he/turb_diam)**2
                    aa = ct/val
                    ###---- applying consition to ensure aa value not more than 1 ------
                    if aa > 1:
                        aa = 0.999
                    else:
                        aa = aa
                    
                    arg = 1 - (aa)
                    if arg > 0:
                        cen_model[j] = hxx[j] * (1 - np.sqrt(arg))
                        
                    cen_model_limit[j] = np.minimum(cen_model[j], ind_fact[j])
                    ###------------radial--------------------------------------------------------------------------------- 
                    for iz in range(nz):
                        for iy in range(ny):
                            rsq = (z[iz] - zturb)**2 + (y[iy] - yturb)**2
                            ex = np.exp((-1/2)*(rsq/sig_he**2))
                            rad_model[j,iy,iz] = cen_model_limit[j]*ex

        # numa = vel_hnt*rad_model
        numa = u0_model*rad_model
        numa_mask = np.sum(numa * mask_radial, axis=(1, 2)) * dy * dz

        # zhub= zturb; yhub = [yturb]
        # numa_mask = np.zeros(len(x))
        # for im in range(len(x)):
        #     numa_mask[im] = fun.mask_param(numa[im,:,:],y,z,zhub,yhub,0.5+0.001)

        dena_mask = np.sum(rad_model * mask_radial, axis=(1, 2)) * dy * dz
        # dena_mask = np.zeros(len(x))
        # for im in range(len(x)):
        #     dena_mask[im] = fun.mask_param(rad_model[im,:,:],y,z,zhub,yhub,0.5+0.001)

        dena = uhub_model_1d*dena_mask

        hx = np.divide(numa_mask, dena, out=np.zeros_like(numa_mask, dtype=float), where=dena != 0)

        err = np.sqrt(np.sum(abs(hxx - hx))/len(hx))  ### L2 err
        # err = np.sqrt(np.mean((hxx - hx)**2))  ### RMSE 
        
        hxx = hx


    return cen_model, rad_model
