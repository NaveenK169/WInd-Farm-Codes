import numpy as np
import math as mt

####---finding f(x) ------#######
####see eq - (6); kethavth et.al (POF - 2022)
##--for your kind information the below both ways gives correct f(x) but the error between both ways is below L2_norm =6%, RMS=1.8 ------- 
##--moslty use the first way of calculation

####--- first way calculated --------#######################################################
def cal_eq6_naveenpof2022_niranjan_naveen(vel_he,vel_hnt,uhubhnt_1d,y,dy,ny,z,dz,nz,yturb,zturb,turb_diam,radial_factor):
    normfac = uhubhnt_1d**2
    # Initialize mask_radial with zeros of the same shape as u
    mask_radial = np.zeros_like(vel_he)
    # Calculate the square of the radius for comparison
    rad_sq = radial_factor * turb_diam**2 / 4
    # Loop over the indices of y and z dimensions
    for j in range(ny):
        for k in range(nz):
            # Compute the squared radial distance
            rad_dist_sq = (y[j] - yturb)**2 + (z[k] - zturb)**2
            if rad_dist_sq < rad_sq:
                mask_radial[:, j, k] = 1

    ###-- see eq - (6); kethavth et.al (POF - 2022) -----
    right_term = np.sum(vel_he * (vel_hnt - vel_he) * mask_radial, axis=(1, 2)) * dy * dz / normfac
    left_term = np.sum(vel_he * (vel_hnt - uhubhnt_1d) * mask_radial, axis=(1, 2)) * dy * dz / normfac

    Fcorr = left_term/right_term

    return left_term, right_term, Fcorr


####---Second way ----#############################################################
################################################################################################################
#mask function for each turbine (finding radial distance)
def masking(arr,radius,xcenter,ycenter,dimarr1,dimarr2):    #dimaar1 = x and dimaar2 = y
    num = 0                      # number points inside the mask circle
    for i in range(len(dimarr1)):
        for j in range(len(dimarr2)):
            dist = mt.sqrt((dimarr1[i]-xcenter)**2+(dimarr2[j]-ycenter)**2)     
            if dist < radius:
                arr[i,j] = 1
                num = num+1    
    return arr, num

def mask_param(arr,y,z,zhub,yturb,radius):
    iz = np.argmin(abs(z-zhub))
    if z[iz] > zhub:
        iz = iz-1
    maskarr = np.zeros((len(y),len(z)))
    for ii in range(len(yturb)):
        iy = np.argmin(abs(y-yturb[ii]))
        if y[iy] > yturb[ii]:
            iy = iy-1
        # mask
        [maskarr, num] = masking(maskarr,radius,y[iy],z[iz],y,z)
    arrmask = np.multiply(arr,maskarr)   ####multplying mask with velocity because mask gives ones 
    arrmean = np.sum(arrmask)/(num*len(yturb)) #3); print(num)   ### ave of 
    return arrmean

################################################################################################
#####-----finding f(x)-------------------
def cal_eq6_naveenpof2022_naveen(left_term,right_term,x,y,z,yturb,zturb,turb_diam,radial_factor):
    # Calculate the square of the radius for comparison
    turb_radius = radial_factor * (turb_diam**2 / 4)
    yturb = [yturb]
    ###----left term mask on eq6: Naveenpof2022----
    left_term_mask = np.zeros(len(x))
    for im in range(len(x)):
        left_term_mask[im] = mask_param(left_term[im,:,:],y,z,zturb,yturb,turb_radius+0.01)
    ###----right term mask eq6: Naveenpof2022----
    right_term_mask = np.zeros(len(x))
    for im in range(len(x)):
        right_term_mask[im] = mask_param(right_term[im,:,:],y,z,zturb,yturb,turb_radius+0.01)

    epsilon = 1e-10  # A small value to avoid division by zero
    final_term_mask = left_term_mask/(right_term_mask+epsilon)
    
    return left_term_mask, right_term_mask, final_term_mask
