import numpy as np
import math as mt

def generate_positions(num_rows, num_cols, s_x, s_y, start_x, start_y):
    """
    This function generates turbine positions on a grid based on the specified number of rows and columns,
    spacing in the x and y directions, and starting coordinates.
    
    Arguments:
    num_rows -- Number of rows of turbines (integer).
    num_cols -- Number of columns of turbines (integer).
    s_x -- Spacing between turbines in the x-direction (float).
    s_y -- Spacing between turbines in the y-direction (float).
    start_x -- Starting x-coordinate for the first turbine (float).
    start_y -- Starting y-coordinate for the first turbine (float).
    
    Returns:
    positions -- List of tuples representing turbine positions [(x1, y1), (x2, y2), ..., (xn, yn)].
    """
    
    # Step 1: Initialize an empty list to store the turbine positions
    positions = []
    
    # Step 2: Iterate through each row and column to calculate positions
    for i in range(num_rows):
        for j in range(num_cols):
            # Step 3: Calculate the x and y coordinates for the turbine
            x = start_x + j * s_x
            y = start_y + i * s_y
            # Step 4: Append the calculated position to the positions list
            positions.append((x, y))
    
    # Step 5: Return the list of turbine positions
    return positions

def write_ADM_NR_file(turbine_id, xLoc, yLoc, zLoc, diam, yaw, tilt, cT, outdir_path):
    """
    This function writes the actuator disk data for a turbine to an input file.
    The data includes turbine location, diameter, yaw, tilt, and thrust coefficient.
    
    Arguments:
    turbine_id -- ID of the turbine (integer).
    xLoc -- x-coordinate of the turbine location (float).
    yLoc -- y-coordinate of the turbine location (float).
    zLoc -- z-coordinate of the turbine location (float).
    diam -- Diameter of the actuator disk (float).
    yaw -- Yaw angle of the turbine (float).
    tilt -- Tilt angle of the turbine (float).
    cT -- Thrust coefficient (float).
    outdir_path -- Directory path to save the input file (string).
    
    Returns:
    None
    """
    
    # Step 1: Create the filename based on turbine ID
    filename = outdir_path + f"ActuatorDisk_{turbine_id:04d}_input.inp"
    
    # Step 2: Open the file in write mode
    with open(filename, "w") as file:
        # Step 3: Write the actuator disk data to the file
        file.write(f"&ACTUATOR_DISK\n")
        file.write(f" xLoc             =  {xLoc}d0\n")
        file.write(f" yLoc             =  {yLoc}d0\n")
        file.write(f" zLoc             =  {zLoc}d0\n")
        file.write(f" diam             =  {diam}d0\n")
        file.write(f" yaw              =  {yaw}d0\n")
        file.write(f" tilt             =  {tilt}d0\n")
        file.write(f" cT               =  {cT}d0\n")
        file.write(f"/\n")
    
    # Step 4: The file is automatically closed after exiting the 'with' block


printed = False

def rotate_coordinates(x, y, center_x, center_y, angle_degrees):
    global printed
    
    # Step 1: Translate the point to the origin
    x_translated = x - center_x
    y_translated = y - center_y
    
    # Step 2: Convert the angle from degrees to radians
    angle_radians = np.radians(angle_degrees)
    
    # Step 3: Define the rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])
    
    # if not printed:
    #     print(rotation_matrix)
    #     printed = True
    
    # Step 4: Rotate the point using the rotation matrix
    rotated_coordinates = rotation_matrix @ np.array([x_translated, y_translated])
    
    # Step 5: Translate the point back to its original position
    x_new = rotated_coordinates[0] + center_x
    y_new = rotated_coordinates[1] + center_y
    
    # Step 6: Return the rotated coordinates
    return x_new, y_new

###########################################################################################
def act_turbine_location_xturb_yturb(turbine_positions):
    """
    This function extracts the unique x and y values from a list of turbine positions.
    
    Arguments:
    turbine_positions -- List of tuples representing turbine positions [(x1, y1), (x2, y2), ..., (xn, yn)].
    
    Returns:
    xturb -- Sorted list of unique x values from the turbine positions.
    yturb -- Sorted list of unique y values from the turbine positions.
    """
    
    # Step 1: Extract unique x values and sort them
    xturb = np.sort(list(set([position[0] for position in turbine_positions])))
    
    # Step 2: Extract unique y values and sort them
    yturb = np.sort(list(set([position[1] for position in turbine_positions])))
    
    # Step 3: Return the unique sorted x and y values
    return xturb, yturb

###########################################################################################
def rot_turbine_location_xturb_yturb(turbine_positions):
    """
    This function extracts the unique x and y values from a list of turbine positions while maintaining their order.
    Arguments:
    turbine_positions -- List of tuples representing turbine positions [(x1, y1), (x2, y2), ..., (xn, yn)].
    
    Returns:
    xturb -- List of unique x values from the turbine positions in original order.
    yturb -- List of unique y values from the turbine positions in original order.
    """
    # Step 1: Initialize lists to store unique x and y values
    xturb = [];  yturb = []
    
    # Step 2: Sets to track seen x and y values
    seen_x = set(); seen_y = set()
    
    # Step 3: Iterate through turbine positions and extract unique values
    for position in turbine_positions:
        x, y = position
        # Add unique x values to xturb while maintaining order
        if x not in seen_x:
            xturb.append(x) ; seen_x.add(x)
        
        # Add unique y values to yturb while maintaining order
        if y not in seen_y:
            yturb.append(y); seen_y.add(y)
        
    # Step 4: Convert lists to numpy arrays (if needed)
    xturb = np.array(xturb)
    yturb = np.array(yturb)

    return xturb, yturb


def find_turbine_values(dir, nx, ny, nz, Run, tidx, turb_diam, sx, sy):
    """
    This function calculates the thrust coefficient (CT) from a data file at a given timestep and 
    Computes CT values, disk velocities, and power for each turbine from .sth data file.

    Parameters:
    homogt_turb       -- Base directory for data (string)
    nx, ny, nz        -- Grid sizes (int)
    Run               -- Run number (string or int)
    tidx_hm           -- Time index (int)
    turb_diam         -- Turbine diameter (float)
    sx, sy            -- Layout dimensions (int) (sx * sy = total number of turbines)
    turbine_positions -- List of turbine positions (used to verify total count)

    Returns:
    Dictionary with mean CT, thrust, udisk, and power along rows.
    """

    # Step 1: Construct the file path to the .sth file using the input parameters
    file = dir + str(nx) + str(ny) + str(nz) + '/main_data/Run' + Run + '_t' + format(tidx, '06') + '.sth'

    # Step 2: Open the file and read the data
    data = open(file, 'r')  # Open the file for reading
    value = np.genfromtxt(data)  # Read the file data into a numpy array
    data.close()  # Close the file after reading

    ## Step 3: Extract turbine data
    thrust = -value[7::8] ; udisk = value[12::8];  power = thrust * udisk

    # Step 4: Reshape to (sx, sy)
    reshaped_thrust = thrust.reshape(sx, sy); reshaped_udisk = udisk.reshape(sx, sy); reshaped_power = power.reshape(sx, sy)

    # Step 5: Calculate mean across rows
    mean_thrust = np.mean(reshaped_thrust, axis=0); mean_udisk = np.mean(reshaped_udisk, axis=0)
    mean_power = np.mean(reshaped_power, axis=0)

    # Step 6: Calculate ct_prime and CT for all turbines
    ct_prime = (8 * thrust) / (mt.pi * udisk**2 * turb_diam**2)
    CT_value = (16 * ct_prime) / (ct_prime + 4)**2

    # Step 6: Return the calculated CT value
    return CT_value, thrust, udisk, power, mean_thrust, mean_udisk, mean_power