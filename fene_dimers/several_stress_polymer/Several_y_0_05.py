

# ### SELM via PyLAMMPs for Simulations
# Author: Paul Atzberger <br>
# http://atzberger.org/
# 

# In[1]:
#hi


from pprint import pprint
import pdb
import os;
script_base_name = "Several_y_0_05";
script_dir = os.getcwd();
import numpy as np

#for the position of the polymer
position_table = np.loadtxt('random_position_output.txt')


# In[2]:

# import the lammps module
try:  
  from selm_lammps.lammps import IPyLammps # use this for the pip install of pre-built package
  lammps_import_comment = "from selm_lammps.lammps import IPyLammps";  
  from selm_lammps import util as atz_util;
except Exception as e:  
  from lammps import IPyLammps # use this for direct install of package
  lammps_import_comment = "from lammps import IPyLammps";
  from atz_lammps import util as atz_util;
except Exception as e: # if fails to import, report the exception   
  print(e);
  lammps_import_comment = "import failed";
  pdb.post_mortem()
  
import numpy as np;
import matplotlib;
import matplotlib.pyplot as plt;

import sys,shutil,pickle,pdb;

import logging;

fontsize = 14;
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : fontsize};

matplotlib.rc('font', **font);


# ### Setup SELM Simulation

# In[3]:


# @base_dir
base_dir_output   = '%s/output/%s'%(script_dir,script_base_name);
atz_util.create_dir(base_dir_output);

dir_run_name = 'harmonic';
base_dir = '%s/%s_test000'%(base_dir_output,dir_run_name);

# remove all data from dir
atz_util.rm_dir(base_dir);

# setup the directories
base_dir_fig    = '%s/fig'%base_dir;
atz_util.create_dir(base_dir_fig);

base_dir_vtk    = '%s/vtk'%base_dir;
atz_util.create_dir(base_dir_vtk);

# setup logging
# @@@! Setup AtzLogging class and references...
atzLog = atz_util.AtzLogging(print,base_dir);
#atz_util.setup_log(print,base_dir);
print("");

print_log = atzLog.print_log;

# print the import comment
print_log(lammps_import_comment);

# change directory for running LAMMPS in output
print_log("For running LAMMPS changing the current working directory to:\n%s"%base_dir);
os.chdir(base_dir); # base the current working directory
#os.chdir(script_dir); # base the current working directory


# ### Setup LAMMPs

# In[4]:


L = IPyLammps();
atz_util.print_version_info(L);    


# ### Copy files to the output directory

# In[5]:


# copy the model files to the destination
src = script_dir + '/' + "Model_Several_y_0_05";
dst = base_dir + '/';
atz_util.copytree2(src,dst,symlinks=False,ignore=None);

print_log("Model files being copied:\n" + "src = " + str(src) + "\n" + "dst = " + str(dst));


# In[6]:


flag_copy_notebook_to_output = True;
if flag_copy_notebook_to_output:
  #cur_dir = os.getcwd();
  #src = cur_dir + '/' + script_base_name + '.ipynb';
  src = script_dir + '/' + script_base_name + '.py';    
  dst = base_dir + '/' + 'archive__' + script_base_name + '.py';
  shutil.copyfile(src, dst);
  print_log("Copying notebook to archive:\n" + "src = " + str(src) + "\n" + "dst = " + str(dst));


# ### Common Physical Parameters (nano units)

# In[7]:


# Reference units and parameters
units = {'name':'nano','mu':1.0,'rho':0.001,
         'KB':0.01380651,'T':298.15};
units.update({'KBT':units['KB']*units['T']});


# ### Setup the Simulation Files (such as .read_data)

# In[8]:


num_dim = 3;
box = np.zeros((num_dim,2));
LL = 202.5; box[:,0] = -LL; box[:,1] = LL;


# setup atoms
I_id = 1; I_type = 1; atom_types = []; 
atom_list = []; atom_mass_list = []; atom_id_list = []; 
atom_mol_list = []; atom_name_list = [];

# Setup polymers
num_polymers = 10  # Number of polymers
num_pts = 2        # Number of atoms in each polymer
spacing = 30.0     # Spacing between atoms in a polymer
polymer_spacing = 20.0  # Spacing between polymers

# Loop to create multiple polymers
for p in range(num_polymers):
    atom_name = f"polymer_pts_{p+1}"
    atom_name_list.append(atom_name)
    atom_types.append(I_type) 
    start_idx = p * num_pts
    end_idx = (p+1) * num_pts
    x = np.zeros((num_pts, num_dim))  # Allocate positions
    for i in range(num_pts):
        x[i, 0] = position_table[start_idx + i,0]  # Linear arrangement along x-axis
        x[i, 1] = position_table[start_idx + i,1]          # y-coordinate
        x[i, 2] = position_table[start_idx + i,2]         # z-coordinate
    num_pts = x.shape[0]; m0 = 1.123;
    atom_id = np.arange(I_id + 0,I_id + num_pts,dtype=int);
    mol_id = p + 1; atom_mol = np.ones(num_pts,dtype=int)*mol_id;
    atom_list.append(x); atom_mass_list.append(m0);
    atom_id_list.append(atom_id); atom_mol_list.append(atom_mol);
    I_id += num_pts;
    
    print_log("atom_name = " + str(atom_name));
    print_log(atom_id)
    print_log("num_pts = " + str(num_pts));
    print_log(f"Setup {x} polymers ")
    I_type += 1;

# tracer atoms
flag_tracer = True;
if flag_tracer:
  atom_name = "tracer_pts";
  atom_name_list.append(atom_name);
  atom_types.append(I_type); 
  atom_types[I_type - 1] = I_type;  
  num_pts_dir = 10; m0 = 1.123; 
  x1 = np.linspace(-LL,LL,num_pts_dir + 1,endpoint=False); dx = x1[1] - x1[0];
  x1 = x1 + 0.5*dx;
  xx = np.meshgrid(x1,x1,x1);
  x = np.stack((xx[0].flatten(),xx[1].flatten(),xx[2].flatten()),axis=1); # shape = [num_pts,num_dim]
  
  # >>> ADD small random noise <<<
  noise_amplitude = 2.0  # nm, much smaller than LJ cutoff (~10 nm)
  x += noise_amplitude * np.random.randn(*x.shape)

  #ipdb.set_trace();
  num_pts = x.shape[0];
  atom_id = np.arange(I_id + 0,I_id + num_pts,dtype=int);
  mol_id = 11; atom_mol = np.ones(x.shape[0],dtype=int)*mol_id;
  atom_list.append(x); atom_mass_list.append(m0); 
  atom_id_list.append(atom_id); atom_mol_list.append(atom_mol);
  I_type += 1; I_id += num_pts;
  print_log("atom_name = " + str(atom_name));
  print_log("num_pts = " + str(num_pts));
  print_log(atom_id)
# summary data    
# get total number of atoms
atom_types = np.array(atom_types,dtype=int);
num_atoms = I_id - 1; # total number of atoms

# setup bonds
I_id = 1; I_type = 1; bond_types = [1]; bond_name_list = ["fene_1"];
bond_list = []; bond_coeff_list = []; bond_id_list = [];

flag_bond_1 = True
if flag_bond_1:
    bond_name_list.append("fene_1")

    # Define FENE bond parameters
    KBT = units['KBT']
    ell = 5
    K = 10
    r0 = 50 # cannot be too close to the inital distance
    epsilon = 1
    sigma = 1
    bond_coeff_list.append(f"{K:.7f} {r0:.7f} {epsilon:.7f} {sigma:.7f}")

    bond_type_id = 1
    I_id = 1  # Bond ID counter

    bonds = []       # Will K = 50be a list of [atom1, atom2]
    bond_ids = []    # Will be list of bond IDs

    for i in range(num_polymers):
        atom_id = atom_id_list[i]  # [1, 2], [3, 4], etc.
        assert len(atom_id) == 2, f"Each polymer must have exactly 2 atoms, got {len(atom_id)}"

        bonds.append([atom_id[0], atom_id[1]])
        bond_ids.append(I_id)
        I_id += 1

    bond_list = [np.array(bonds, dtype=int)]
    bond_id_list = [np.array(bond_ids, dtype=int)]
    bond_types = np.array([1], dtype=int)

    num_bonds = len(bonds)

  
# summary data    
# summary data    
num_bonds = I_id - 1;
bond_types = np.array(bond_types,dtype=int);

# setup angles
I_id = 1; I_type =1 ; angle_types = []; angle_name_list = [];
angle_list = []; angle_coeff_list = []; angle_id_list = [];

flag_angles_1 = False;
if flag_angles_1:
  angle_name_list.append("atom_type_1");
  angle_types.append(I_type);
  #KBT = 2478959.87; K = 10*KBT; theta_0 = 180.0; # degrees
  KBT = units['KBT']; K = 5*KBT; theta_0 = 180.0; # degrees
  b = "harmonic %.7f %.7f"%(K,theta_0);
  angle_coeff_list.append(b);

# build angle bonds for type 1 atoms with type 1 atoms, closed loop
if flag_angles_1:
  I0 = atz_util.atz_find_name(atom_name_list,"polymer_pts"); I_atom_type = atom_types[I0];
  atom_id = atom_id_list[I_atom_type - 1]; nn = atom_id.shape[0];
  angles = np.zeros((nn,3),dtype=int);
  angle_id = np.zeros(angles.shape[0],dtype=int);
  for i in range(0,nn):
    i1 = atom_id[i]; i2 = atom_id[(i + 1)%nn]; i3 = atom_id[(i + 2)%nn]; # base 1 indexing
    angles[i,0] = i1; angles[i,1] = i2; angles[i,2] = i3;
    angle_id[i] = I_id; I_id += 1;
  angle_list.append(angles); angle_id_list.append(angle_id);
  I_type += 1;

# summary data    
num_angles = I_id - 1;
angle_types = np.array(angle_types,dtype=int);

# store the model information
model_info = {};
model_info.update({'num_dim':num_dim,'box':box,'atom_types':atom_types,
          'atom_list':atom_list,'atom_mass_list':atom_mass_list,'atom_name_list':atom_name_list,
          'atom_id_list':atom_id_list,'atom_mol_list':atom_mol_list,
          'bond_types':bond_types,'bond_list':bond_list,'bond_id_list':bond_id_list,
          'bond_coeff_list':bond_coeff_list,'bond_name_list':bond_name_list,
          'angle_types':angle_types,'angle_list':angle_list,'angle_id_list':angle_id_list,
          'angle_coeff_list':angle_coeff_list,'angle_name_list':angle_name_list});

# Combine all atom IDs into one list or array
all_atom_ids = np.concatenate(atom_id_list)

#pprint(model_info)
# In[9]:


# write .pickle data with the model setup information
filename = "model_setup.pickle";
print_log("Writing model data .pickle");
print_log("filename = " + filename);
s = model_info;
f = open(filename,'wb'); pickle.dump(s,f); f.close();

import pickle

# Load the pickle file
filename = "model_setup.pickle"
with open(filename, 'rb') as f:
    model_info = pickle.load(f)

# Access the data
print("Atom Types:", atom_types)

#print("Loaded model_info:", model_info)

# write the model .read_data file for lammps
filename = "Polymer.LAMMPS_read_data";
print_log("Writing model data .read_data");
print_log("filename = " + filename);
atz_util.write_read_data(filename=filename,print_log=print_log,**model_info);

for i, b in enumerate(bond_list):
    print(f"Bond {i+1}: {b}")


# In[10]:


#!cat Polymer.LAMMPS_read_data
#We can send collection of commands using the triple quote notation
s = """
# =========================================================================
# LAMMPS main parameter file and script                                    
#                                                                          
# Author: Paul J. Atzberger.               
#
# Based on script generated by MANGO-SELM Model Builder.
#                                                                          
# =========================================================================

# == Setup variables for the script 
variable dumpfreq         equal    1
variable restart          equal    0
variable neighborSkinDist equal    1.0 # distance for bins beyond force cut-off (1.0 = 1.0 Ang for units = real) 
variable baseFilename     universe Polymer

# == Setup the log file
#log         ${baseFilename}.LAMMPS_logFile

# == Setup style of the run

# type of units to use in the simulation (units used are in fact: amu, nm, ns, Kelvins)
units       nano

# indicates possible types allowed for interactions between the atoms
#atom_style  angle 
# Modified so only needs bonds
atom_style bond

# indicates possible types allowed for bonds between the atoms 
#bond_style hybrid harmonic 
bond_style fene
special_bonds fene


# indicates possible types allowed for bond angles between the atoms 
#angle_style hybrid harmonic

# indicates type of boundary conditions in each direction (p = periodic) 
boundary p p p 

read_data ${baseFilename}.LAMMPS_read_data # file of atomic coordinates and topology
velocity all zero linear                   # initialize all atomic velocities initially to zero

group polymers id 1:20
# == Compute stress tensor (MODIFIED PART STARTS HERE)
# Calculate per-atom stress using the Irving-Kirkwood formula
compute myTemp all temp
compute myStress all stress/atom NULL
compute polymerStress polymers stress/atom NULL
compute XZPolymerStress polymers reduce sum c_polymerStress[5]


variable xzStress equal c_XZPolymerStress
# == Add thermo style to output temperature 
#thermo 1
#thermo_style custom step temp press vol c_avgStress[1] c_avgStress[2] c_avgStress[3]

#thermo_style custom step temp press vol c_reducedStress[1] c_reducedStress[2] c_reducedStress[3] #after vol was changed 
# Ensure temperature, volume, and pressure are printed

run 0 
# == Green-Kubo relation setup for viscosity 
#variable pxy equal c_press[4] # Pxy component of the stress tensor (off-diagonal)

# Autocorrelate the stress tensor for Green-Kubo viscosity calculation
#fix correlator all ave/correlate 100 10 1000 v_pxy v_pxy type auto file autocorr.out ave running

# Setup Green-Kubo integration for viscosity 
#variable dt equal 1.0
#variable V equal vol  # Volume of the system
#variable kB equal 1.380649e-23  # Boltzmann constant in J/K
variable temp equal temp

# == Interactions 
pair_style lj/cut 2.5 
pair_coeff * * 1.0 1.0 1.12246
atom_modify sort 1000 ${neighborSkinDist}          # setup sort data explicitly since no interactions to set this data. 

# == Setup neighbor list distance
comm_style tiled
comm_modify mode single cutoff 202.0 vel yes

neighbor ${neighborSkinDist} bin                    # first number gives a distance beyond the force cut-off ${neighborSkinDist}
neigh_modify every 1

atom_modify sort 0 ${neighborSkinDist}           # setup sort data explicitly since no interactions to set this data. 

# == Setup the SELM integrator
fix 1 all selm Main.SELM_params


# note langevin just computes forces, nve integrates the motions
#fix 1 all langevin 298.15 298.15 0.00001 48279
#fix 2 all nve

fix printPolymerStress all print 10 "${xzStress}" file polymer_xz_stress_output.txt screen no
# Overall average temperature for the entire simulation
fix avg_temp all ave/time 1 1000 100000 v_temp file overall_avg_temp.txt mode scalar
fix printPolymerStress all print 10 "${xzStress}" file polymer_xz_stress_output.txt screen no


# == Setup output data write to disk
#dump        dmp_dcd all dcd ${dumpfreq} ${baseFilename}_LAMMPS_atomCoords.dcd
#dump_modify dmp_dcd unwrap yes                   # indicates for periodic domains that unwrapped coordinate should be given
dump        dmp_vtk all vtk ${dumpfreq} ./vtk/Particles_*.vtp id type vx fx c_myStress[1] c_myStress[2] c_myStress[3]
dump_modify dmp_vtk pad 8

# == Thermodynamic output
thermo_style custom step temp v_temp
thermo 1000

# == simulation time-stepping
timestep 0.01


"""

# feed commands to LAMMPs one line at a time
print_log("Sending commands to LAMMPs");
for line in s.splitlines():
  print_log(line);
  L.command(line);

# We can send collection of commands using the triple quote notation
s = """
# == Run the simulation
run      40000

# == Write restart data
write_restart ${baseFilename}.LAMMPS_restart_data
"""

# feed commands to LAMMPs one line at a time
print_log("Sending commands to LAMMPs");
for line in s.splitlines():
  print_log(line);
  L.command(line);


