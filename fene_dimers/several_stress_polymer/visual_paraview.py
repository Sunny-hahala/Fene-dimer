# state file generated using paraview version 5.8.0

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# trace generated using paraview version 5.8.0
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *

import glob;
import os;
import sys;
import pdb;

# @base_dir
#base_dir = './output/simulation_polymer4/fene_test003/'
#base_dir = './output/simulation_polymer4/harmonic_test008/'
base_dir = './output/Several_y_0_01/harmonic_test000/'
#base_dir = './output/several_polymer_y/harmonic_test000/'

print("base_dir = " + str(base_dir));

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [1365, 747]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.OrientationAxesLabelColor = [0.0, 0.0, 0.0]
renderView1.OrientationAxesOutlineColor = [0.49019607843137253, 0.49019607843137253, 0.49019607843137253]
renderView1.CenterOfRotation = [0.0, 1e-20, 0.0]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [0.0, -1355.156412055249, 0.0]
renderView1.CameraFocalPoint = [0.0, 1e-20, 0.0]
renderView1.CameraViewUp = [0.0, 0.0, 1.0]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 350.74028853269766
renderView1.Background = [1.0, 1.0, 1.0]
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

# init the 'GridAxes3DActor' selected for 'AxesGrid'
renderView1.AxesGrid.XTitle = 'x'
renderView1.AxesGrid.YTitle = 'y'
renderView1.AxesGrid.ZTitle = 'z'
renderView1.AxesGrid.XTitleColor = [0.0, 0.0, 0.0]
renderView1.AxesGrid.YTitleColor = [0.0, 0.0, 0.0]
renderView1.AxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
renderView1.AxesGrid.GridColor = [0.49019607843137253, 0.49019607843137253, 0.49019607843137253]
renderView1.AxesGrid.XLabelColor = [0.0, 0.0, 0.0]
renderView1.AxesGrid.YLabelColor = [0.0, 0.0, 0.0]
renderView1.AxesGrid.ZLabelColor = [0.0, 0.0, 0.0]
renderView1.AxesGrid.UseCustomBounds = 1
renderView1.AxesGrid.CustomBounds = [-0.15, 1.15, -0.15, 1.15, -0.6, 0.6]

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XML Unstructured Grid Reader'
path_cwd = os.getcwd();
#os.chdir(base_dir + '/vtk');
#file_list = sorted(glob.glob('Particles_????????.vtp'));
#pdb.set_trace();
file_list = sorted(glob.glob(base_dir + '/vtk/Particles_????????.vtp'));
#os.chdir(path_cwd);
print("Particles:");
print("file_list[0] = " + str(file_list[0]));
print("file_list[-1] = " + str(file_list[-1]));
print("len(file_list) = " + str(len(file_list)));

polymerBeads = XMLPolyDataReader(FileName=file_list);
RenameSource("polymer",polymerBeads);
polymerBeads.PointArrayStatus = ['id', 'type', 'vx', 'fx']

#seperate Polymer Bread and tracers
polymer_id=[1,10]
tracer_id=11

polymerThreshold = Threshold(Input=polymerBeads)
polymerThreshold.Scalars = ['POINTS', 'type']
polymerThreshold.LowerThreshold =1
polymerThreshold.UpperThreshold = 10
RenameSource("Polymer Beads", polymerThreshold)

# Create Threshold filter for Tracer Particles
tracerThreshold = Threshold(Input=polymerBeads)
tracerThreshold.Scalars = ['POINTS', 'type']
tracerThreshold.LowerThreshold =11
tracerThreshold.UpperThreshold =11
RenameSource("Tracer Particles", tracerThreshold)

#create Box frame
file_list = sorted(glob.glob(base_dir + '/vtk/Particles_????????_boundingBox.vtu'));
#os.chdir(path_cwd);
print("Bounding Box:");
print("file_list[0] = " + str(file_list[0]));
print("file_list[-1] = " + str(file_list[-1]));
print("len(file_list) = " + str(len(file_list)));
boundingBox1 = XMLUnstructuredGridReader(FileName=file_list);
RenameSource("bounding_box1",boundingBox1);

#glyph for polymer
glyph1 = Glyph(Input=polymerThreshold,GlyphType='Sphere')
glyph1.OrientationArray = ['POINTS', 'No orientation array']
glyph1.ScaleArray = ['POINTS', 'No scale array']
glyph1.ScaleFactor = 33.459614562988286
glyph1.GlyphTransform = 'Transform2'
glyph1.GlyphMode = 'All Points'
RenameSource("polymer",glyph1);
#glyph for tracers
glyph1 = Glyph(Input=tracerThreshold,GlyphType='Sphere')
glyph1.OrientationArray = ['POINTS', 'No orientation array']
glyph1.ScaleArray = ['POINTS', 'No scale array']
glyph1.ScaleFactor = 33.459614562988286
glyph1.GlyphTransform = 'Transform2'
glyph1.GlyphMode = 'All Points'
RenameSource("tracer",glyph1);

# === Draw bonds between beads of FENE dimers ===
bondsFilter1 = ProgrammableFilter(Input=polymerBeads)
RenameSource("fene_bonds", bondsFilter1)
bondsFilter1.OutputDataSetType = 'vtkMolecule'

# Script for bonding adjacent beads
bondsFilter1.Script = """
import numpy as np
from paraview.vtk.util.numpy_support import vtk_to_numpy

pdi = self.GetPolyDataInput()
pdo = self.GetMoleculeOutput()

num_points = pdi.GetNumberOfPoints()
types = vtk_to_numpy(pdi.GetPointData().GetArray("type"))
coords = np.array([pdi.GetPoint(i) for i in range(num_points)])

# Filter polymer beads with type from 1 to 10
polymer_indices = np.where((types >= 1) & (types <= 10))[0]
polymer_coords = coords[polymer_indices]

# Add atoms for polymer beads only
for i in range(len(polymer_coords)):
    x, y, z = polymer_coords[i]
    pdo.AppendAtom(1, x, y, z)

# Bond beads in pairs (assumes dimers)
threshold_sq = 170**2
for i in range(0, len(polymer_coords) - 1, 2):
    d2 = np.sum((polymer_coords[i] - polymer_coords[i + 1])**2)
    if d2 < threshold_sq:
        pdo.AppendBond(i, i + 1, 1)

"""

# Show all components
Show(polymerThreshold, renderView1)
Show(glyph1, renderView1)
feneBondsDisplay = Show(bondsFilter1, renderView1, 'PVMoleculeRepresentation')
feneBondsDisplay.Representation = 'Molecule'
feneBondsDisplay.BondRadius = 2.5
feneBondsDisplay.BondColor = [0.0, 0.2, 0.8]
Render()

"""import numpy as np
from paraview.vtk.util.numpy_support import vtk_to_numpy

pdi = self.GetPolyDataInput()
pdo = self.GetMoleculeOutput()

coords = np.array([pdi.GetPoint(i) for i in range(pdi.GetNumberOfPoints())])
types = vtk_to_numpy(pdi.GetPointData().GetArray("type"))

# Define simulation box
box_size = np.array([405.0, 405.0, 405.0])  # size
box_lo = np.array([-202.5, -202.5, -202.5])  # lower bounds

# Group by polymer type
polymer_pairs = {}
for i, t in enumerate(types):
    if 1 <= t <= 10:
        polymer_pairs.setdefault(t, []).append(coords[i])

threshold_sq = 400**2
atom_coords = []
bond_pairs = []

for t, pair in polymer_pairs.items():
    if len(pair) != 2:
        continue

    r1 = pair[0]
    r2 = pair[1]

    dr = r2 - r1
    dr_wrap = dr - box_size * np.round(dr / box_size)
    r2_image = r1 + dr_wrap

    d2 = np.sum(dr_wrap**2)
    if d2 > threshold_sq:
        continue

    # Use correct wrapping that works with negatives
    def wrap(coord):
        return np.mod(coord - box_lo, box_size) + box_lo

    r1_wrapped = wrap(r1)
    r2_wrapped = wrap(r2_image)

    idx1 = len(atom_coords)
    atom_coords.append(r1_wrapped)
    idx2 = len(atom_coords)
    atom_coords.append(r2_wrapped)

    bond_pairs.append((idx1, idx2))

# Add atoms and bonds
for x, y, z in atom_coords:
    pdo.AppendAtom(1, x, y, z)
for i, j in bond_pairs:
    pdo.AppendBond(i, j, 1)"""

