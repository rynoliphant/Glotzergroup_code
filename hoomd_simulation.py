import itertools
import math
import hoomd
import os

import freud
import gsd.hoomd
import matplotlib
import numpy

# %matplotlib inline
# matplotlib.style.use('ggplot')
import matplotlib_inline

matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

# fn = os.path.join(os.getcwd(), 'random.gsd')
# ![ -e "$fn" ] && rm "$fn"

import io
import warnings

import fresnel
import IPython
import numpy as np
import packaging.version
import PIL

device = fresnel.Device()
tracer = fresnel.tracer.Path(device=device, w=300, h=300)

FRESNEL_MIN_VERSION = packaging.version.parse('0.13.0')
FRESNEL_MAX_VERSION = packaging.version.parse('0.14.0')

#----- Shape Dictionaries -----
shape_vertices = {'cube': [
    [0.5, 0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, 0.5, -0.5],
    [0.5, -0.5, 0.5],
    [0.5, -0.5, -0.5],
    [0.5, 0.5, -0.5],
    [-0.5, -0.5, -0.5],
], 'octahedron': [
    (-0.5, 0, 0),
    (0.5, 0, 0),
    (0, -0.5, 0),
    (0, 0.5, 0),
    (0, 0, -0.5),
    (0, 0, 0.5),
], 'tetrahedron': [
    [0, 0, 0.5],
    [0, 0.7071, -0.5],
    [0.5*np.sqrt(3/2), -0.3535, -0.5],
    [-0.5*np.sqrt(3/2), -0.3535, -0.5],
], 'dodecahedron': [

]}

a_values = {'cube': 1,
            'octahedron': np.sqrt(2) * 0.5,
            'tetrahedron': np.sqrt(3/2)
}

#----- Inputs -----
import sys
inputs = sys.argv
#file.py | filename | density | N_particles | spacing | shape_name_1 | size_mult_1 | ratio | shape_name_2 | size_mult_2 |

filename = ''
density = 0.42
m = 2
N_particles = 2*m**3
spacing = 1.5
shape_names = ['tetrahedron']
size_mult = [1]
ratio = 1

vertices_1 = shape_vertices[shape_names[0]]
shape_d = 0.15
shape_a = 0.2
target_a = 0.2
seed = 9
sim_length = 1e6

if len(inputs) == 2:
    filename = inputs[1]
elif len(inputs) == 3:
    filename = inputs[1]
    density = float(inputs[2])
elif len(inputs) == 4:
    filename = inputs[1]
    density = float(inputs[2])
    N_particles = int(inputs[3])
elif len(inputs) == 5:
    filename = inputs[1]
    density = float(inputs[2])
    N_particles = int(inputs[3])
    spacing = float(inputs[4])
    shape_names = [inputs[5]]
elif len(inputs) == 6:
    filename = inputs[1]
    density = float(inputs[2])
    N_particles = int(inputs[3])
    spacing = float(inputs[4])
    shape_names = [inputs[5]]
    size_mult = float(inputs[6])
elif len(inputs) == 7:
    filename = inputs[1]
    density = float(inputs[2])
    N_particles = int(inputs[3])
    spacing = float(inputs[4])
    shape_names = [inputs[5]]
    size_mult = float(inputs[6])
elif len(inputs) > 6:
    filename = inputs[1]
    density = float(inputs[2])
    N_particles = int(inputs[3])
    spacing = float(inputs[4])
    shape_names = [inputs[5], inputs[8]]
    size_mult = [float(inputs[6]), float(inputs[9])]
    ratio = float(inputs[7])

#----- Volumes -----
if size_mult[0] != 1:
    try:
        a_values[shape_names[0]]
    except:
        print(f'{shape_names[0]} is not among the list of shapes available! :(')
    else:
        old_a_1 = a_values[shape_names[0]]
        a_values[shape_names[0]] = size_mult[0] * old_a_1
if len(size_mult) > 1:
    if size_mult[1] != 1:
        try:
            a_values[shape_names[1]]
        except:
            print(f'{shape_names[1]} is not among the list of shapes available! :(')
        else:
            old_a_2 = a_values[shape_names[1]]
            a_values[shape_names[1]] = size_mult[1] * old_a_2

shape_volumes = {'cube': a_values['cube']**3, 
                 'octahedron': 1/3 * np.sqrt(2) * a_values['octahedron']**3,
                 'tetrahedron': a_values['tetrahedron']**3 / (6*np.sqrt(2))
}

shape_surface_area = {'cube': 6*a_values['cube']**2,
                      'octahedron': 2*np.sqrt(3) * a_values['octahedron']**2,
                      'tetrahedron': np.sqrt(3) * a_values['tetrahedron']**2
}

#----- Integrator + Initial State -----
mc = hoomd.hpmc.integrate.ConvexPolyhedron()
mc.nselect = 2
cpu = hoomd.device.CPU()

K = math.ceil(N_particles**(1/3))
L = K*spacing
x = np.linspace(-L/2, L/2, K, endpoint=False)

if len(shape_names) == 1:
    mc.shape[shape_names[0]] = dict(vertices=vertices_1)

    mc.d[shape_names[0]] = shape_d
    mc.a[shape_names[0]] = shape_a

    shape_position = list(itertools.product(x, repeat=3))
    shape_position = shape_position[0:N_particles]

    shape_orientation = [(1,0,0,0)] * N_particles

    frame = gsd.hoomd.Frame()
    frame.particles.N = N_particles
    frame.particles.position = shape_position
    frame. particles.orientation = shape_orientation
    frame.particles.typeid = [0] * N_particles
    frame.particles.types = shape_names
    frame.particles.type_shapes = [dict(type = "ConvexPolyhedron", rounding_radius = 0.01, vertices = vertices_1)]
    frame.configuration.box = [L,L,L,0,0,0]

else:
    N_1 = int(N_particles*ratio)
    N_2 = N_particles - N_1

    #First Shape
    mc.shape[shape_names[0]] = dict(vertices=vertices_1)
    mc.d[shape_names[0]] = shape_d
    mc.a[shape_names[0]] = shape_a
    shape_position_1 = list(itertools.product(x, repeat=3))
    shape_position_1 = shape_position_1[0:N_1]
    shape_orientation_1 = [(1,0,0,0)] * N_1

    #Second Shape
    mc.shape[shape_names[1]] = dict(vertices=vertices_1)
    mc.d[shape_names[1]] = shape_d
    mc.a[shape_names[1]] = shape_a
    shape_position_2 = list(itertools.product(x, repeat=3))
    shape_position_2 = shape_position_2[N_1:N_2]
    shape_orientation_2 = [(1,0,0,0)] * N_2

    #Combine Shapes
    shape_position = shape_position_1 + shape_position_2
    shape_orientation = shape_orientation_1 + shape_orientation_2

    frame = gsd.hoomd.Frame()
    frame.particles.N = N_particles
    frame.particles.position = shape_position
    frame. particles.orientation = shape_orientation
    frame.particles.typeid = [0] * N_1 + [1] * N_2
    frame.particles.types = shape_names
    frame.configuration.box = [L,L,L,0,0,0]

with gsd.hoomd.open(name='initial_state_'+filename, mode='x') as f:
    f.append(frame)

simulation = hoomd.Simulation(device=cpu, seed=seed)
simulation.operations.integrator = mc
simulation.create_state_from_gsd(filename='initial_state_'+filename)

gsd_writer = hoomd.write.GSD(filename=filename, trigger=hoomd.trigger.Periodic(1000), mode='xb')
simulation.operations.writers.append(gsd_writer)

#----- Randomizing -----
simulation.run(10e3)
if mc.overlaps != 0:
    print('WARNING: Particles are overlapping!')
    print('Number of Overlaps:', mc.overlaps)

#----- Compressing -----
if len(shape_names) == 2:
    V_particle_1 = shape_volumes[shape_names[0]]
    V_particle_2 = shape_volumes[shape_names[1]]

    total_shape_volume = N_1*V_particle_1 + N_2*V_particle_2

else:
    V_particle = shape_volumes[shape_names[0]]
    total_shape_volume = N_particles * V_particle

initial_volume_fraction = (total_shape_volume/simulation.state.box.volume)
initial_box = simulation.state.box
final_box = hoomd.Box.from_box(initial_box)
final_box.volume = total_shape_volume / density
compress = hoomd.hpmc.update.QuickCompress(trigger=hoomd.trigger.Periodic(10), target_box=final_box)
simulation.operations.updaters.append(compress)

periodic = hoomd.trigger.Periodic(10)
tune = hoomd.hpmc.tune.MoveSize.scale_solver(moves=['a','d'], target=target_a, trigger=periodic, max_translation_move=target_a, max_rotation_move=target_a)
simulation.operations.tuners.append(tune)

while not compress.complete and simulation.timestep < 1e7:
    simulation.run(1000)

if not compress.complete:
    print('WARNING: Simulation did not successfully compress!')

#remove compress and tune
simulation.operations.remove(compress)
simulation.operations.remove(tune)

#----- Equilibrium -----
tune2 = hoomd.hpmc.tune.MoveSize.scale_solver(moves=['a','d'], target=target_a, trigger=hoomd.trigger.And([hoomd.trigger.Periodic(100), hoomd.trigger.Before(simulation.timestep + 5000)]))
simulation.operations.tuners.append(tune2)

simulation.run(sim_length + 5100)


gsd_writer.flush()