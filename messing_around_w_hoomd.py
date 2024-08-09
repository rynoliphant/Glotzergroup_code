import itertools
import math
import hoomd
import os

import freud
import gsd.hoomd
import matplotlib

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

def render_initial(position, orientation, L, vertices):
    if (
        'version' not in dir(fresnel)
        or packaging.version.parse(fresnel.version.version) < FRESNEL_MIN_VERSION
        or packaging.version.parse(fresnel.version.version) >= FRESNEL_MAX_VERSION
    ):
        warnings.warn(
            f'Unsupported fresnel version {fresnel.version.version} - expect errors.'
        )

    poly_info = fresnel.util.convex_polyhedron_from_vertices(vertices)

    scene = fresnel.Scene(device)
    geometry = fresnel.geometry.ConvexPolyhedron(scene, poly_info, N=len(position))
    geometry.material = fresnel.material.Material(
        color=fresnel.color.linear([0.01, 0.74, 0.26]), roughness=0.5
    )
    geometry.position[:] = position[:]
    geometry.orientation[:] = orientation[:]
    geometry.outline_width = 0.01
    fresnel.geometry.Box(scene, [L, L, L, 0, 0, 0], box_radius=0.02)

    scene.lights = [
        fresnel.light.Light(direction=(0, 0, 1), color=(0.8, 0.8, 0.8), theta=math.pi),
        fresnel.light.Light(
            direction=(1, 1, 1), color=(1.1, 1.1, 1.1), theta=math.pi / 3
        ),
    ]
    scene.camera = fresnel.camera.Orthographic(
        position=(L * 2, L, L * 2), look_at=(0, 0, 0), up=(0, 1, 0), height=L * 1.4 + 1
    )
    scene.background_color = (1, 1, 1)
    scene.background_alpha = 1
    return IPython.display.Image(tracer.sample(scene, samples=500)._repr_png_())

def render(snapshot, vertices):
    if (
        'version' not in dir(fresnel)
        or packaging.version.parse(fresnel.version.version) < FRESNEL_MIN_VERSION
        or packaging.version.parse(fresnel.version.version) >= FRESNEL_MAX_VERSION
    ):
        warnings.warn(
            f'Unsupported fresnel version {fresnel.version.version} - expect errors.'
        )
    L = snapshot.configuration.box[0]
    # vertices = [
    #     (-0.5, 0, 0),
    #     (0.5, 0, 0),
    #     (0, -0.5, 0),
    #     (0, 0.5, 0),
    #     (0, 0, -0.5),
    #     (0, 0, 0.5),
    # ]
    poly_info = fresnel.util.convex_polyhedron_from_vertices(vertices)

    scene = fresnel.Scene(device)
    geometry = fresnel.geometry.ConvexPolyhedron(
        scene, poly_info, N=snapshot.particles.N
    )
    geometry.material = fresnel.material.Material(
        color=fresnel.color.linear([0.01, 0.74, 0.26]), roughness=0.5
    )
    geometry.position[:] = snapshot.particles.position[:]
    geometry.orientation[:] = snapshot.particles.orientation[:]
    geometry.outline_width = 0.01
    fresnel.geometry.Box(scene, snapshot.configuration.box, box_radius=0.02)

    scene.lights = [
        fresnel.light.Light(direction=(0, 0, 1), color=(0.8, 0.8, 0.8), theta=math.pi),
        fresnel.light.Light(
            direction=(1, 1, 1), color=(1.1, 1.1, 1.1), theta=math.pi / 3
        ),
    ]
    scene.camera = fresnel.camera.Orthographic(
        position=(L * 2, L, L * 2), look_at=(0, 0, 0), up=(0, 1, 0), height=L * 1.4 + 1
    )
    scene.background_alpha = 1
    scene.background_color = (1, 1, 1)
    return IPython.display.Image(tracer.sample(scene, samples=500)._repr_png_())

def render_movie(frames, particles=None, is_solid=None):
    if is_solid is None:
        is_solid = [None] * len(frames)
    a = render(frames[0], particles, is_solid[0])

    im0 = PIL.Image.fromarray(a[:, :, 0:3], mode='RGB').convert(
        'P', palette=PIL.Image.Palette.ADAPTIVE
    )
    ims = []
    for i, f in enumerate(frames[1:]):
        a = render(f, particles, is_solid[i])
        im = PIL.Image.fromarray(a[:, :, 0:3], mode='RGB')
        im_p = im.quantize(palette=im0)
        ims.append(im_p)

    blank = np.ones(shape=(im.height, im.width, 3), dtype=np.uint8) * 255
    im = PIL.Image.fromarray(blank, mode='RGB')
    im_p = im.quantize(palette=im0)
    ims.append(im_p)

    f = io.BytesIO()
    im0.save(f, 'gif', save_all=True, append_images=ims, duration=1000, loop=0)

    size = len(f.getbuffer()) / 1024
    if size > 3000:
        warnings.warn(f'Large GIF: {size} KiB')
    return IPython.display.display(IPython.display.Image(data=f.getvalue()))

#Integrator
cube_vertices = [
    (0.5, 0.5, 0.5),
    (-0.5, 0.5, 0.5),
    (-0.5, -0.5, 0.5),
    (-0.5, 0.5, -0.5),
    (0.5, -0.5, 0.5),
    (0.5, -0.5, -0.5),
    (0.5, 0.5, -0.5),
    (-0.5, -0.5, -0.5),
]

mc = hoomd.hpmc.integrate.ConvexPolyhedron()
mc.shape['cube'] = dict(vertices=cube_vertices)

mc.nselect = 2
mc.d['cube'] = 0.15
mc.a['cube'] = 0.2

#Initial State
m = 4
N_particles = 2*m**3

spacing = 1.5
K = math.ceil(N_particles**(1/3))
L = K*spacing

x = np.linspace(-L/2, L/2, K, endpoint=False)
position = list(itertools.product(x, repeat=3))
position = position[0:N_particles]

orientation = [(1,0,0,0)] * N_particles

render_initial(position, orientation, L, cube_vertices)

frame = gsd.hoomd.Frame()
frame.particles.N = N_particles
frame.particles.position = position
frame.particles.orientation = orientation
frame.particles.typeid = [0] * N_particles
frame.particles.types = ['cube']
frame.configuration.box = [L,L,L,0,0,0]

with gsd.hoomd.open(name='Glotzergroup/lattice.gsd', mode='x') as f:
    f.append(frame)

# print([0]*10 + [1]*5)

cpu = hoomd.device.CPU()
simulation = hoomd.Simulation(device=cpu)
simulation.create_state_from_gsd(filename='lattice.gsd')

initial_snapshot = simulation.state.get_snapshot()
simulation.run(10e3)
mc.overlaps
final_snapshot = simulation.state.get_snapshot()
render(final_snapshot)