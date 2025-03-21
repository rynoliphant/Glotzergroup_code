import ase 
import coxeter
import ebonds
import freud
import gsd
import gsd.hoomd
import numpy as np

filename = 'hello.gsd'
frame = -10

cube_vertices = [
    #square
    # [0.5, 0.5],
    # [-0.5, 0.5],
    # [-0.5, -0.5],
    # [0.5, -0.5],

    #triangle
    # [0, 0.5],
    # [-0.25*np.sqrt(3), -0.25],
    # [0.25*np.sqrt(3), -0.25],

    #pentagon
    # [0.5*np.cos(-0.9424777961), 0.5*np.sin(-0.9424777961)],
    # [0.5*np.cos(0.3141592654), 0.5*np.sin(0.3141592654)],
    # [0.0, 0.5],
    # [-0.5*np.cos(0.3141592654), 0.5*np.sin(0.3141592654)],
    # [-0.5*np.cos(-0.9424777961), 0.5*np.sin(-0.9424777961)],

    #cube
    # [-0.5, 0, 0],
    # [0.5, 0, 0],
    # [0, -0.5, 0],
    # [0, 0.5, 0],
    # [0, 0, -0.5],
    # [0, 0, 0.5],

    #dodecahedron
    # [0.5, 0.5, 0.5],
    # [-0.5, 0.5, 0.5],
    # [-0.5, -0.5, 0.5],
    # [-0.5, 0.5, -0.5],
    # [0.5, -0.5, 0.5],
    # [0.5, -0.5, -0.5],
    # [0.5, 0.5, -0.5],
    # [-0.5, -0.5, -0.5],
    # [0, 0.5/((1+np.sqrt(5))*0.5), 0.5*((1+np.sqrt(5))*0.5)],
    # [0, -0.5/((1+np.sqrt(5))*0.5), 0.5*((1+np.sqrt(5))*0.5)],
    # [0, 0.5/((1+np.sqrt(5))*0.5), -0.5*((1+np.sqrt(5))*0.5)],
    # [0, -0.5/((1+np.sqrt(5))*0.5), -0.5*((1+np.sqrt(5))*0.5)],
    # [0.5/((1+np.sqrt(5))*0.5), 0.5*((1+np.sqrt(5))*0.5), 0],
    # [-0.5/((1+np.sqrt(5))*0.5), 0.5*((1+np.sqrt(5))*0.5), 0],
    # [0.5/((1+np.sqrt(5))*0.5), -0.5*((1+np.sqrt(5))*0.5), 0],
    # [-0.5/((1+np.sqrt(5))*0.5), -0.5*((1+np.sqrt(5))*0.5), 0],
    # [0.5*((1+np.sqrt(5))*0.5), 0, 0.5/((1+np.sqrt(5))*0.5)],
    # [-0.5*((1+np.sqrt(5))*0.5), 0, 0.5/((1+np.sqrt(5))*0.5)],
    # [0.5*((1+np.sqrt(5))*0.5), 0, -0.5/((1+np.sqrt(5))*0.5)],
    # [-0.5*((1+np.sqrt(5))*0.5), 0, -0.5/((1+np.sqrt(5))*0.5)],

    #octahedron
    (0.5, 0.5, 0.5),
    (-0.5, 0.5, 0.5),
    (-0.5, -0.5, 0.5),
    (-0.5, 0.5, -0.5),
    (0.5, -0.5, 0.5),
    (0.5, -0.5, -0.5),
    (0.5, 0.5, -0.5),
    (-0.5, -0.5, -0.5),
]

data_hpmc = gsd.hoomd.open(filename, mode='r')

orientations = data_hpmc[frame].particles.orientation
positions = data_hpmc[frame].particles.position
types = data_hpmc[frame].particles.types
typeid_list = data_hpmc[frame].particles.typeid
box = data_hpmc[frame].configuration.box

N_atoms = len(orientations)

box_tmp = freud.box.Box(Lx=box[0], Ly=box[1], Lz=box[2], xy=box[3], xz=box[4], yz=box[5])
shift_mat = np.transpose(box_tmp.to_matrix())
positions = positions + shift_mat[0,:]*0.5 + shift_mat[1,:]*0.5 + shift_mat[2,:]*0.5

atoms = ase.Atoms('C'+str(N_atoms), positions=positions,cell=shift_mat,pbc=True)

system = ebonds.data.System.from_ase_atoms(atoms)
system.particles.types = types
system.particles.type_shapes = [dict(vertices = cube_vertices)]
system.particles.typeid = typeid_list
system.particles.orientation = orientations
system.particles.position
# system = ebonds.data.System.from_gsd(filename, frame_idx=frame)

system.validate


solver = ebonds.solvers.Solver(system, phi_inp=0.6, grid_spacing=0.2, sphere_q=False)

box = system.configuration.box
class var:
    def __init__(self, box):
        self.a = 0.5*box.a
        self.b = 0.5*box.b
        self.c = 0.5*box.c
        self.Lx = 0.5*box.Lx
        self.Ly = 0.5*box.Ly
        self.Lz = 0.5*box.Lz

#small_box.a, small_box.b, small_box.c, small_box.Lx, small_box.Ly, small_box.Lz

solver.solve(mu=3, rshift=0.3, r_exponent=2, cpu_cores=3, small_system_q = False, small_box = var(box))
energy = solver.energy
print('Energy:',energy)
#----- Define System -----
#sys_n (system name?)
#a
#b
#c

#alpha
#beta
#gamma

#nrep

#basis


#----- Create Periodic Box -----
#cx
#cy
#cz

#a1
#a2
#a3

#mat
#box

#uc
#box, positions

#----- Generate System for EBT Calc -----
#box1, pos1 = uc.generate_system(nrep)

#N_particles

#np.savetxt()
