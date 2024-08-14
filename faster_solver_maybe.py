import numpy as np
import time
from scipy.optimize import LinearConstraint, OptimizeResult, minimize
import coxeter
import rowan
import awkward as ak

#----- messing around -----
arr = np.array([
    [1,200,3.37521],
    [3,5,7]
])
tile = np.tile(arr, (3,1))
print(tile)
repeat = np.repeat(arr, 3, axis=0)
print(repeat)
print(ak.sum(ak.Array(arr * ak.Array([1,2,2])), axis=1, highlevel=False))
variable = ak.to_list(ak.sum(arr * np.array([1,2,2]), axis=1))
print(variable)
variable_ak = ak.Array([variable])
print(variable_ak)
print(ak.to_numpy(variable_ak))
print(ak.Array([2, 3.521987, 94.19348]))

test = ak.Array([
    [[1,2,3], [4,5,6],[7,8,9]],
    [[9,4,5]],
    [[3,7,5], [8,5,2]]
])

test_broadcast = ak.broadcast_fields(np.array([2,2.5672,9.2538]), test)
print(test_broadcast)
test0 = test*ak.Array([2,2.5672,9.2538])
print(test0[0])
test_bool = test == ak.Array([1,2,3])
print(test[test_bool])

up_test = ak.Array([
    [15, 30, 42],
    [37],
    [30, 28]
])
print(test0 <= up_test)

con_col1 = ak.Array([
    [[1,2,3]],
    [[9,4,5]],
    [[2,7,5]]
])
con_col2 = ak.Array([
    [[4,5,6], [7,8,9], [3,5,7]],
    [[2,4,1], [], []],
    [[], [8,5,2], [6,9,2]]
])

con_col3 = ak.Array([
    [[], []],
    [[2,5,0], []],
    [[1,0,6], [8,6,9]]
])

con_mat = ak.concatenate((con_col1, con_col2, con_col3), axis=1)
con_counts = ak.num(con_mat, axis=2)
con_counts = ak.flatten(con_counts[con_counts != 0])
con_mat = ak.flatten(con_mat, axis=2)
con_mat = ak.unflatten(con_mat, con_counts, axis=1)
print(con_mat[0])
print(con_mat[1])
print(con_mat[2])

none_ak = ak.Array([
    [[], [], []]
])

none_counts = ak.num(none_ak, axis=2)
none_counts = ak.flatten(none_counts[none_counts != 0])

none_ak = ak.flatten(none_ak, axis=2)
none_ak = ak.unflatten(none_ak, none_counts, axis=1)
print(none_ak)

#

start = time.time()

first_list = []
for i in (-1, 0, 1):
    for j in (-1, 0, 1):
        for k in (-1, 0, 1):
            first_list.append((i, j, k))
            # print( i, j, k)

middle = time.time()

ijk_list = np.array([
    (-1, -1, -1), (-1, -1, 0), (-1, -1, 1), 
    (-1, 0, -1), (-1, 0, 0), (-1, 0, 1), 
    (-1, 1, -1), (-1, 1, 0), (-1, 1, 1), 
    (0, -1, -1), (0, -1, 0), (0, -1, 1), 
    (0, 0, -1), (0, 0, 0), (0, 0, 1), 
    (0, 1, -1), (0, 1, 0), (0, 1, 1), 
    (1, -1, -1), (1, -1, 0), (1, -1, 1), 
    (1, 0, -1), (1, 0, 0), (1, 0, 1), 
    (1, 1, -1), (1, 1, 0), (1, 1, 1)
])
second_list = []
for ijk in ijk_list:
    second_list.append(ijk)
    # print(ijk[0], ijk[1], ijk[2])

end = time.time()

print('First time:', middle-start)
print('Second time:', end-middle)

print('Difference in time:', (middle-start)/(end-middle))

def distance_point_point (pt1, pt2):
    v = pt2 - pt1
    return np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def distance_point_point_multiple (pts1, pts2):
    v = pts2 - pts1
    # print(v)
    output = np.sqrt(v[:,0]**2 + v[:,1]**2 + v[:,2]**2)
    # print(output)
    return output

def distance_point_point_new_maybe (pt1, pt2):
    if len(pt1) > 3 and len(pt1)%3 == 0:
        pts1 = pt1.reshape((int(len(pt1)/3), 3))
        pts2 = pt2.reshape((len(pts1),3))
        v = pts2 - pts1
        output = np.min(np.sqrt(v[:,0]**2 + v[:,1]**2 + v[:,2]**2))
    else:
        v = pt2 - pt1
        output = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

    return output

points1 = np.array([
    [0,0,0],
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [1,1,0],
    [1,0,1],
    [0,1,1],
    [1,1,1],
    [2,1,1],
    [1,2,1],
    [1,1,2],
    [2,2,1],
    [2,1,2],
    [1,2,2],
    [2,2,2],
    [3,2,2],
    [3,3,2],
    [3,3,3],
    [4,3,3],
    [4,4,3],
    [4,4,4],
    [5,4,4],
    [5,5,4],
    [5,5,5],
    [6,5,5],
    [6,6,5],
    [6,6,6]

])


shape_pos = np.array([
    [2,2,2],
    [4,4,3],
    # [2,2,2],
    # [4,4,3],
    # [2,2,2],
    # [4,4,3],
    # [2,2,2],
    # [4,4,3],
    # [2,2,2],
    # [4,4,3],
    # [2,2,2],
    # [4,4,3],
])

vertices = [
    (0.5, 0.5, 0.5),
    (-0.5, 0.5, 0.5),
    (-0.5, -0.5, 0.5),
    (-0.5, 0.5, -0.5),
    (0.5, -0.5, 0.5),
    (0.5, -0.5, -0.5),
    (0.5, 0.5, -0.5),
    (-0.5, -0.5, -0.5),
]

orientations = [(1,0,0,0), (0.3647046, 0.1159188, -0.2798528, 0.8804747)]

shapes = []
new_halfspaces = []
for i, pos in enumerate(shape_pos):
    rot_vert = rowan.rotate(orientations[i], vertices)
    tmpshape = coxeter.shapes.ConvexPolyhedron(vertices=rot_vert)

    normals = tmpshape.normals

    halfspace = []
    image_diff = [[6,0,0], [0,0,0], [-6,0,0]]
    for image in image_diff:
        position = pos + image
        upper_bounds = normals.dot(position) + 0.5
        smile = np.block([normals, upper_bounds.reshape((6,1))])

        halfspace.append(smile)

    new_halfspaces.append(halfspace)

    shapes.append(tmpshape)
new_halfspaces = np.asarray(new_halfspaces)

# new_halfspaces = np.array([
#     [[[0,0,1, 2.5],[0,0,-1, -1.5],[1,0,0, 8.5],
#       [-1,0,0, -7.5],[0,1,0, 2.5],[0,-1,0, -1.5]],
#     [[0,0,1, 2.5],[0,0,-1, -1.5],[1,0,0, 2.5],
#      [-1,0,0, -1.5],[0,1,0, 2.5],[0,-1,0, -1.5]],
#     [[0,0,1, 2.5],[0,0,-1, -1.5],[1,0,0, -3.5],
#      [-1,0,0, 4.5],[0,1,0, 2.5],[0,-1,0, -1.5]]],


#     [[[ 1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3),  10.315],
#       [-1/np.sqrt(3),-1/np.sqrt(3),-1/np.sqrt(3), -9.315],
#       [ 1/np.sqrt(3), 1/np.sqrt(3),-1/np.sqrt(3),  6.851],
#       [-1/np.sqrt(3),-1/np.sqrt(3), 1/np.sqrt(3), -5.851],
#       [ 1/np.sqrt(2),-1/np.sqrt(2),            0,  4.743],
#       [-1/np.sqrt(2), 1/np.sqrt(2),            0, -3.743]],

#     [ [ 1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3),  6.851],
#       [-1/np.sqrt(3),-1/np.sqrt(3),-1/np.sqrt(3), -5.851],
#       [ 1/np.sqrt(3), 1/np.sqrt(3),-1/np.sqrt(3),  3.387],
#       [-1/np.sqrt(3),-1/np.sqrt(3), 1/np.sqrt(3), -2.387],
#       [ 1/np.sqrt(2),-1/np.sqrt(2),            0,  0.5],
#       [-1/np.sqrt(2), 1/np.sqrt(2),            0,  0.5]],

#     [ [ 1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3),  3.387],
#       [-1/np.sqrt(3),-1/np.sqrt(3),-1/np.sqrt(3), -2.387],
#       [ 1/np.sqrt(3), 1/np.sqrt(3),-1/np.sqrt(3), -0.207],
#       [-1/np.sqrt(3),-1/np.sqrt(3), 1/np.sqrt(3),  1.207],
#       [ 1/np.sqrt(2),-1/np.sqrt(2),            0, -3.743],
#       [-1/np.sqrt(2), 1/np.sqrt(2),            0,  4.743]]],
# ])

print(new_halfspaces.shape)

# halfspaces_ub = new_halfspaces[0][:,-1]
# halfspaces_values = []
# for i in new_halfspaces:
#     halfspaces_values.append(i[:,:-1].flatten())
# halfspaces_values = np.array(halfspaces_values)
# print(halfspaces_values)

# constraint = LinearConstraint(new_halfspaces[0], -np.inf, 6)
# # print(constraint.residual(3))
# other_constraint = (-np.inf<=new_halfspaces.dot(3))*(new_halfspaces.dot(3)<=6)
# print((-np.inf<=new_halfspaces.dot(3))*(new_halfspaces.dot(3)<=6))

#----- Current Min Dist Calculations -----
current_start = time.time()
r_list = []
for s, x in enumerate(shape_pos):
    r = []
    for i in range(len(points1)):
        min_list = []
        x_list = []
        for n in new_halfspaces[s]:
            current_min = minimize(
                fun=distance_point_point,
                x0=np.zeros(3),
                args=(points1[i],),
                constraints=[LinearConstraint(n[:,:-1], -np.inf, n[:,-1])]
            )
            # print(n[:,:-1].dot(current_min.x) <= n[:,-1])
            min_list.append(current_min.fun)
            # x_list.append(current_min.x)
        # print('min',np.asarray(min_list))
        # print('x',np.asarray(x_list))
        r.append(np.min(min_list))
    r_list.append(np.asarray(r))

r_list = np.asarray(r_list)
# print(r_list.shape)
current_end = time.time()

def hell_if_i_know (pts1, pts2):
    diff = pts2 - pts1
    mag = np.sqrt(np.sum(diff**2))
    return mag

def uhhh (pts1, pt2):
    width = int(len(pts1)/3)
    pts1 = pts1.reshape((width, 3))
    pts2 = np.tile(pt2, width).reshape((width, 3))

    diff = pts2 - pts1
    mag = np.sqrt(np.sum(diff**2, axis=1))
    return mag

def why_not (q, A, a):
    result = A.dot(q)
    diff = a - result
    mag = np.sqrt(np.sum(diff**2))
    return mag

n_faces = len(new_halfspaces[0][0])
n_shapes = len(shape_pos)
n_constraints = 3
new_mega_ub = []
for zeros_l in range(n_shapes*n_constraints):
    shape_ind = zeros_l//n_constraints
    constraint_ind = zeros_l%n_constraints
    zeros_w = n_shapes*n_constraints-zeros_l-1

    if zeros_l == 0:
        block = np.block([
            [new_halfspaces[shape_ind][constraint_ind,:,:-1], np.zeros((n_faces, zeros_w*3))]
        ])
    elif zeros_l == (n_shapes*3)-1:
        block = np.block([
            [new_mega_matrix],
            [np.zeros((n_faces, zeros_l*3)), new_halfspaces[shape_ind][constraint_ind,:,:-1]]
        ])
    else:
        block = np.block([
            [new_mega_matrix],
            [np.zeros((n_faces, zeros_l*3)), new_halfspaces[shape_ind][constraint_ind,:,:-1],np.zeros((n_faces, zeros_w*3))]
        ])
    new_mega_matrix = block
    new_mega_ub.append(new_halfspaces[shape_ind][constraint_ind,:,-1])

new_mega_ub = np.asarray(new_mega_ub).flatten()
print(new_mega_matrix.shape)
print(new_mega_ub.shape)

#----- New Min Dist Calculations -----
new_start = time.time()
r_list_new = []

for i in range(len(points1)):
    xp_min = minimize(
        fun=hell_if_i_know,
        x0=np.zeros(n_shapes*n_constraints*3),
        args=(np.tile(points1[i],(n_shapes*n_constraints))),
        constraints=[LinearConstraint(new_mega_matrix, -np.inf, new_mega_ub)]
    )
    xp_min_list = uhhh(xp_min.x, points1[i])

    find_min = np.min(xp_min_list.reshape((n_shapes, n_constraints)), axis=1)

    r_list_new.append(find_min)

r_list_new = np.asarray(r_list_new)
r_list_new = np.transpose(r_list_new)
new_end = time.time()

#----- Other New Dist Calculations -----
other_start = time.time()
other_r_list = []
for s, x in enumerate(shape_pos):
    other_r = []
    for i in range(len(points1)):
        # other_mega_matrix = np.block([
        #     [new_halfspaces[s][0,:,:-1], np.zeros((n_faces, 6))],
        #     [np.zeros((n_faces,3)),new_halfspaces[s][1,:,:-1],np.zeros((n_faces,3))],
        #     [np.zeros((n_faces,6)),new_halfspaces[s][2,:,:-1]]
        # ])

        # other_mega_ub = np.array([new_halfspaces[s][0,:,-1], new_halfspaces[s][1,:,-1], new_halfspaces[s][2,:,-1]]).flatten()
        images_pos = (x+np.array([6,0,0]), x, x+np.array([-6,0,0]))

        dist_images_list = np.sqrt(np.sum((images_pos - points1[i])**2, axis=1))

        con_ind = np.argmin(dist_images_list)

        other_min = minimize(
            fun=distance_point_point,
            x0=np.zeros(3),
            args=(points1[i],),
            constraints=[LinearConstraint(new_halfspaces[s][con_ind,:,:-1],-np.inf,new_halfspaces[s][con_ind,:,-1])]
        )

        other_min_list = uhhh(other_min.x, points1[i])
        #     # print(n[:,:-1].dot(current_min.x) <= n[:,-1])
        #     min_list.append(current_min.fun)
        #     # x_list.append(current_min.x)
        # # print('min',np.asarray(min_list))
        # # print('x',np.asarray(x_list))
        other_r.append(np.min(other_min_list))
    other_r_list.append(np.asarray(other_r))

other_r_list = np.asarray(other_r_list)
# print(r_list.shape)
other_end = time.time()


def get_edge_face_neighbors (shape):
    faces = ak.Array(shape.faces)
    face_inds = np.arange(0, shape.num_faces, 1)

    edge_face_neighbors = []
    for edge in shape.edges:
          edge1_bool = faces == edge[0]
          edge2_bool = faces == edge[1]

          check_edge = ak.sum(edge1_bool + edge2_bool, axis=1) == 2

          neighbors = face_inds[check_edge]

          sum_edges = edge1_bool + edge2_bool
          face_verts = faces[sum_edges * check_edge]
          flat_face_verts = ak.ravel(face_verts)
          reshape_face_verts = np.array(flat_face_verts).reshape((2,2))
          # print(reshape_face_verts)

          face_verts_bool = (reshape_face_verts != edge)[:,0] +1 -1

          face_all_verts0 = np.array(faces[check_edge][:,0]).reshape((2,1))
          face_all_verts_last = np.array(faces[check_edge][:,-1]).reshape((2,1))
          first_last_verts = np.block([face_all_verts0, face_all_verts_last])
          # print(first_last_verts)

          ind_edge_bool = (np.sum(first_last_verts == edge, axis=1) == 2) +1 -1

          switch_neighbor_bool = face_verts_bool + ind_edge_bool
          # print('edge', edge)
          # print('switch bool', switch_neighbor_bool)

          # print('previous neighbor', neighbors)
          neighbors = neighbors[switch_neighbor_bool]
          # print('new neighbor', neighbors)

          edge_face_neighbors.append(neighbors)

    return np.asarray(edge_face_neighbors)

bc_start = time.time()

for s,x in enumerate(shape_pos):
    #TODO: normalize all constraints so that they are unit vectors!!!

    shp = shapes[s]

    shp_faces = shp.normals
    shp_vert = shp.vertices
    shp_edges = shp.edge_vectors

    shp_edge_vert = shp.edges
    shp_edge_face = get_edge_face_neighbors(shp)

    # faces = np.asarray(shp.faces)
    # print(shp_edge_vert)
    # print(shp_edge_face)
    # print(faces)

    #----- Building the Edge Sections (Constraints & Boundary Conditions) -----
    new_edge_constraint = np.zeros((len(shp_edges), 4, 3))
    new_edge_bounds = np.zeros((len(shp_edges), 4))

    edge_constraint_col_1 = -1*shp_edges
    edge_constraint_col_2 = shp_edges
    edge_constraint_col_3 = -1*np.cross(shp_edges, shp_faces[shp_edge_face[:,1]])
    edge_constraint_col_4 = np.cross(shp_edges, shp_faces[shp_edge_face[:,0]])

    new_edge_constraint[:,0] = edge_constraint_col_1
    new_edge_constraint[:,1] = edge_constraint_col_2
    new_edge_constraint[:,2] = edge_constraint_col_3
    new_edge_constraint[:,3] = edge_constraint_col_4


    new_edge_verts = np.zeros((len(shp_edges), 2, 3))
    new_edge_verts[:,0] = shp_vert[shp_edge_vert[:,0]] + x
    new_edge_verts[:,1] = shp_vert[shp_edge_vert[:,1]] + x

    edge_bounds_1 = np.sum(new_edge_constraint[:,0] *(new_edge_verts[:,1]), axis=1)
    edge_bounds_2 = np.sum(new_edge_constraint[:,1] *(new_edge_verts[:,0]), axis=1)
    edge_bounds_3 = np.sum(new_edge_constraint[:,2] *(new_edge_verts[:,0]), axis=1)
    edge_bounds_4 = np.sum(new_edge_constraint[:,3] *(new_edge_verts[:,0]), axis=1)

    new_edge_bounds[:,0] = edge_bounds_1
    new_edge_bounds[:,1] = edge_bounds_2
    new_edge_bounds[:,2] = edge_bounds_3
    new_edge_bounds[:,3] = edge_bounds_4

    # print(new_edge_constraint.dot(np.array([2,3,3])) >= new_edge_bounds)
    #constraint matrix normals are pointing inwards, so use lower bounds


    #----- Building the Face Sections (Constraints & Boundary Conditions) -----
    n_faces = shp.num_faces #number of faces
    n_edges = shp.num_edges #number of edges
    nfaces_tile_edges = np.tile(shp_edges, (n_faces, 1)).reshape((n_faces, n_edges, 3)) #Edge vectors tiled
    nfaces_tile_efneighbors = np.tile(shp_edge_face, (n_faces, 1)) #Tiling shp_edge_face so that it is now shape (n_faces, n_edges, 2)
    efneighbors0 = nfaces_tile_efneighbors[:,0].reshape((n_faces, n_edges)) #breaking nfaces_tile_efneighbors into the first column [0]
    efneighbors1 = nfaces_tile_efneighbors[:,1].reshape((n_faces, n_edges)) #second column [1] of nfaces_tile_efneighbors
    faces_inds = np.arange(0, n_faces, 1).reshape((n_faces, 1)) #Making an array containing the indices of the faces

    # print('ef',efneighbors0)
    # print('face inds',faces_inds)

    # print('ef bool',efneighbors0 == faces_inds)
    efbool0 = ak.from_numpy(efneighbors0 == faces_inds) #Making a bool to find the edges that are next to each face
    efbool1 = ak.from_numpy(efneighbors1 == faces_inds) #^---
    ak_tile_edges = ak.from_numpy(nfaces_tile_edges) 
    # print('corresponding edges', ak.mask(ak_tile_edges, efbool0))
    # print('length', len(ak.mask(ak_tile_edges, efbool0)))

    ak_edges_mask0 = ak.to_numpy(ak.mask(ak_tile_edges, efbool0)) #Applying the bool: efbool0 to ak_tile_edges, but returning Nones for False instead of removing them
    ak_edges_mask1 = ak.to_numpy(ak.mask(ak_tile_edges, efbool1))#^---
    faces_repeat = np.repeat(shp_faces, n_edges, axis=0).reshape((n_faces, n_edges, 3)) 

    neg_constraints = ak.from_numpy(-1*np.cross(ak_edges_mask0, faces_repeat)) #Doing the cross product of the edges to the faces
    neg_constraints = ak.drop_none(ak.nan_to_none(neg_constraints)) #changing the NaN back to None, and then dropping the None
    neg_counts = ak.num(neg_constraints, axis=2)
    neg_counts = ak.flatten(neg_counts[neg_counts != 0])
    neg_constraints = ak.flatten(neg_constraints, axis=2)
    neg_constraints = ak.unflatten(neg_constraints, neg_counts, axis=1)
    neg_con_count = ak.num(neg_constraints, axis=1)
    # print(neg_con_count)

    pos_constraints = ak.from_numpy(np.cross(ak_edges_mask1, faces_repeat))
    pos_constraints = ak.drop_none(ak.nan_to_none(pos_constraints))
    pos_counts = ak.num(pos_constraints, axis=2)
    pos_counts = ak.flatten(pos_counts[pos_counts != 0])
    pos_constraints = ak.flatten(pos_constraints, axis=2)
    pos_constraints = ak.unflatten(pos_constraints, pos_counts, axis=1)
    pos_con_count = ak.num(pos_constraints, axis=1)
    # print(pos_con_count)

    face_normals = shp_faces.reshape((n_faces, 1, 3))
    
    new_face_constraints = ak.concatenate((face_normals, neg_constraints, pos_constraints), axis=1)
    face_con_counts = ak.num(new_face_constraints, axis=2)
    face_con_counts = ak.flatten(face_con_counts[face_con_counts != 0])
    new_face_constraints = ak.flatten(new_face_constraints, axis=2)
    new_face_constraints = ak.unflatten(new_face_constraints, face_con_counts, axis=1)
    # print(new_face_constraints[0])

    #bounds
    nfaces_tile_evneighbors = np.tile(shp_edge_vert, (n_faces, 1)).reshape((n_faces, n_edges, 2))
    neg_face_verts = ak.from_numpy(shp_vert[nfaces_tile_evneighbors[efbool0][:,0]] + x)
    neg_face_verts = ak.unflatten(neg_face_verts, neg_con_count, axis=0)

    pos_face_verts = ak.from_numpy(shp_vert[nfaces_tile_evneighbors[efbool1][:,1]] + x)
    pos_face_verts = ak.unflatten(pos_face_verts, pos_con_count, axis=0)

    neg_bounds = ak.sum(neg_constraints * neg_face_verts, axis=2)
    pos_bounds = ak.sum(pos_constraints * pos_face_verts, axis=2)
    norm_distances = (shp.face_centroids + shp.centroid + x).reshape((n_faces, 1, 3))
    normal_bounds = np.sum(face_normals*norm_distances, axis=2)
    
    # print(neg_bounds[1])
    # print(pos_bounds[1])
    new_face_bounds = ak.concatenate((normal_bounds, neg_bounds, pos_bounds), axis=1)
    # print(new_face_bounds)



    #TODO: add np.pad so that it works for shapes with faces of varying number of edges
      #don't use np.pad, instead use akward arrays (assuming that you can multiply them with [x,y,z] point)
    # face_constraint = []
    # face_bounds = []
    # for face_i, face in enumerate(shp_faces):
    #     edge_face_bool = shp_edge_face == face_i
    #     bool_0 = edge_face_bool[:,0]
    #     bool_1 = edge_face_bool[:,1]

    #     pos_con = np.cross(shp_edges[edge_face_bool[:,1]], face)
    #     neg_con = -1*np.cross(shp_edges[edge_face_bool[:,0]], face)

    #     # print(face.reshape((1,3)).shape)
    #     # print(neg_con.shape)
    #     # print(pos_con.shape)
    #     face_constr = np.concatenate((
    #         face.reshape((1,3)),
    #         neg_con,
    #         pos_con
    #     ), axis=0)

    #     face_vert_pos = shp_vert[shp_edge_vert[bool_1][:,1]] + x
    #     face_vert_neg = shp_vert[shp_edge_vert[bool_0][:,0]] + x


    #     neg_bound = np.sum(neg_con * face_vert_neg, axis=1)
    #     pos_bound = np.sum(pos_con * face_vert_pos, axis=1)

    #     face_bound = np.hstack((
    #         np.array(face.dot(x + shp.centroid + shp.face_centroids[face_i])),
    #         np.sum(neg_con * face_vert_neg, axis=1),
    #         np.sum(pos_con * face_vert_pos, axis=1)
    #     ))

    #     #trying point (2,2,3)
    #     # print(face_constr.dot(np.array([2,2,3])) >= face_bound)
    #     face_constraint.append(face_constr)
    #     face_bounds.append(face_bound)

    # face_constraint = np.asarray(face_constraint)
    # face_bounds = np.asarray(face_bounds)

    # print((face_constraint == ak.to_numpy(new_face_constraints)).all())
    # print((face_bounds == ak.to_numpy(new_face_bounds)).all())
    # print(face_bounds)
    

    #----- Building the Vertice Sections (Constraint & Boundary Conditions) -----
    #TODO: make it more general


    # make_edges = shp_vert[shp_edge_vert[:,1]] - shp_vert[shp_edge_vert[:,0]]
    

bc_end = time.time()



print(np.round(r_list, 2))
print(np.round(r_list_new, 2))
print(np.round(other_r_list, 2))

print('Current', current_end - current_start)
print('New', new_end - new_start)
print('Other', other_end - other_start)
print('Bounds', bc_end - bc_start)
print('Current/New Ratio', (current_end - current_start)/(new_end - new_start))
print('Current/Other Ratio', (current_end - current_start)/(other_end - other_start))


a = np.tile(np.array([1,0.5,0]),27).reshape((27,3))
b = np.tile(np.array([0,1,0]),27).reshape((27,3))
c = np.tile(np.array([0.25, 0, 0.5]),27).reshape((27,3))

i_array = np.repeat(ijk_list[:,0],3).reshape((27,3))
j_array = np.repeat(ijk_list[:,1],3).reshape((27,3))
k_array = np.repeat(ijk_list[:,2],3).reshape((27,3))

# print(a)
# print(i_array)
vec_diff = a*i_array + b*j_array + c*k_array
# print(vec_diff+shape_pos[0])
# mega_mega = np.ones((18144,3888))
# print(mega_mega)

# print(np.max(min_list))
# print('New Min:',new_min.fun)
# print('Current Min:',np.min(min_list))
# print(min_list)
# # print(np.max(min_list)==new_min.fun)
# print(x_list)
# print(points1[2])
# print(new_halfspaces[:,:,:-1])

# print(new_halfspaces[0,:,:-1].dot(x_list[0]))
# print(new_halfspaces[0,:,-1])


# new_maybe_min = minimize(
#     fun=distance_point_point_multiple,
#     x0=np.zeros(3),
#     args=(points1,),
#     constraints=[(-np.inf<=new_halfspaces.dot(3))*(new_halfspaces.dot(3)<=6)]
# )
# print(new_maybe_min.fun)