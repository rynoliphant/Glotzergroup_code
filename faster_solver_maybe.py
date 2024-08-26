import numpy as np
import numpy.ma as ma
import numpy.linalg as LA
import time
from scipy.optimize import LinearConstraint, OptimizeResult, minimize
import coxeter
import rowan
import awkward as ak

#----- messing around -----
big_arr = np.array([
    [[[1,2,3],[4,5,6],[1,1,1],[0,0,0]],[[7,8,9],[2,4,6],[1,1,1],[0,0,0]],[[1,3,5],[4,6,8],[1,1,1],[0,0,0]],[[3,5,7],[2,6,9],[1,1,1],[0,0,0]]],
    [[[1,2,3],[4,5,6],[1,1,1],[0,0,0]],[[7,8,9],[2,4,6],[1,1,1],[0,0,0]],[[1,3,5],[4,6,8],[1,1,1],[0,0,0]],[[3,5,7],[2,6,9],[1,1,1],[0,0,0]]]
])

img_arr = np.array([
    [1,1,1],
    [2,2,2]
]).reshape(2,1,1,3)

neg_one_list = [[2,3],[0,2],[1,4,6]]
neg_one_list[0].insert(5, neg_one_list[0][0])
# neg_one_list[:][-1:] = -1
# print('Hello:',neg_one_list)

# print(big_arr * img_arr)

arr = np.array([
    [1,200,3.37521],
    [3,5,7]
])
tile = np.tile(arr, (3,1))
# print(tile)
repeat = np.repeat(arr, 3, axis=0)
# print(repeat)
# print(ak.sum(ak.Array(arr * ak.Array([1,2,2])), axis=1, highlevel=False))
variable = ak.to_list(ak.sum(arr * np.array([1,2,2]), axis=1))
# print(variable)
variable_ak = ak.Array([variable])
# print(variable_ak)
# print(ak.to_numpy(variable_ak))
# print(ak.Array([2, 3.521987, 94.19348]))

test = ak.Array([
    [[1,2,3], [4,5,6],[7,8,9]],
    [[9,4,5]],
    [[3,7,5], [8,5,2]]
])

test_broadcast = ak.broadcast_fields(np.array([2,2.5672,9.2538]), test)
# print(test_broadcast)
test0 = test*ak.Array([2,2.5672,9.2538])
# print(test0[0])
test_bool = test == ak.Array([1,2,3])
# test_np = ak.pad_none(test, 3, axis=1)
# test_np.show()
# test_np = ak.to_numpy(test_np)
# print(test_np)
# print(test_np + np.array([1,3,4.5]))
test_num = ak.num(test, axis=1)
# print(test_num)
# test_num_tot = ak.sum(test_num)
# test_vect_diff = np.tile(np.array([[1,1,0],[0,2,-1]]).reshape(2,1,3), (1, test_num_tot, 1))
# print(test_vect_diff)
# test_unflatten = ak.unflatten(test_vect_diff, ak.flatten(ak.Array([test_num, test_num])), axis=1)
# big_test = ak.Array([test, test])
# print(ak.sum(big_test * test_unflatten, axis=3))
# print(test[test_bool])

up_test = ak.Array([
    [15, 30, 42],
    [37],
    [30, 28]
])
# print(test0 <= up_test)

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
# print(con_mat[0])
# print(con_mat[1])
# print(con_mat[2])

none_ak = ak.Array([
    [[], [], []]
])

none_counts = ak.num(none_ak, axis=2)
none_counts = ak.flatten(none_counts[none_counts != 0])

none_ak = ak.flatten(none_ak, axis=2)
none_ak = ak.unflatten(none_ak, none_counts, axis=1)
# print(none_ak)

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

# print('First time:', middle-start)
# print('Second time:', end-middle)

# print('Difference in time:', (middle-start)/(end-middle))

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
    [2,0,0],
    [0,2,0],
    [0,0,2],
    [2,2,0],
    [2,0,2],
    [0,2,2],
    [2,1,1],
    [1,2,1],
    [1,1,2],
    [2,2,1],
    [2,1,2],
    [1,2,2],
    [2,2,2],
    [2,2,3],
    [3,1,1],
    [1,3,1],
    [1,1,3],
    [3,3,1],
    [3,1,3],
    [1,3,3],
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
    [6,6,6],

])

points1 = np.asarray(list(points1) * 5)

shape_pos = np.array([
    [2,2,2],[4,4,3],
    # [2,2,2],[4,4,3],
    # [2,2,2],[4,4,3],
    # [2,2,2],[4,4,3],
    # [2,2,2],[4,4,3],
    # [2,2,2],[4,4,3],
])

vertices = [
    #Jen Shape 1
    # [-0.67167939, -0.13433588,  0.26867175],
    # [-0.67167939,  0.13433588, -0.26867175],
    # [-0.50375954, -0.50375954,  0.50375954],
    # [-0.50375954,  0.50375954, -0.50375954],
    # [-0.33583969, -0.33583969, -0.33583969],
    # [-0.33583969,  0.33583969,  0.33583969],
    # [-0.26867175, -0.67167939,  0.13433588],
    # [-0.26867175,  0.67167939, -0.13433588],
    # [-0.13433588, -0.26867175,  0.67167939],
    # [-0.13433588,  0.26867175, -0.67167939],
    # [ 0.13433588, -0.26867175, -0.67167939],
    # [ 0.13433588,  0.26867175,  0.67167939],
    # [ 0.26867175, -0.67167939, -0.13433588],
    # [ 0.26867175,  0.67167939,  0.13433588],
    # [ 0.33583969, -0.33583969,  0.33583969],
    # [ 0.33583969,  0.33583969, -0.33583969],
    # [ 0.50375954, -0.50375954, -0.50375954],
    # [ 0.50375954,  0.50375954,  0.50375954],
    # [ 0.67167939, -0.13433588, -0.26867175],
    # [ 0.67167939,  0.13433588,  0.26867175]

    #Jen Shape 2
    # [-0.71962257, -0.28497054,  0.28784903],
    # [-0.71962257,  0.28497054, -0.28784903],
    # [-0.71258997, -0.71258997,  0.71258997],
    # [-0.71258997,  0.71258997, -0.71258997],
    # [-0.28784903, -0.71962257,  0.28497054],
    # [-0.28784903,  0.71962257, -0.28497054],
    # [-0.28497054, -0.28784903,  0.71962257],
    # [-0.28497054,  0.28784903, -0.71962257],
    # [-0.24279215, -0.24279215, -0.24279215],
    # [-0.24279215,  0.24279215,  0.24279215],
    # [ 0.24279215, -0.24279215,  0.24279215],
    # [ 0.24279215,  0.24279215, -0.24279215],
    # [ 0.28497054, -0.28784903, -0.71962257],
    # [ 0.28497054,  0.28784903,  0.71962257],
    # [ 0.28784903, -0.71962257, -0.28497054],
    # [ 0.28784903,  0.71962257,  0.28497054],
    # [ 0.71258997, -0.71258997, -0.71258997],
    # [ 0.71258997,  0.71258997,  0.71258997],
    # [ 0.71962257, -0.28497054, -0.28784903],
    # [ 0.71962257,  0.28497054,  0.28784903]

    #Jen Shape 3
    # [-0.68808966,  0.        , -0.34404483],
    # [-0.68808966,  0.        ,  0.34404483],
    # [-0.34404483, -0.59590312, -0.34404483],
    # [-0.34404483, -0.59590312,  0.34404483],
    # [-0.34404483,  0.59590312, -0.34404483],
    # [-0.34404483,  0.59590312,  0.34404483],
    # [ 0.34404483, -0.59590312, -0.34404483],
    # [ 0.34404483, -0.59590312,  0.34404483],
    # [ 0.34404483,  0.59590312, -0.34404483],
    # [ 0.34404483,  0.59590312,  0.34404483],
    # [ 0.68808966,  0.        , -0.34404483],
    # [ 0.68808966,  0.        ,  0.34404483],
    # [-0.93743438,  0.54122799,  0.        ],
    # [ 0.93743438,  0.54122799,  0.        ]

    #Jen Shape 4
    [-0.15009435, -0.15009435, -0.63580989],
    [-0.15009435, -0.15009435,  0.63580989],
    [-0.15009435,  0.15009435, -0.63580989],
    [-0.15009435,  0.15009435,  0.63580989],
    [-0.15009435, -0.63580989, -0.15009435],
    [-0.15009435,  0.63580989, -0.15009435],
    [ 0.        , -0.39295212, -0.54304647],
    [ 0.        ,  0.39295212, -0.54304647],
    [ 0.15009435, -0.15009435, -0.63580989],
    [ 0.15009435, -0.15009435,  0.63580989],
    [ 0.15009435,  0.15009435, -0.63580989],
    [ 0.15009435,  0.15009435,  0.63580989],
    [ 0.15009435, -0.63580989, -0.15009435],
    [ 0.15009435,  0.63580989, -0.15009435],
    [-0.54304647,  0.        , -0.39295212],
    [-0.54304647,  0.        ,  0.39295212],
    [-0.24285777, -0.48571553, -0.39295212],
    [-0.24285777,  0.48571553, -0.39295212],
    [-0.48571553, -0.39295212, -0.24285777],
    [-0.48571553, -0.39295212,  0.24285777],
    [-0.48571553,  0.39295212, -0.24285777],
    [-0.48571553,  0.39295212,  0.24285777],
    [-0.63580989, -0.15009435, -0.15009435],
    [-0.63580989, -0.15009435,  0.15009435],
    [-0.63580989,  0.15009435, -0.15009435],
    [-0.63580989,  0.15009435,  0.15009435],
    [-0.39295212, -0.54304647,  0.        ],
    [-0.39295212, -0.24285777, -0.48571553],
    [-0.39295212, -0.24285777,  0.48571553],
    [-0.39295212,  0.24285777, -0.48571553],
    [-0.39295212,  0.24285777,  0.48571553],
    [-0.39295212,  0.54304647,  0.        ],
    [ 0.24285777, -0.48571553, -0.39295212],
    [ 0.24285777,  0.48571553, -0.39295212],
    [ 0.48571553, -0.39295212, -0.24285777],
    [ 0.48571553, -0.39295212,  0.24285777],
    [ 0.48571553,  0.39295212, -0.24285777],
    [ 0.48571553,  0.39295212,  0.24285777],
    [ 0.63580989, -0.15009435, -0.15009435],
    [ 0.63580989, -0.15009435,  0.15009435],
    [ 0.63580989,  0.15009435, -0.15009435],
    [ 0.63580989,  0.15009435,  0.15009435],
    [ 0.39295212, -0.54304647,  0.        ],
    [ 0.39295212, -0.24285777, -0.48571553],
    [ 0.39295212, -0.24285777,  0.48571553],
    [ 0.39295212,  0.24285777, -0.48571553],
    [ 0.39295212,  0.24285777,  0.48571553],
    [ 0.39295212,  0.54304647,  0.        ],
    [ 0.54304647,  0.        , -0.39295212],
    [ 0.54304647,  0.        ,  0.39295212]

    #Dodecahedron
    # (0.5, 0.5, 0.5),
    # (-0.5, 0.5, 0.5),
    # (-0.5, -0.5, 0.5),
    # (-0.5, 0.5, -0.5),
    # (0.5, -0.5, 0.5),
    # (0.5, -0.5, -0.5),
    # (0.5, 0.5, -0.5),
    # (-0.5, -0.5, -0.5),
    # (0, 0.5/((1+np.sqrt(5))*0.5), 0.5*((1+np.sqrt(5))*0.5)),
    # (0, -0.5/((1+np.sqrt(5))*0.5), 0.5*((1+np.sqrt(5))*0.5)),
    # (0, 0.5/((1+np.sqrt(5))*0.5), -0.5*((1+np.sqrt(5))*0.5)),
    # (0, -0.5/((1+np.sqrt(5))*0.5), -0.5*((1+np.sqrt(5))*0.5)),
    # (0.5/((1+np.sqrt(5))*0.5), 0.5*((1+np.sqrt(5))*0.5), 0),
    # (-0.5/((1+np.sqrt(5))*0.5), 0.5*((1+np.sqrt(5))*0.5), 0),
    # (0.5/((1+np.sqrt(5))*0.5), -0.5*((1+np.sqrt(5))*0.5), 0),
    # (-0.5/((1+np.sqrt(5))*0.5), -0.5*((1+np.sqrt(5))*0.5), 0),
    # (0.5*((1+np.sqrt(5))*0.5), 0, 0.5/((1+np.sqrt(5))*0.5)),
    # (-0.5*((1+np.sqrt(5))*0.5), 0, 0.5/((1+np.sqrt(5))*0.5)),
    # (0.5*((1+np.sqrt(5))*0.5), 0, -0.5/((1+np.sqrt(5))*0.5)),
    # (-0.5*((1+np.sqrt(5))*0.5), 0, -0.5/((1+np.sqrt(5))*0.5)),

    #Tetrahedron
    # (0, 0, 0.5),
    # (0, 0.7071, -0.5),
    # (0.5*np.sqrt(3/2), -0.3535, -0.5),
    # (-0.5*np.sqrt(3/2), -0.3535, -0.5),

    #Octahedron
    # (-0.5, 0, 0),
    # (0.5, 0, 0),
    # (0, -0.5, 0),
    # (0, 0.5, 0),
    # (0, 0, -0.5),
    # (0, 0, 0.5),

    #Cube
    # (0.5, 0.5, 0.5),
    # (-0.5, 0.5, 0.5),
    # (-0.5, -0.5, 0.5),
    # (-0.5, 0.5, -0.5),
    # (0.5, -0.5, 0.5),
    # (0.5, -0.5, -0.5),
    # (0.5, 0.5, -0.5),
    # (-0.5, -0.5, -0.5),
]

orientations = [
    (1,0,0,0), (0.3647046, 0.1159188, -0.2798528, 0.8804747),
    # (1,0,0,0), (0.3647046, 0.1159188, -0.2798528, 0.8804747),
    # (1,0,0,0), (0.3647046, 0.1159188, -0.2798528, 0.8804747),
    # (1,0,0,0), (0.3647046, 0.1159188, -0.2798528, 0.8804747),
    # (1,0,0,0), (0.3647046, 0.1159188, -0.2798528, 0.8804747),
    # (1,0,0,0), (0.3647046, 0.1159188, -0.2798528, 0.8804747),
]

shapes = []
new_halfspaces = []
for i, pos in enumerate(shape_pos):
    rot_vert = rowan.rotate(orientations[i], vertices)
    tmpshape = coxeter.shapes.ConvexPolyhedron(vertices=rot_vert)

    normals = tmpshape.normals
    num_faces = tmpshape.num_faces
    face_dist = np.sum(tmpshape.face_centroids*normals, axis=1)

    dx = 0.001
    norms = np.asarray(tmpshape.normals)
    tmpcentroid = np.asarray(tmpshape.centroid)
    norm_dists = []

    for norm in norms:
        x = tmpcentroid

        dx_n = 0
        while tmpshape.is_inside(x):
            dx_n = dx_n + 1
            x = tmpcentroid + dx_n * dx * norm
        norm_dists.append((dx_n - 1) * dx)

    halfspace = []
    image_diff = [[6,0,0], [0,0,0], [-6,0,0]]
    for image in image_diff:
        position = pos + image
        upper_bounds = normals.dot(position) + face_dist
        smile = np.block([normals, upper_bounds.reshape((num_faces,1))])

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

# print(new_halfspaces.shape)

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
                constraints=[LinearConstraint(n[:,:-1], -np.inf, n[:,-1])],
                tol = 0.0000000001
            )
            if current_min.success == False:
                print(current_min.message)
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
# print(new_mega_matrix.shape)
# print(new_mega_ub.shape)

#----- New Min Dist Calculations -----
new_start = time.time()
# r_list_new = []

# for i in range(len(points1)):
#     xp_min = minimize(
#         fun=hell_if_i_know,
#         x0=np.zeros(n_shapes*n_constraints*3),
#         args=(np.tile(points1[i],(n_shapes*n_constraints))),
#         constraints=[LinearConstraint(new_mega_matrix, -np.inf, new_mega_ub)]
#     )
#     xp_min_list = uhhh(xp_min.x, points1[i])

#     find_min = np.min(xp_min_list.reshape((n_shapes, n_constraints)), axis=1)

#     r_list_new.append(find_min)

# r_list_new = np.asarray(r_list_new)
# r_list_new = np.transpose(r_list_new)
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
    # face_ragged_list = shape.faces
    # new_ragged_list = np.array([])
    # for rag_i in range(len(face_ragged_list)):
    #     new_arr = np.append(face_ragged_list[rag_i],np.array([face_ragged_list[rag_i][0], -1]))
    #     new_ragged_list = np.append(new_ragged_list, new_arr)

    # list_len = len(new_ragged_list)
    # print(new_ragged_list)
    # face_edge_mat = np.block([new_ragged_list[:-1].reshape(list_len-1,1), new_ragged_list[1:].reshape(list_len-1,1)])
    # print(face_edge_mat.shape)

    faces = ak.Array(shape.faces)
    faces_len = shape.num_faces
    num_edges = shape.num_edges

    faces_plus = ak.concatenate((faces, ak.to_numpy(faces[:,0]).reshape(faces_len,1), np.array([-1]*faces_len).reshape(faces_len,1)), axis=1)
    faces_flat = ak.to_numpy(ak.flatten(faces_plus))

    list_len = len(faces_flat)
    face_edge_mat = np.block([faces_flat[:-1].reshape(list_len-1,1), faces_flat[1:].reshape(list_len-1,1)])
    fe_mat_inds = np.arange(0,list_len-1,1)
    find_num_edges = fe_mat_inds[(fe_mat_inds==0) + (np.any(face_edge_mat==-1, axis=1))]
    find_num_edges[:][0] = -1
    find_num_edges = find_num_edges.reshape(faces_len,2)
    face_num_edges = find_num_edges[:,1] - find_num_edges[:,0] -1
    face_correspond_inds = np.repeat(np.arange(0,faces_len,1), face_num_edges)
    # print(face_correspond_inds)

    true_face_edge_mat = np.tile(face_edge_mat[np.all(face_edge_mat!=-1, axis=1)].reshape(num_edges*2,1,2), (1, num_edges,1))
    # edge_ind_bool0 = np.any(np.all(true_face_edge_mat == shape.edges.reshape(1, num_edges, 2), axis=2), axis=1)
    # ef_sort_bool0 = np.all(true_face_edge_mat == shape.edges.reshape(1, num_edges, 2), axis=2)
    # edges1_reshape = np.hstack(((shape.edges[:,1]).reshape(num_edges,1), (shape.edges[:,0]).reshape(num_edges,1)))
    # edge_ind_bool1 = np.any(np.all(true_face_edge_mat == edges1_reshape.reshape(1,num_edges,2), axis=2), axis=1)
    # ef_sort_bool1 = np.all(true_face_edge_mat == edges1_reshape.reshape(1,num_edges,2), axis=2)
    # ef_sort = np.tile(np.arange(0, num_edges, 1), (num_edges*2, 1))
    # print(ef_sort[ef_sort_bool1])
    # print(ef_sort[ef_sort_bool0])
    # print(face_correspond_inds[edge_ind_bool0])
    # print(edge_ind_bool)
    # print(true_face_edge_mat.shape)

    edges1_reshape = np.hstack(((shape.edges[:,1]).reshape(num_edges,1), (shape.edges[:,0]).reshape(num_edges,1)))
    new_edge_ind_bool0 = np.all(true_face_edge_mat == shape.edges.reshape(1, num_edges, 2), axis=2)
    new_edge_ind_bool1 = np.all(true_face_edge_mat == edges1_reshape.reshape(1,num_edges,2), axis=2)
    new_face_corr_inds = np.tile(face_correspond_inds.reshape(2*num_edges,1), (1,num_edges))


    ef_neighbor0 = np.sum(new_face_corr_inds*new_edge_ind_bool0, axis=0).reshape(num_edges, 1)
    ef_neighbor1 = np.sum(new_face_corr_inds*new_edge_ind_bool1, axis=0).reshape(num_edges, 1)
    ef_neighbor = np.hstack((ef_neighbor0, ef_neighbor1))
    



    # face_inds = np.arange(0, shape.num_faces, 1)

    # edge_face_neighbors = []
    # for edge in shape.edges:
    #       edge1_bool = faces == edge[0]
    #       edge2_bool = faces == edge[1]

    #       check_edge = ak.sum(edge1_bool + edge2_bool, axis=1) == 2

    #       neighbors = face_inds[check_edge]

    #       sum_edges = edge1_bool + edge2_bool
    #       face_verts = faces[sum_edges * check_edge]
    #       flat_face_verts = ak.ravel(face_verts)
    #       reshape_face_verts = np.array(flat_face_verts).reshape((2,2))
    #       # print(reshape_face_verts)

    #       face_verts_bool = (reshape_face_verts != edge)[:,0] +1 -1

    #       face_all_verts0 = np.array(faces[check_edge][:,0]).reshape((2,1))
    #       face_all_verts_last = np.array(faces[check_edge][:,-1]).reshape((2,1))
    #       first_last_verts = np.block([face_all_verts0, face_all_verts_last])
    #       # print(first_last_verts)

    #       ind_edge_bool = (np.sum(first_last_verts == edge, axis=1) == 2) +1 -1

    #       switch_neighbor_bool = face_verts_bool + ind_edge_bool
    #       # print('edge', edge)
    #       # print('switch bool', switch_neighbor_bool)

    #       # print('previous neighbor', neighbors)
    #       neighbors = neighbors[switch_neighbor_bool]
    #       # print('new neighbor', neighbors)

    #       edge_face_neighbors.append(neighbors)

    # edge_face_neighbors = np.asarray(edge_face_neighbors)
    # print(np.all(edge_face_neighbors == ef_neighbor))
    # print('old', edge_face_neighbors)

    return ef_neighbor

def bounds_for_images (bounds, constraint):
    #for the EBT code, will need lx, ly, lz, a, b, & c variables
    # con_num = ak.num(constraint, axis=1)
    # con_num_sum = ak.sum(con_num)
    # image_diff = np.tile(np.array([[6,0,0], [0,0,0], [-6,0,0]]).reshape(3,1,3), (1,con_num_sum, 1))
    # big_con_num = ak.flatten(ak.Array([con_num, con_num, con_num]))
    # image_diff_unflatten = ak.unflatten(image_diff, big_con_num, axis=1)
    big_bounds = np.array([bounds, bounds, bounds])
    big_constraint = np.array([constraint, constraint, constraint])

    image_diff = np.array([[6,0,0],[0,0,0],[-6,0,0]]).reshape(3,1,1,3)

    new_bounds = big_bounds + (np.sum(big_constraint * image_diff, axis=3))
    return new_bounds

def point_to_edge_distance (point, vert, edge_vector, multiple=False):
    if multiple:
        edge_units = edge_vector / LA.norm(edge_vector, axis=1).reshape(len(edge_vector), 1)
        dist = LA.norm(((vert - point) - (np.sum((vert-point)*edge_units, axis=1).reshape(len(edge_vector),1) *edge_units)), axis=1)
        return dist
    
    edge_vect_mag = LA.norm(edge_vector)
    if edge_vect_mag == 0:
        return
    edge_unit = edge_vector / edge_vect_mag
    dist = LA.norm((vert - point) - ((vert - point).dot(edge_unit)*edge_unit))
    return dist

def point_to_face_distance(point, vert, face_normal, multiple=False):
    if multiple:
        vert_point_vect = -1*vert + point
        face_unit = face_normal / LA.norm(face_normal, axis=1).reshape(len(face_normal), 1)
        dist = np.sum(vert_point_vect*face_unit, axis=1)
        return dist
    
    vert_point_vect = point - vert
    face_unit = face_normal / LA.norm(face_normal)
    dist = vert_point_vect.dot(face_unit)
    return dist

def get_vert_boundaries (shp_vert, shp_edges, shp_edge_vert, n_verts, n_edges, x):
    #Tiling for set up
    nverts_edge_vert0 = np.tile(shp_edge_vert[:,0], (n_verts, 1))
    nverts_edge_vert1 = np.tile(shp_edge_vert[:,1], (n_verts, 1))
    vert_inds = np.arange(0, n_verts, 1).reshape((n_verts, 1))
    nverts_tile_edges = np.tile(shp_edges, (n_verts, 1)).reshape((n_verts, n_edges, 3))

    #Creating the bools need to get the edges that correspond to each vertice
    evbool0 = np.expand_dims(nverts_edge_vert0 == vert_inds, axis=2)
    evbool1 = np.expand_dims(nverts_edge_vert1 == vert_inds, axis=2)

    #Removing the None values from the ak arrays
    ak_vert_neg = nverts_tile_edges * evbool0
    ak_vert_pos = nverts_tile_edges * evbool1 * (-1) 

    #Concatenating the ak arrays together to make the vertice constraint array | shape:(n_vert, #, 3)
    vert_constraint = ak.to_numpy(ak_vert_neg + ak_vert_pos)
    
    #Building the boundary conditions
    vert_bounds = np.sum(vert_constraint * (shp_vert+x).reshape(n_verts,1,3), axis=2)

    return vert_constraint, vert_bounds

def get_edge_boundaries (shp_vert, shp_edges, shp_faces, shp_edge_vert, shp_edge_face, n_edges, x):
    new_edge_constraint = np.zeros((n_edges, 4, 3))
    new_edge_bounds = np.zeros((n_edges, 4))

    edge_constraint_col_1 = shp_edges
    edge_constraint_col_2 = -1*shp_edges
    edge_constraint_col_3 = np.cross(shp_edges, shp_faces[shp_edge_face[:,1]])
    edge_constraint_col_4 = -1*np.cross(shp_edges, shp_faces[shp_edge_face[:,0]])

    new_edge_constraint[:,0] = edge_constraint_col_1
    new_edge_constraint[:,1] = edge_constraint_col_2
    new_edge_constraint[:,2] = edge_constraint_col_3
    new_edge_constraint[:,3] = edge_constraint_col_4

    new_edge_verts = np.zeros((n_edges, 2, 3))
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

    return new_edge_constraint, new_edge_bounds

def get_face_boundaries (shp_vert, shp_edges, shp_faces, shp_edge_vert, shp_edge_face, n_edges, n_faces, x, face_centroids):
    #Setting up
    nfaces_tile_edges = np.tile(shp_edges, (n_faces, 1)).reshape((n_faces, n_edges, 3)) #Edge vectors tiled
    nfaces_tile_efneighbors = np.tile(shp_edge_face, (n_faces, 1)) #Tiling shp_edge_face so that it is now shape (n_faces, n_edges, 2)
    efneighbors0 = nfaces_tile_efneighbors[:,0].reshape((n_faces, n_edges)) #breaking nfaces_tile_efneighbors into the first column [0]
    efneighbors1 = nfaces_tile_efneighbors[:,1].reshape((n_faces, n_edges)) #second column [1] of nfaces_tile_efneighbors
    faces_inds = np.arange(0, n_faces, 1).reshape((n_faces, 1)) #Making an array containing the indices of the faces

    #Creating bools to find the edges that correspond to each face
    efbool0 = (efneighbors0 == faces_inds)
    efbool1 = (efneighbors1 == faces_inds)

    ak_edges_mask0 = nfaces_tile_edges * efbool0.reshape((n_faces, n_edges, 1))
    ak_edges_mask1 = nfaces_tile_edges * efbool1.reshape((n_faces, n_edges, 1))
    faces_repeat = np.repeat(shp_faces, n_edges, axis=0).reshape((n_faces, n_edges, 3)) 

    neg_constraints = np.cross(ak_edges_mask0, faces_repeat)
    pos_constraints = -1* np.cross(ak_edges_mask1, faces_repeat)
    face_normals = -1* shp_faces

    new_face_constraints = np.zeros((n_faces, n_edges+1, 3))
    new_face_constraints[:,0] = face_normals
    new_face_constraints[:,1:] = neg_constraints + pos_constraints

    #bounds
    nfaces_tile_evneighbors = np.tile(shp_edge_vert, (n_faces, 1)).reshape((n_faces, n_edges, 2))
    neg_face_verts = (shp_vert[(nfaces_tile_evneighbors[:,:,0]*efbool0)] + x)
    pos_face_verts = (shp_vert[(nfaces_tile_evneighbors[:,:,1]*efbool1)] + x)

    neg_bounds = np.sum(neg_constraints * neg_face_verts, axis=2)
    pos_bounds = np.sum(pos_constraints * pos_face_verts, axis=2)
    norm_distances = (face_centroids + x)
    normal_bounds = np.sum(face_normals*norm_distances, axis=1)

    new_face_bounds = np.zeros((n_faces, n_edges+1))
    new_face_bounds[:,0] = normal_bounds
    new_face_bounds[:,1:] = neg_bounds + pos_bounds

    return new_face_constraints, new_face_bounds


bc_start = time.time()

bc_mid = 0
bc_edge = 0
bc_face = 0
bc_vert = 0
bc_for = 0
bc_r_list = []
for s,x in enumerate(shape_pos):
    #TODO: normalize all constraints so that they are unit vectors!!!?

    shp = shapes[s]

    shp_faces = shp.normals
    shp_vert = shp.vertices
    shp_edges = shp.edge_vectors

    shp_edge_vert = shp.edges #column [0]: o-->  | column [1]: o<--

    bc_mid_s = time.time()
    shp_edge_face = get_edge_face_neighbors(shp)
    bc_mid_e = time.time()
    bc_mid += (bc_mid_e - bc_mid_s)

    n_faces = shp.num_faces #number of faces
    n_edges = shp.num_edges #number of edges
    n_verts = shp.num_vertices #number of vertices
    face_centroids = shp.face_centroids

    bc_vert_s = time.time()
    vert_constraint, vert_bounds = get_vert_boundaries (shp_vert, shp_edges, shp_edge_vert, n_verts, n_edges, x)
    img_vert_bounds = bounds_for_images(vert_bounds, vert_constraint)
    bc_vert_e = time.time()
    bc_vert += (bc_vert_e - bc_vert_s)

    bc_edge_s = time.time()
    new_edge_constraint, new_edge_bounds = get_edge_boundaries (shp_vert, shp_edges, shp_faces, shp_edge_vert, shp_edge_face, n_edges, x)
    img_edge_bounds = bounds_for_images(new_edge_bounds, new_edge_constraint)
    bc_edge_e = time.time()
    bc_edge += (bc_edge_e - bc_edge_s)

    bc_face_s = time.time()
    new_face_constraints, new_face_bounds = get_face_boundaries (shp_vert, shp_edges, shp_faces, shp_edge_vert, shp_edge_face, n_edges, n_faces, x, face_centroids)
    img_face_bounds = bounds_for_images(new_face_bounds, new_face_constraints)
    bc_face_e = time.time()
    bc_face += (bc_face_e - bc_face_s)

    # faces = np.asarray(shp.faces)
    # print(shp_edge_vert)
    # print(shp_edge_face)
    # print(faces)

    # bc_edge_s = time.time()
    #----- Building the Edge Sections (Constraints & Boundary Conditions) -----
    # new_edge_constraint = np.zeros((len(shp_edges), 4, 3))
    # new_edge_bounds = np.zeros((len(shp_edges), 4))

    # edge_constraint_col_1 = -1*shp_edges
    # edge_constraint_col_2 = shp_edges
    # edge_constraint_col_3 = -1*np.cross(shp_edges, shp_faces[shp_edge_face[:,1]])
    # edge_constraint_col_4 = np.cross(shp_edges, shp_faces[shp_edge_face[:,0]])

    # new_edge_constraint[:,0] = edge_constraint_col_1
    # new_edge_constraint[:,1] = edge_constraint_col_2
    # new_edge_constraint[:,2] = edge_constraint_col_3
    # new_edge_constraint[:,3] = edge_constraint_col_4
    # # new_edge_constraint = np.block([edge_constraint_col_1, edge_constraint_col_2, edge_constraint_col_3, edge_constraint_col_4]).reshape(n_edges, 4, 3)

    # new_edge_verts = np.zeros((len(shp_edges), 2, 3))
    # new_edge_verts[:,0] = shp_vert[shp_edge_vert[:,0]] + x
    # new_edge_verts[:,1] = shp_vert[shp_edge_vert[:,1]] + x

    # edge_bounds_1 = np.sum(new_edge_constraint[:,0] *(new_edge_verts[:,1]), axis=1)
    # edge_bounds_2 = np.sum(new_edge_constraint[:,1] *(new_edge_verts[:,0]), axis=1)
    # edge_bounds_3 = np.sum(new_edge_constraint[:,2] *(new_edge_verts[:,0]), axis=1)
    # edge_bounds_4 = np.sum(new_edge_constraint[:,3] *(new_edge_verts[:,0]), axis=1)

    # new_edge_bounds[:,0] = edge_bounds_1
    # new_edge_bounds[:,1] = edge_bounds_2
    # new_edge_bounds[:,2] = edge_bounds_3
    # new_edge_bounds[:,3] = edge_bounds_4

    # new_edge_bounds = np.block([edge_bounds_1, edge_bounds_2, edge_bounds_3, edge_bounds_4]).reshape(n_edges, 4)

    # print(new_edge_constraint.dot(np.array([2,3,3])) >= new_edge_bounds)
    #constraint matrix normals are pointing inwards, so use lower bounds
    # img_edge_bounds = bounds_for_images(new_edge_bounds, new_edge_constraint)
    # bc_edge_e = time.time()
    # bc_edge += (bc_edge_e - bc_edge_s)



    # bc_face_s = time.time()
    #----- Building the Face Sections (Constraints & Boundary Conditions) -----
    #Setting up
    # nfaces_tile_edges = np.tile(shp_edges, (n_faces, 1)).reshape((n_faces, n_edges, 3)) #Edge vectors tiled
    # nfaces_tile_efneighbors = np.tile(shp_edge_face, (n_faces, 1)) #Tiling shp_edge_face so that it is now shape (n_faces, n_edges, 2)
    # efneighbors0 = nfaces_tile_efneighbors[:,0].reshape((n_faces, n_edges)) #breaking nfaces_tile_efneighbors into the first column [0]
    # efneighbors1 = nfaces_tile_efneighbors[:,1].reshape((n_faces, n_edges)) #second column [1] of nfaces_tile_efneighbors
    # faces_inds = np.arange(0, n_faces, 1).reshape((n_faces, 1)) #Making an array containing the indices of the faces

    # #Creating bools to find the edges that correspond to each face
    # efbool0 = (efneighbors0 == faces_inds)
    # efbool1 = (efneighbors1 == faces_inds)
    
    # #Applying the bools to get the edge vectors corresponding to each face
    # # ak_edges_mask0 = ak.to_numpy(ak.mask(nfaces_tile_edges, efbool0)) #Applying the bool: efbool0 to ak_tile_edges, but returning Nones for False instead of removing them
    # # ak_edges_mask1 = ak.to_numpy(ak.mask(nfaces_tile_edges, efbool1))#^---

    # ak_edges_mask0 = nfaces_tile_edges * efbool0.reshape((n_faces, n_edges, 1))
    # ak_edges_mask1 = nfaces_tile_edges * efbool1.reshape((n_faces, n_edges, 1))
    # faces_repeat = np.repeat(shp_faces, n_edges, axis=0).reshape((n_faces, n_edges, 3)) 

    # # print('numpy',ak_edges_mask0[0])
    # # print('numpy',ak_edges_mask1[0])

    # #Creating the constraints that are a -1 cross product (edges X faces) (points inwards)
    # # neg_constraints = ak.from_numpy(-1*np.cross(ak_edges_mask0, faces_repeat)) #Doing the cross product of the edges to the faces
    # # neg_constraints = ak.drop_none(ak.nan_to_none(neg_constraints)) #changing the NaN back to None, and then dropping the None
    # # neg_counts = ak.num(neg_constraints, axis=2)
    # # neg_counts = ak.flatten(neg_counts[neg_counts != 0])
    # # neg_constraints = ak.flatten(neg_constraints, axis=2)
    # # neg_constraints = ak.unflatten(neg_constraints, neg_counts, axis=1)
    # # neg_con_count = ak.num(neg_constraints, axis=1)
    # neg_constraints = -1* np.cross(ak_edges_mask0, faces_repeat)
    # # print(neg_con_count)

    # #Creating the constraints that are a +1 cross product (edges X faces) (points inwards)
    # # pos_constraints = ak.from_numpy(np.cross(ak_edges_mask1, faces_repeat))
    # # pos_constraints = ak.drop_none(ak.nan_to_none(pos_constraints))
    # # pos_counts = ak.num(pos_constraints, axis=2)
    # # pos_counts = ak.flatten(pos_counts[pos_counts != 0])
    # # pos_constraints = ak.flatten(pos_constraints, axis=2)
    # # pos_constraints = ak.unflatten(pos_constraints, pos_counts, axis=1)
    # # pos_con_count = ak.num(pos_constraints, axis=1)
    # pos_constraints = np.cross(ak_edges_mask1, faces_repeat)
    # # print(pos_con_count)

    # #Creating the constraints that come from the normals of each face
    # face_normals = shp_faces
    
    # #Concatenating the separate constraints together into one
    # # new_face_constraints = ak.concatenate((face_normals, neg_constraints, pos_constraints), axis=1)
    # # face_con_counts = ak.num(new_face_constraints, axis=2)
    # # face_con_counts = ak.flatten(face_con_counts[face_con_counts != 0])
    # # new_face_constraints = ak.flatten(new_face_constraints, axis=2)
    # # new_face_constraints = ak.to_numpy(ak.unflatten(new_face_constraints, face_con_counts, axis=1))
    # new_face_constraints = np.zeros((n_faces, n_edges+1, 3))
    # new_face_constraints[:,0] = face_normals
    # new_face_constraints[:,1:] = neg_constraints + pos_constraints

    # #bounds
    # nfaces_tile_evneighbors = np.tile(shp_edge_vert, (n_faces, 1)).reshape((n_faces, n_edges, 2))
    # # neg_face_verts = ak.from_numpy(shp_vert[nfaces_tile_evneighbors[efbool0][:,0]] + x)
    # # neg_face_verts = ak.unflatten(neg_face_verts, neg_con_count, axis=0)
    # neg_face_verts = (shp_vert[(nfaces_tile_evneighbors[:,:,0]*efbool0)] + x)

    # # pos_face_verts = ak.from_numpy(shp_vert[nfaces_tile_evneighbors[efbool1][:,1]] + x)
    # # pos_face_verts = ak.unflatten(pos_face_verts, pos_con_count, axis=0)
    # pos_face_verts = (shp_vert[(nfaces_tile_evneighbors[:,:,1]*efbool1)] + x)

    # neg_bounds = np.sum(neg_constraints * neg_face_verts, axis=2)
    # pos_bounds = np.sum(pos_constraints * pos_face_verts, axis=2)
    # norm_distances = (shp.face_centroids + x)
    # normal_bounds = np.sum(face_normals*norm_distances, axis=1)

    # # print('neg', neg_bounds)
    # # print('pos', pos_bounds)
    
    # # print(neg_bounds[1])
    # # print(pos_bounds[1])
    # # new_face_bounds = ak.to_numpy(ak.concatenate((normal_bounds, neg_bounds, pos_bounds), axis=1))pos_neg_bounds = np.sum()
    # # face_dot = neg_face_verts + pos_face_verts
    # # face_mat_dot = neg_constraints + pos_constraints
    # # pos_neg_bounds = np.sum(face_mat_dot * face_dot, axis=2)

    # new_face_bounds = np.zeros((n_faces, n_edges+1))
    # new_face_bounds[:,0] = normal_bounds
    # new_face_bounds[:,1:] = neg_bounds + pos_bounds

    # print(new_face_bounds)
    
    # img_face_bounds = bounds_for_images(new_face_bounds, new_face_constraints)
    # bc_face_e = time.time()
    # bc_face += (bc_face_e - bc_face_s)

    #----- Old Face Boundaries -----
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
    

    # bc_vert_s = time.time()
    #----- Building the Vertice Sections (Constraint & Boundary Conditions) -----
    # #Tiling for set up
    # nverts_edge_vert0 = np.tile(shp_edge_vert[:,0], (n_verts, 1))
    # nverts_edge_vert1 = np.tile(shp_edge_vert[:,1], (n_verts, 1))
    # vert_inds = np.arange(0, n_verts, 1).reshape((n_verts, 1))
    # nverts_tile_edges = np.tile(shp_edges, (n_verts, 1)).reshape((n_verts, n_edges, 3))

    # #Creating the bools need to get the edges that correspond to each vertice
    # evbool0 = np.expand_dims(nverts_edge_vert0 == vert_inds, axis=2)
    # evbool1 = np.expand_dims(nverts_edge_vert1 == vert_inds, axis=2)

    # #Finding the edge vectors that correspond to each vertice using the bools previously created
    # # ak_vert_mask0 = ak.mask(nverts_tile_edges, evbool0)
    # # ak_vert_mask1 = ak.mask(nverts_tile_edges, evbool1)

    # #Removing the None values from the ak arrays
    # # ak_vert_neg = ak.drop_none(ak_vert_mask0) * (-1)
    # # ak_vert_pos = ak.drop_none(ak_vert_mask1)
    # ak_vert_neg = nverts_tile_edges * evbool0 * (-1) #ak.fill_none(ak_vert_mask0, [0,0,0], axis=None) * (-1)
    # ak_vert_pos = nverts_tile_edges * evbool1 #ak.fill_none(ak_vert_mask1, [0,0,0], axis=None)

    # #Concatenating the ak arrays together to make the vertice constraint array | shape:(n_vert, #, 3)
    # vert_constraint = ak.to_numpy(ak_vert_neg + ak_vert_pos)
    # # vert_constraint = ak.fill_none(vert_constraint, [0,0,0], axis=None)
    
    # #Building the boundary conditions
    # # evbool_tot = evbool0 + evbool1
    # # nedges_repeat_verts = np.repeat((shp_vert+x).reshape(n_verts,1,3), n_edges, axis=1)
    # # dot_verts = ak.fill_none(ak.mask(nedges_repeat_verts, evbool_tot), [0,0,0], axis=None)
    # vert_bounds = np.sum(vert_constraint * (shp_vert+x).reshape(n_verts,1,3), axis=2)

    # vert_constraint_np = ak.to_numpy(vert_constraint)
    # vert_bounds_np = ak.to_numpy(vert_bounds)

    # print(shp_vert[0] + x)
    # print(vert_constraint_np[0])
    # print(np.sum(vert_constraint_np * np.array([3,3,3]), axis=1) >= vert_bounds_np)

    
    # img_vert_bounds = bounds_for_images(vert_bounds, vert_constraint)
    # bc_vert_e = time.time()
    # bc_vert += (bc_vert_e - bc_vert_s)

    # make_edges = shp_vert[shp_edge_vert[:,1]] - shp_vert[shp_edge_vert[:,0]]

    bc_for_s = time.time()
    image_diff = np.array([[6,0,0], [0,0,0], [-6,0,0]])
    images_pos = (x+np.array([6,0,0]), x, x+np.array([-6,0,0]))
    img_verts = np.tile(shp_vert.reshape(1, n_verts,3), (3,1,1))
    img_edges = np.tile(shp_edges.reshape(1, n_edges,3), (3,1,1))
    img_faces = np.tile(shp_faces.reshape(1, n_faces,3), (3,1,1))
    img_ev_neighbors = np.tile(shp_edge_vert.reshape(1, n_edges, 2), (3,1,1))
    # img_ef_neighbors = np.tile(shp_edge_face.reshape(1, n_edges, 2), (3,1,1))
    img_face_centroids = np.tile(face_centroids.reshape(1,n_faces,3), (3,1,1))
    bc_r = np.array([])
    for coord_i in range(len(points1)):

        #IMPORTANT: When finding the 8 nearest shapes, find the distance between coord and shape_pos + shape_centroid

        #Building code to do more than one image at once
        coord = points1[coord_i]
        min_dist_list = np.array([])

        vert_bool = np.all(img_vert_bounds >= vert_constraint.dot(coord), axis=2)
        if np.any(vert_bool):
            vert_bool_img = np.any(vert_bool, axis=1)
            min_dist = LA.norm(-1*(img_verts[vert_bool] + image_diff[vert_bool_img] + x) + coord, axis=1)
            min_dist_list = np.append(min_dist_list, min_dist)

        edge_bool = np.all(img_edge_bounds >= new_edge_constraint.dot(coord), axis=2)
        if np.any(edge_bool):
            edge_bool_img = np.any(edge_bool, axis=1)
            vert_on_edge = shp_vert[img_ev_neighbors[edge_bool][:,0]] + x + image_diff[edge_bool_img]
            min_dist = point_to_edge_distance(coord, vert_on_edge, img_edges[edge_bool], multiple=True)
            min_dist_list = np.append(min_dist_list, min_dist)

        face_bool = np.all(img_face_bounds >= new_face_constraints.dot(coord), axis=2)
        if np.any(face_bool):
            face_bool_img = np.any(face_bool, axis=1)
            vert_on_face = img_face_centroids[face_bool] + x + image_diff[face_bool_img] 
            min_dist = point_to_face_distance(coord, vert_on_face, img_faces[face_bool], multiple=True)
            min_dist_list = np.append(min_dist_list, min_dist)

        if np.any(shp.is_inside(-1*image_diff - x + coord)):
            min_dist=np.array([0])
            min_dist_list = np.append(min_dist_list, min_dist)

        true_min_dist = np.min(min_dist_list)
        bc_r = np.append(bc_r,true_min_dist)



        # dist_images_list = np.sqrt(np.sum((images_pos - points1[coord_i])**2, axis=1))
        # con_ind = np.argmin(dist_images_list)
        # coord = points1[coord_i]

        # min_dist = -1

        # vert_bool = np.all(vert_constraint.dot(coord) >= img_vert_bounds[con_ind], axis=1)
        # if np.any(vert_bool):
        #     min_dist = LA.norm(coord - (shp_vert[vert_bool]+x+image_diff[con_ind]))
        #     bc_r.append(min_dist)
        #     # continue

        # edge_bool = np.all(new_edge_constraint.dot(coord) >= img_edge_bounds[con_ind], axis=1)
        # if np.any(edge_bool):
        #     vert_on_edge = shp_vert[shp_edge_vert[edge_bool][0,0]] + x + image_diff[con_ind]
        #     min_dist = point_to_edge_distance(coord, vert_on_edge, shp_edges[edge_bool][0])
        #     bc_r.append(min_dist)
        #     # continue

        # face_bool = np.all(new_face_constraints.dot(coord) >= img_face_bounds[con_ind], axis=1)
        # if np.any(face_bool):
        #     vert_on_face = (face_centroids[face_bool] + x + image_diff[con_ind] + centroid)[0]
        #     min_dist = point_to_face_distance(coord, vert_on_face, shp_faces[face_bool][0])
        #     bc_r.append(min_dist)
        #     # continue

        # if shp.is_inside(coord - x - image_diff[con_ind]):
        #     min_dist = 0
        #     bc_r.append(min_dist)
        #     # continue

        # if min_dist == -1:
        #     print('WARNING: Point not found!')
        #     bc_r.append(min_dist)

        # print(min_dist == true_min_dist)
        # if min_dist != true_min_dist:
        #     print('old', min_dist)
        #     print('new', true_min_dist)
            
         
        
    bc_r_list.append(bc_r)

    bc_for_e = time.time()
    bc_for += (bc_for_e - bc_for_s)
        
    

bc_r_list = np.asarray(bc_r_list)
bc_end = time.time()



print(np.round(r_list, 2))
# print(np.round(r_list_new, 2))
# print(np.round(other_r_list, 2))
print(np.round(bc_r_list, 2))
# print('difference',abs(r_list - bc_r_list))
print('max difference:',np.max(abs(r_list - bc_r_list)))
print('min difference:',np.min(abs(r_list - bc_r_list)))
print('mean difference:',np.mean(abs(r_list - bc_r_list)))
print('std:', np.std(abs(r_list - bc_r_list)))

print('Exactly the Same?', np.all(bc_r_list ==  r_list))
print('Rounded (0.00000001) the Same?', np.all(np.round(bc_r_list,8) == np.round(r_list,8)))
print('Rounded (0.0000001) the Same?', np.all(np.round(bc_r_list,7) == np.round(r_list,7)))
print('Rounded (0.000001) the Same?', np.all(np.round(bc_r_list,6) == np.round(r_list,6)))
print('Rounded (0.00001) the Same?', np.all(np.round(bc_r_list,5) == np.round(r_list,5)))
print('Rounded (0.0001) the Same?', np.all(np.round(bc_r_list,4) == np.round(r_list,4)))
print('Rounded (0.001) the Same?', np.all(np.round(bc_r_list,3) == np.round(r_list,3)))
print('Rounded (0.01) the Same?', np.all(np.round(bc_r_list,2) == np.round(r_list,2)))
print('Rounded (0.1) the Same?', np.all(np.round(bc_r_list,1) == np.round(r_list,1)))

print('Current', '\033[1m'+str(current_end - current_start)+ '\033[0m')
# print('New', new_end - new_start)
print('Other', other_end - other_start)
print('Bounds', '\033[1m'+str(bc_end - bc_start)+ '\033[0m')
print('Bounds E-F Neighbors', bc_mid)
print('Bounds Edges', bc_edge)
print('Bounds Faces', bc_face)
print('Bounds Verts', bc_vert)
print('Bounds For Loop', bc_for)
# print('Current/New Ratio', (current_end - current_start)/(new_end - new_start))
print('Current/Other Ratio', (current_end - current_start)/(other_end - other_start))
print('Current/Bounds Ratio', '\033[1m'+str((current_end - current_start)/(bc_end - bc_start)) + '\033[0m')
print('Current/Bounds For Ratio', (current_end - current_start)/bc_for)
print('Other/Bounds Ratio', (other_end - other_start)/(bc_end - bc_start))
print('len_points/len_shapes', len(points1) / len(shape_pos))


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