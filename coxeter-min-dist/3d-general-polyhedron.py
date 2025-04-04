import coxeter
import numpy as np
import numpy.linalg as LA
import awkward as ak


#TODO: remove awkward arrays while retaining the speed???
def get_edge_face_neighbors (shape: coxeter.shapes.ConvexPolyhedron) -> np.ndarray:
    '''
    Get the indices for the faces that are adjacent to each edge.

    Args:
        shape (coxeter.shapes.convex_polyhedron.ConvexPolyhedron): the shape that is being looked at

    Returns:
        np.ndarray: the indices of the nearest faces for each edge (shape=(n_edges, 2))
    '''

    faces = ak.Array(shape.faces) #awkward array containing the indices of the vertices associated with each face; starts with the lowest index then lists them counterclockwise
    faces_one_vertex = ak.to_numpy(faces[:,0])
    faces_len = shape.num_faces
    num_edges = shape.num_edges

    #appending the first value of each vertex list and a -1 to the end of each list of vertices, then flattening the awkward array (the -1 indicates a change in face)
    faces_plus = ak.concatenate((faces, faces_one_vertex.reshape(faces_len,1), np.array([-1]*faces_len).reshape(faces_len,1)), axis=1)
    faces_flat = ak.to_numpy(ak.flatten(faces_plus))

    #creating a matrix where each row corresponds to an edge that contains the indices of its two corresponding vertices (a -1 index indicates a change in face)
    list_len = len(faces_flat)
    face_edge_mat = np.block([faces_flat[:-1].reshape(list_len-1,1), faces_flat[1:].reshape(list_len-1,1)]) 

    #finding the number of edges associated with each face
    fe_mat_inds = np.arange(0,list_len-1,1)
    find_num_edges = fe_mat_inds[(fe_mat_inds==0) + (np.any(face_edge_mat==-1, axis=1))]
    find_num_edges[:][0] = -1
    find_num_edges = find_num_edges.reshape(faces_len,2)
    face_num_edges = find_num_edges[:,1] - find_num_edges[:,0] -1

    #repeating each face index for the number of edges that are associated with it; length equals num_edges * 2
    face_correspond_inds = np.repeat(np.arange(0,faces_len,1), face_num_edges)

    #shape.edges lists the indices of the edge vertices lowest to highest. edges1_reshape lists the indices of the edge vertices highest to lowest
    edges1_reshape = np.hstack(((shape.edges[:,1]).reshape(num_edges,1), (shape.edges[:,0]).reshape(num_edges,1)))

    #For the new_edge_ind_bool: rows correspond with the face_correspond_inds and columns correpond with the edge index; finding the neighboring faces for each edge
    true_face_edge_mat = np.tile(face_edge_mat[np.all(face_edge_mat!=-1, axis=1)].reshape(num_edges*2,1,2), (1, num_edges,1))
    new_edge_ind_bool0 = np.all(true_face_edge_mat == shape.edges.reshape(1, num_edges, 2), axis=2).astype(int) #faces to the LEFT of edges if edges are oriented pointing upwards
    new_edge_ind_bool1 = np.all(true_face_edge_mat == edges1_reshape.reshape(1,num_edges,2), axis=2).astype(int) #faces to the RIGHT of edges if edges are oriented pointing upwards

    #tiling face_correspond_inds so it can be multiplied to the new_edge_ind_bool0 and new_edge_ind_bool1
    new_face_corr_inds = np.tile(face_correspond_inds.reshape(2*num_edges,1), (1,num_edges))

    #getting the face indices and completing the edge-face neighbors
    ef_neighbor0 = np.sum(new_face_corr_inds*new_edge_ind_bool0, axis=0).reshape(num_edges, 1) #faces to the LEFT
    ef_neighbor1 = np.sum(new_face_corr_inds*new_edge_ind_bool1, axis=0).reshape(num_edges, 1) #faces to the RIGHT
    ef_neighbor = np.hstack((ef_neighbor0, ef_neighbor1))

    return ef_neighbor, faces_one_vertex

def point_to_edge_distance (point: np.ndarray, vert: np.ndarray, edge_vector: np.ndarray) -> np.ndarray:
    '''
    Calculates the distances between several points and several varying lines.

    Args:
        point (np.ndarray): the positions of the points (shape=(n_points,3)) 
        vert (np.ndarray): positions of the points that lie on each corresponding line (shape=(n_lines,3))
        edge_vector (np.ndarray): the vectors that describe each line (shape=(n_lines,3))

    Returns:
        np.ndarray: distances (n_points, n_lines)
    '''
    n_points = len(point)
    n_lines = len(edge_vector)

    vert = np.repeat(np.reshape(vert, (1,n_lines,3)), n_points, axis=0)
    point = np.reshape(point, (n_points, 1, 3))

    edge_units = edge_vector / LA.norm(edge_vector, axis=1).reshape(len(edge_vector), 1)
    big_edge_units = np.repeat(np.reshape(edge_units, (1,n_lines,3)), n_points, axis=0)

    dist = LA.norm(((vert - point) - (np.sum((vert-point)*edge_units, axis=2).reshape(n_points, n_lines,1) *big_edge_units)), axis=2)
    return dist

def point_to_face_distance(point: np.ndarray, vert: np.ndarray, face_normal: np.ndarray) -> np.ndarray:
    '''
    Calculates the distances between a single point and several varying planes.

    Args:
        point (np.ndarray): the position of the point (shape=(n_points, 3))
        vert (np.ndarray): points that lie on each corresponding plane (shape=(n_faces,3))
        face_normal (np.ndarray): the normals that describe each plane (shape=(n_faces,3))

    Returns:
        np.ndarray: distances (n_points, n_faces)
    '''
    n_points = len(point)
    n_faces = len(face_normal)

    vert = np.repeat(np.reshape(vert, (1,n_faces,3)), n_points, axis=0)
    point = np.reshape(point, (n_points, 1, 3))
    
    vert_point_vect = -1*vert + point
    face_unit = face_normal / LA.norm(face_normal, axis=1).reshape(len(face_normal), 1)
    dist = np.sum(vert_point_vect*face_unit, axis=2)
    return dist

#TODO: remove shp_x and add it to the distance_to_surface function
def get_vert_zones (
        shp_verts: np.ndarray,
        shp_edges: np.ndarray,
        shp_edge_vert: np.ndarray, 
        n_verts: int,
        n_edges: int,
        x: np.ndarray
) -> np.ndarray:
    '''
    Get the constraints and bounds needed to partition the volume surrounding a shape into zones 
    where the min distance from any point that is within a vert_zone is the distance between the 
    point and one of the vertices.

    Args:
        shp_verts (np.ndarray): vertices of the shape
        shp_edges (np.ndarray): the vectors that correspond to each edge of the shape
        shp_edge_vert (np.ndarray): the indices of the vertices that correspond to each edge (shape=(n_edge, 2))
        n_verts (int): the number of total vertices for this shape
        n_edges (int): the number of total edges for this shape
        x (np.ndarray): position of the shape 

    Returns:
        np.ndarray: constraint (shape=(n_verts, n_edges, 3)), np.ndarray: bounds (shape=(n_verts, n_edges))
    '''
    #Tiling for set up
    nverts_edge_vert0 = np.tile(shp_edge_vert[:,0], (n_verts, 1))
    nverts_edge_vert1 = np.tile(shp_edge_vert[:,1], (n_verts, 1))
    vert_inds = np.arange(0, n_verts, 1).reshape((n_verts, 1))
    nverts_tile_edges = np.tile(shp_edges, (n_verts, 1)).reshape((n_verts, n_edges, 3))

    #Creating the bools needed to get the edges that correspond to each vertex
    evbool0 = np.expand_dims(nverts_edge_vert0 == vert_inds, axis=2)
    evbool1 = np.expand_dims(nverts_edge_vert1 == vert_inds, axis=2)

    #Applying the bools to find the corresponding edges
    vert_pos = nverts_tile_edges * evbool0.astype(int)
    vert_neg = nverts_tile_edges * evbool1.astype(int) * (-1) 

    #Concatenating the arrays together to make the vertex constraint array | shape:(n_verts, n_edges, 3)
    vert_constraint = ak.to_numpy(vert_pos + vert_neg)
    
    #Building the boundary conditions | shape: (n_verts, n_edges)
    vert_bounds = np.sum(vert_constraint * (shp_verts+x).reshape(n_verts,1,3), axis=2)

    return vert_constraint, vert_bounds

def get_edge_zones (
        shp_verts: np.ndarray,
        shp_edges: np.ndarray,
        shp_faces: np.ndarray, 
        shp_edge_vert: np.ndarray,
        shp_edge_face: np.ndarray, 
        n_edges: int,
        x: np.ndarray
) -> np.ndarray:
    '''
    Get the constraints and bounds needed to partition the volume surrounding a shape into zones 
    where the min distance from any point that is within an edge_zone is the distance between the 
    point and one of the edges.

    Args:
        shp_verts (np.ndarray): vertices of the shape
        shp_edges (np.ndarray): the vectors that correspond to each edge of the shape
        shp_faces (np.ndarray): the normals that correspond to each face of the shape
        shp_edge_vert (np.ndarray): the indices of the vertices that correspond to each edge (shape=(n_edge, 2))
        shp_edge_face (np.ndarray): the indices of the faces that correspond to each edge (shape=(n_edge, 2))
        n_edges (int): the number of total edges for this shape
        x (np.ndarray): position of the shape

    Returns:
        np.ndarray: constraint (shape=(n_edges, 4, 3)), np.ndarray: bounds (shape=(n_edges, 4))
    '''
    #Set up
    edge_constraint = np.zeros((n_edges, 4, 3))
    edge_bounds = np.zeros((n_edges, 4))

    #Calculating the normals of the planar boundaries
    edge_constraint_col_1 = shp_edges
    edge_constraint_col_2 = -1*shp_edges
    edge_constraint_col_3 = np.cross(shp_edges, shp_faces[shp_edge_face[:,1]])
    edge_constraint_col_4 = -1*np.cross(shp_edges, shp_faces[shp_edge_face[:,0]])

    edge_constraint[:,0] = edge_constraint_col_1
    edge_constraint[:,1] = edge_constraint_col_2
    edge_constraint[:,2] = edge_constraint_col_3
    edge_constraint[:,3] = edge_constraint_col_4


    #Bounds
    edge_verts = np.zeros((n_edges, 2, 3))
    edge_verts[:,0] = shp_verts[shp_edge_vert[:,0]] + x
    edge_verts[:,1] = shp_verts[shp_edge_vert[:,1]] + x

    edge_bounds_1 = np.sum(edge_constraint[:,0] *(edge_verts[:,1]), axis=1)
    edge_bounds_2 = np.sum(edge_constraint[:,1] *(edge_verts[:,0]), axis=1)
    edge_bounds_3 = np.sum(edge_constraint[:,2] *(edge_verts[:,0]), axis=1)
    edge_bounds_4 = np.sum(edge_constraint[:,3] *(edge_verts[:,0]), axis=1)

    edge_bounds[:,0] = edge_bounds_1
    edge_bounds[:,1] = edge_bounds_2
    edge_bounds[:,2] = edge_bounds_3
    edge_bounds[:,3] = edge_bounds_4

    return edge_constraint, edge_bounds

def get_face_zones (
        shp_verts: np.ndarray,
        shp_edges: np.ndarray,
        shp_faces: np.ndarray, 
        shp_edge_vert: np.ndarray,
        shp_edge_face: np.ndarray, 
        n_edges: int,
        n_faces: int, 
        x: np.ndarray,
        face_one_vertex: np.ndarray
) -> np.ndarray:
    '''
    Get the constraints and bounds needed to partition the volume surrounding a shape into zones 
    where the min distance from any point that is within a face_zone is the distance between the 
    point and one of the faces.

    Args:
        shp_verts (np.ndarray): vertices of the shape
        shp_edges (np.ndarray): the vectors that correspond to each edge of the shape
        shp_faces (np.ndarray): the normals that correspond to each face of the shape
        shp_edge_vert (np.ndarray): the indices of the vertices that correspond to each edge (shape=(n_edge, 2))
        shp_edge_face (np.ndarray): the indices of the faces that correspond to each edge (shape=(n_edge, 2))
        n_edges (int): the number of total edges for this shape
        n_faces (int): the number of total faces for this shape
        x (np.ndarray): position of the shape
        face_centroids (np.ndarray): positions of the centroids for each of the faces

    Returns:
        np.ndarray: constraint (shape=(n_faces, n_edges+1, 3)), np.ndarray: bounds (shape=(n_faces, n_edges+1))
    '''
    #Setting up
    nfaces_tile_edges = np.tile(shp_edges, (n_faces, 1)).reshape((n_faces, n_edges, 3)) #Edge vectors tiled
    nfaces_tile_efneighbors = np.tile(shp_edge_face, (n_faces, 1)) #Tiling shp_edge_face so that it is now shape (n_faces, n_edges, 2)
    efneighbors0 = nfaces_tile_efneighbors[:,0].reshape((n_faces, n_edges)) #breaking nfaces_tile_efneighbors into the first column [0]
    efneighbors1 = nfaces_tile_efneighbors[:,1].reshape((n_faces, n_edges)) #second column [1] of nfaces_tile_efneighbors
    faces_inds = np.arange(0, n_faces, 1).reshape((n_faces, 1)) #Making an array containing the indices of the faces

    #Creating bools to find the edges that correspond to each face
    efbool0 = (efneighbors0 == faces_inds).astype(int)
    efbool1 = (efneighbors1 == faces_inds).astype(int)

    #Applying the bools
    ak_edges_mask0 = nfaces_tile_edges * efbool0.reshape((n_faces, n_edges, 1))
    ak_edges_mask1 = nfaces_tile_edges * efbool1.reshape((n_faces, n_edges, 1))
    faces_repeat = np.repeat(shp_faces, n_edges, axis=0).reshape((n_faces, n_edges, 3)) 

    #Calculating the parts of face_constraint: the normals of the planar boundaries
    pos_constraints = np.cross(ak_edges_mask0, faces_repeat)
    neg_constraints = -1* np.cross(ak_edges_mask1, faces_repeat)
    face_normals = -1* shp_faces

    face_constraint = np.zeros((n_faces, n_edges+1, 3))
    face_constraint[:,0] = face_normals
    face_constraint[:,1:] = pos_constraints + neg_constraints

    #bounds
    nfaces_tile_evneighbors = np.tile(shp_edge_vert, (n_faces, 1)).reshape((n_faces, n_edges, 2))
    pos_face_verts = (shp_verts[(nfaces_tile_evneighbors[:,:,0]*efbool0)] + x)
    neg_face_verts = (shp_verts[(nfaces_tile_evneighbors[:,:,1]*efbool1)] + x)

    print(pos_face_verts.shape)

    pos_bounds = np.sum(pos_constraints * pos_face_verts, axis=2)
    neg_bounds = np.sum(neg_constraints * neg_face_verts, axis=2)
    # norm_distances = (face_centroids + x)
    normal_bounds = np.sum(face_normals*(shp_verts[face_one_vertex]+x), axis=1)

    face_bounds = np.zeros((n_faces, n_edges+1))
    face_bounds[:,0] = normal_bounds
    face_bounds[:,1:] = pos_bounds + neg_bounds

    return face_constraint, face_bounds

def distance_to_surface (
        coord: np.ndarray,
        shp_x: np.ndarray,

        #v---will be part of self (coxeter shape class)---v
        shp: coxeter.shapes.ConvexPolyhedron,
        vert_bounds: np.ndarray,
        vert_constraint: np.ndarray,
        edge_bounds: np.ndarray,
        edge_constraint: np.ndarray,
        ev_neighbors: np.ndarray, 
        face_bounds: np.ndarray,
        face_constraint: np.ndarray,
        face_one_vertex: np.ndarray
) -> float:
    '''
    Solves for the minimum distance between a point and a shape assuming periodic boundary conditions are true.

    Args:
        coord (np.ndarray): position of the point
        shp_x (np.ndarray): position of the shape
        shp (coxeter.shapes.convex_polyhedron.ConvexPolyhedron): the shape that is being looked at
        image_pos (np.ndarray): the positions of the shape images
        image_diff (np.ndarray): the vector differences between the shape and all of its images

        img_vert_bounds (np.ndarray): the bounds associated with the vertice zones of all the images
        vert_constraint (np.ndarray): the constraint matrices associated with the vertice zones
        img_verts (np.ndarray): shp_verts tiled so that it is the same shape as image_diff

        img_edge_bounds (np.ndarray): the bounds associated with the edge zones of all the images
        edge_constraint (np.ndarray): the constraint matrices associated with the edge zones
        img_edges (np.ndarray): shp_edges tiled so that it is the same shape as image_diff
        img_ev_neighbors (np.ndarray): shp_edge_vert tiled so that it is the same shape as image_diff

        img_face_bounds (np.ndarray): the bounds associated with the face zones of all the images
        face_constraint (np.ndarray): the constraint matrices associated with the face zones
        img_faces (np.ndarray): shp_faces tiled so that it is the same shape as image_diff
        img_face_centroids (np.ndarray): face_centroids tiled so that it is the same shape as image_diff

        r_cut_q (bool): -------
        r_cut (float): -------

    Returns:
        float: min distance
    '''
    
    # if r_cut_q:
    #     r_check = distance_point_point(coord, shp_x)
    #     if r_check > r_cut:
    #         dist = -1
    #         return dist

    n_coords = len(coord)
    n_verts = shp.num_vertices
    n_edges = shp.num_edges
    n_faces = shp.num_faces


    coord_trans = np.transpose(coord)

    max_value = 3*np.max(LA.norm(coord - shp_x, axis=1))

    # #Finding the 8 nearest images to the coord
    # dist_coord_images = np.sqrt(np.sum(((image_pos-coord)**2), axis=1))
    # nearest_images_ind = np.argsort(dist_coord_images)
    # near_img_ind = nearest_images_ind[:8]

    min_dist_list = np.ones((len(coord),1))*max_value

    #Solving for the distances between the coord and any relevant vertices
    vert_bool = np.all((vert_constraint @ coord_trans) <= np.reshape(vert_bounds, (n_verts,n_edges,1)), axis=1) #<--- shape = (number_of_vertex_zones, number_of_coordinates)
    if np.any(vert_bool):

        vert_any_bool = np.any(vert_bool, axis=1)

        vert_dist = LA.norm(np.repeat(np.reshape(coord, (len(coord),1, 3)), len(shp.vertices[vert_any_bool]), axis=1) - (shp.vertices[vert_any_bool] + shp_x), axis=2)
        vert_dist = vert_dist + max_value*(np.transpose(vert_bool[vert_any_bool]) == False).astype(int)
        vert_dist = np.min(vert_dist, axis=1).reshape(n_coords, 1)
        min_dist_list = np.concatenate((min_dist_list, vert_dist), axis=1)


        # vert_bool_img = np.any(vert_bool, axis=1)
        # min_dist = LA.norm(-1*(img_verts[near_img_ind][vert_bool] + image_diff[near_img_ind][vert_bool_img] + shp_x) + coord, axis=1)
        # min_dist_list = np.append(min_dist_list, min_dist)

#   Solving for the distances between the coord and any relevant edges
    edge_bool = np.all((edge_constraint @ coord_trans) <= np.reshape(edge_bounds, (n_edges, 4, 1)), axis=1) #<--- shape = (number_of_edge_zones, number_of_coordinates)
    if np.any(edge_bool):

        edge_any_bool = np.any(edge_bool, axis=1)

        vert_on_edge = shp.vertices[ev_neighbors[edge_any_bool][:,0]] + shp_x

        edge_dist = point_to_edge_distance(coord, vert_on_edge, shp.edge_vectors[edge_any_bool])
        edge_dist = edge_dist + max_value*(np.transpose(edge_bool[edge_any_bool]) == False).astype(int)
        edge_dist = np.min(edge_dist, axis=1).reshape(n_coords, 1)
        min_dist_list = np.concatenate((min_dist_list, edge_dist), axis=1)



        # edge_bool_img = np.any(edge_bool, axis=1)
        # vert_on_edge = shp_verts[img_ev_neighbors[near_img_ind][edge_bool][:,0]] + shp_x + image_diff[near_img_ind][edge_bool_img]
        # min_dist = point_to_edge_distance(coord, vert_on_edge, img_edges[near_img_ind][edge_bool])
        # min_dist_list = np.append(min_dist_list, min_dist)

    #Solving for the distances between the coord and any relevant faces
    face_bool = np.all((face_constraint @ coord_trans) <= np.reshape(face_bounds, (n_faces, n_edges+1, 1)), axis=1) #<--- shape = (number_of_face_zones, number_of_coordinates)
    if np.any(face_bool):

        face_any_bool = np.any(face_bool, axis=1)

        vert_on_face = (shp.vertices[face_one_vertex][face_any_bool]) + shp_x
        face_dist = point_to_face_distance(coord, vert_on_face, shp.normals[face_any_bool])
        face_dist = face_dist + max_value*(np.transpose(face_bool[face_any_bool]) == False).astype(int)
        face_dist = np.min(face_dist, axis=1).reshape(n_coords, 1)
        min_dist_list = np.concatenate((min_dist_list, face_dist), axis=1)


        # face_bool_img = np.any(face_bool, axis=1)
        # vert_on_face = img_face_centroids[near_img_ind][face_bool] + shp_x + image_diff[near_img_ind][face_bool_img] 
        # min_dist = point_to_face_distance(coord, vert_on_face, img_faces[near_img_ind][face_bool])
        # min_dist_list = np.append(min_dist_list, min_dist)

    #Checking if the coord is inside the shape (or inside one of its images) and setting the min distance to zero
    inside_bool = shp.is_inside(coord - shp_x)
    if np.any(inside_bool):
        inside_dist = (max_value*(inside_bool == False).astype(int)).reshape(n_coords, 1)
        min_dist_list = np.concatenate((min_dist_list, inside_dist), axis=1)

    true_min_dist = np.min(min_dist_list, axis=1)

    return true_min_dist


###--- Testing distance_to_surface --- ###

cube_verts = np.array([[1,1,1], [-1,1,1], [1,-1,1], [1,1,-1], [-1,-1,1], [-1,1,-1], [1,-1,-1], [-1,-1,-1]])#,   [0,0,3], [0,3,0]])

cube_faces = [
    [0,1,8],
    [1,4,8],
    [4,2,8],
    [2,0,8],
    [1,0,9],
    [5,1,9],
    [3,5,9],
    [0,3,9],
    [1,5,7,4],
    [0,2,6,3],
    [2,4,7,6],
    [3,6,7,5]
]

# cube = coxeter.shapes.Polyhedron(vertices=cube_verts, faces=cube_faces)

cube = coxeter.shapes.ConvexPolyhedron(vertices=cube_verts)
print('Number of Vertices:', cube.num_vertices)
print('Number of Edges:', cube.num_edges)
print('Number of Faces:', cube.num_faces)
print('Faces List:', cube.faces)

shp_x = np.array([3,3,3])

shp_faces = cube.normals #normals of the faces
shp_verts = cube.vertices 
shp_edges = cube.edge_vectors 

shp_edge_vert = cube.edges #column [0]: o-->  | column [1]: o<--   | shp.edges[1] - shp.edges[0] == shp.edge_vectors
shp_edge_face, face_one_vertex = get_edge_face_neighbors(cube) #column[0] ↑ column[1]
#^-- if the edges are oriented facing upwards, shp_edge_face[0] contains faces on the left of the edges, and shp_edge_face[1] contains faces on the right of the edges.

n_faces = cube.num_faces #number of faces
n_edges = cube.num_edges #number of edges
n_verts = cube.num_vertices #number of vertices

#----- Getting the constraint matrix and bounds needed to define the space partitioned zones -----
vert_constraint, vert_bounds = get_vert_zones (shp_verts, shp_edges, shp_edge_vert, n_verts, n_edges, shp_x)

edge_constraint, edge_bounds = get_edge_zones (shp_verts, shp_edges, shp_faces, shp_edge_vert, shp_edge_face, n_edges, shp_x)

face_constraint, face_bounds = get_face_zones (shp_verts, shp_edges, shp_faces, shp_edge_vert, shp_edge_face, n_edges, n_faces, shp_x, face_one_vertex)


x_coords = np.array([[5,5,1],[5,4,2],[3,4.5,4.5],[3,4,5],[3,3,6],[3.5,2.5,3]])

distance = distance_to_surface(x_coords, shp_x, cube, vert_bounds, vert_constraint, edge_bounds, edge_constraint, shp_edge_vert, face_bounds, face_constraint, face_one_vertex)

print('Distance:', distance)