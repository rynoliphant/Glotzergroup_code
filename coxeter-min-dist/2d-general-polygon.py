import coxeter
import numpy as np
import numpy.linalg as LA

def point_to_edge_distance (point: np.ndarray, vert: np.ndarray, edge_vector: np.ndarray) -> np.ndarray:
    '''
    Calculates the distances between a single point and several varying lines.

    Args:
        point (np.ndarray): the position of the point (shape=(3,)) 
        vert (np.ndarray): positions of the points that lie on each corresponding line (shape=(n,3))
        edge_vector (np.ndarray): the vectors that describe each line (shape=(n,3))

    Returns:
        np.ndarray: distances
    '''
    edge_units = edge_vector / LA.norm(edge_vector, axis=1).reshape(len(edge_vector), 1)
    dist = LA.norm(((vert - point) - (np.sum((vert-point)*edge_units, axis=1).reshape(len(edge_vector),1) *edge_units)), axis=1)
    return dist

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

def bounds_for_images_2d (
        bounds: np.ndarray,
        constraint: np.ndarray, 
        lx: float,
        ly: float,
        a: np.ndarray,
        b: np.ndarray,
) -> np.ndarray:
    '''
    Caclulates the bounds for all periodic images given a constraint (ie. vert_constraint, edge_constraint). This is for 2D systems.

    Args:
        bounds (np.ndarray): Bounds associated with the constraint for the shape in the periodic box
        constraint (np.ndarray): Constraint associated with the shape and its orientation
        lx (float): edge length
        ly (float): edge length
        a (np.ndarray): box unit vector
        b (np.ndarray): box unit vector

    Returns:
        np.ndarray: bounds for all images
    '''

    big_bounds = np.array([
        bounds, bounds, bounds, bounds, bounds, bounds, bounds, bounds, bounds,
    ])
    big_constraint = np.array([
        constraint, constraint, constraint, constraint, constraint, constraint, constraint, constraint, constraint,
    ])

    ijk_list = np.array([
        [-1,-1], [-1,0], [-1,1],
        [0,-1], [0,0], [0,1],
        [1,-1], [1,0], [1,1],
    ])

    a_matrix = np.tile(a,9).reshape((9,2))
    b_matrix = np.tile(b,9).reshape((9,2))
    i_array = np.repeat(ijk_list[:,0],2).reshape((9,2))
    j_array = np.repeat(ijk_list[:,1],2).reshape((9,2))

    image_diff = (lx*i_array*a_matrix + ly*j_array*b_matrix).reshape(9,1,1,2)

    new_bounds = big_bounds + (np.sum(big_constraint * image_diff, axis=3))
    return new_bounds

def min_dist_multiprocessing_2d (
        coord,
        shp_x,
        shp,
        image_pos,
        image_diff,
        img_vert_bounds,
        vert_constraint,
        img_verts,
        img_edge_bounds,
        edge_constraint,
        img_edges,
        img_ev_neighbors,
        r_cut_q,
        r_cut,
):
    '''
    Solves for the minimum distance between a point and a 2D shape assuming periodic boundary conditions are true.

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

        r_cut_q (bool): -------
        r_cut (float): -------

    Returns:
        float: min distance
    '''

    if r_cut_q:
        r_check = distance_point_point(coord, shp_x)
        if r_check > r_cut:
            dist = -1
            return dist
        
    shp_verts = shp.vertices[:,:2]

    #Finding the 4 nearest images to the coord
    dist_coord_images = np.sqrt(np.sum(((image_pos-coord)**2), axis=1))
    nearest_images_ind = np.argsort(dist_coord_images)
    near_img_ind = nearest_images_ind[:4]

    min_dist_list = np.array([])

    #Solving for the distances between the coord and any relevant vertices
    vert_bool = np.all(img_vert_bounds[near_img_ind] >= vert_constraint.dot(coord), axis=2)
    if np.any(vert_bool):
        vert_bool_img = np.any(vert_bool, axis=1)

        min_dist = LA.norm(-1*(img_verts[near_img_ind][vert_bool] + image_diff[near_img_ind][vert_bool_img] + shp_x) + coord, axis=1)
        min_dist_list = np.append(min_dist_list, min_dist)

#   Solving for the distances between the coord and any relevant edges
    edge_bool = np.all(img_edge_bounds[near_img_ind] >= edge_constraint.dot(coord), axis=2)
    if np.any(edge_bool):
        edge_bool_img = np.any(edge_bool, axis=1)
        vert_on_edge = shp_verts[img_ev_neighbors[near_img_ind][edge_bool][:,0]] + shp_x + image_diff[near_img_ind][edge_bool_img]
        min_dist = point_to_edge_distance(coord, vert_on_edge, img_edges[near_img_ind][edge_bool])
        min_dist_list = np.append(min_dist_list, min_dist)

    #Checking if the coord is inside the shape (or inside one of its images) and setting the min distance to zero
    if np.any(shp.is_inside(-1*image_diff - shp_x + coord)):
        min_dist=np.array([0])
        min_dist_list = np.append(min_dist_list, min_dist)

    true_min_dist = np.min(min_dist_list)

    return true_min_dist

def min_distance_calc_2d(
    shapes: coxeter.shapes.ConvexPolygon,
    x_position: np.ndarray,
    y_position: np.ndarray,
    shapes_pos: np.ndarray,
    insphere_diameter: float,
    a: np.ndarray,
    b: np.ndarray,
    lx: float,
    ly: float,
    rshift: float,
    r_exponent: float,
    pool: Pool,
    r_cut_q: bool = False,
    r_cut: float = 0,
    sphere_q: bool = False,
    epsilon: float = 1e-12,
) -> np.ndarray:
    '''
    Calculates the minimum distance between all the grid points and each individual shape in the system, 
    assuming periodic boundary conditions are true, and polygons are convex. This is for 2D systems.

    Args:
        shapes (coxeter.shapes.convex_polyhedron.ConvexPolyhedron): the shape that is being looked at
        x_position (np.ndarray): the x positions of all the grid points
        y_position (np.ndarray): the y positions of all the grid points
        shapes_pos (np.ndarray): the positions of all the shapes
        insphere_diameter (float): the insphere_diameter of the shapes with the assumption that all shapes are the same type
        a (np.ndarray): box unit vector
        b (np.ndarray): box unit vector
        lx (float): edge length
        ly (float): edge length
        rshift (float):
        r_exponent (float): Value of the exponent of the distances (been referred to as gamma in discussions)
        pool (Pool): multiprocessing pool
        r_cut_q (bool): ------
        r_cut (float): ------
        sphere_q (bool): is the shape a circle? Default is False
        epsilon (float): ------

    Returns:
        np.ndarray: min distances
    '''
    dist_squared_sum = np.zeros(len(x_position))
    dist_inverse_squared_sum = np.zeros(len(x_position))
    dist_outshape = np.ones(len(x_position))

    #----- Combining the x,y positions -----
    x_position = x_position.reshape(len(x_position), 1)
    y_position = y_position.reshape(len(y_position), 1)
    coordinates = np.hstack((x_position, y_position))

    #----- Setting up images -----
    ijk_list = np.array([
        [-1,-1], [-1,0], [-1,1],
        [0,-1], [0,0], [0,1],
        [1,-1], [1,0], [1,1],
    ])

    a_matrix = np.tile(a,9).reshape((9,2))
    b_matrix = np.tile(b,9).reshape((9,2))
    i_array = np.repeat(ijk_list[:,0],2).reshape((9,2))
    j_array = np.repeat(ijk_list[:,1],2).reshape((9,2))

    image_diff = lx*i_array*a_matrix + ly*j_array*b_matrix

    if sphere_q:
        for shp_i,shp_x in enumerate(shapes_pos):
            dist_coord_images = np.sqrt(np.sum((np.tile((image_diff + shp_x).reshape(9,1,2), (1,len(coordinates),1)) - np.tile(coordinates.reshape(1,len(coordinates),2), (9,1,1)) )**2, axis=2))
            dist = np.min(dist_coord_images, axis=0) - insphere_diameter/2

            # regularize
            dist = dist / (insphere_diameter)

            current_outshape = (dist > 0).astype(int)
            dist_outshape = dist_outshape * current_outshape

            r_cut_bool = ((dist > 0).astype(int))

            dist = dist + rshift + epsilon

            dist_squared_sum += ((dist**r_exponent) * r_cut_bool)

            if np.any((dist)==0):
                print("Uh oh, you're going to divide by zero!!!")
            dist_inverse_squared_sum += (((dist)**(-r_exponent)) * r_cut_bool)
    
    else:
        for shp_i,shp_x in enumerate(shapes_pos):

            #----- Setting up the necessary variables to get the zones ------
            shp = shapes[shp_i]
            n_verts = shp.num_vertices #number of vertices == number of edges

            shp_verts = shp.vertices[:,:2] #shp.vertices returns an array of shape (N,3), we want a shape of (N,2) | lists vertices counterclockwise
            shp_edges = shp_verts - np.append(shp_verts[1:], shp_verts[0].reshape(1,2), axis=0) #edges point clockwise

            shp_edge_vert = np.append(np.arange(1,n_verts +1).reshape(n_verts,1), np.arange(0,n_verts).reshape(n_verts,1), axis=1) #column [0]: o-->  | column [1]: o<--   | shp.edges[1] - shp.edges[0] == shp.edge_vectors
            shp_edge_vert[-1,0] = 0


            #----- Getting the constraint matrix and bounds needed to define the space partitioned zones -----
            vert_constraint = np.append( -1*shp_edges.reshape(n_verts, 1,2), np.append(shp_edges[-1].reshape(1,2), shp_edges[:-1], axis=0).reshape(n_verts,1,2), axis=1)
            vert_bounds = np.sum(vert_constraint * (shp_verts+shp_x).reshape(n_verts,1,2), axis=2)
            img_vert_bounds = bounds_for_images_2d(vert_bounds, vert_constraint, lx, ly, a, b)



            edges_90 = np.append(shp_edges[:,1].reshape(n_verts,1), shp_edges[:,0].reshape(n_verts,1), axis=1)
            edges_90 = edges_90 * np.array([1,-1])
            edge_constraint = np.append( shp_edges.reshape(n_verts,1,2) , -1*shp_edges.reshape(n_verts,1,2), axis=1 )
            edge_constraint = np.append( edge_constraint, edges_90.reshape(n_verts,1,2) , axis=1)

            edge_bounds = np.zeros((n_verts, 3))
            edge_bounds[:,0] = np.sum(edge_constraint[:,0] *(shp_verts+shp_x), axis=1)
            edge_bounds[:,1] = np.sum(edge_constraint[:,1] *(np.append(shp_verts[1:], shp_verts[0].reshape(1,2), axis=0)+shp_x), axis=1)
            edge_bounds[:,2] = np.sum(edge_constraint[:,2] *(np.append(shp_verts[1:], shp_verts[0].reshape(1,2), axis=0)+shp_x), axis=1)
            img_edge_bounds = bounds_for_images_2d(edge_bounds, edge_constraint, lx, ly, a, b)

            #Zones are defined such that if a point p lies within zone A, it satisfies the condition: A_constraint.dot(p) <= A_bounds |or| A_bounds >= A_constraint.dot(p)
            #The constraint matrices contain the normals (pointing outward from the zone) of each planar boundary


            #----- Accounting for images -----
            image_pos = image_diff + shp_x + shp.centroid[:2]
            img_verts = np.tile(shp_verts.reshape(1, n_verts,2), (9,1,1))
            img_edges = np.tile(shp_edges.reshape(1, n_verts,2), (9,1,1))
            img_ev_neighbors = np.tile(shp_edge_vert.reshape(1, n_verts, 2), (9,1,1))
            # img_face_centroids = np.tile(face_centroids.reshape(1,n_faces,3), (27,1,1))

            #----- Iterating through the coordinates to find the min distance between each one and shapes[shp_i] -----
            dist = pool.starmap(
                min_dist_multiprocessing_2d,
                [
                    (
                        coord,
                        shp_x,
                        shp,
                        image_pos,
                        image_diff,
                        img_vert_bounds,
                        vert_constraint,
                        img_verts,
                        img_edge_bounds,
                        edge_constraint,
                        img_edges,
                        img_ev_neighbors,
                        r_cut_q,
                        r_cut,
                    )
                    for coord in coordinates
                ],
                chunksize=None,
            )

            dist = np.asarray(dist)
            # print(dist)
            # regularize
            dist = dist / (insphere_diameter)

            current_outshape = (dist != 0).astype(int)
            dist_outshape = dist_outshape * current_outshape

            r_cut_bool = ((dist > 0).astype(int))

            dist = dist + rshift + epsilon

            dist_squared_sum += ((dist**r_exponent) * r_cut_bool)

            if np.any((dist)==0):
                print("Uh oh, you're going to divide by zero!!!")
            dist_inverse_squared_sum += (((dist)**(-r_exponent)) * r_cut_bool)

            # print(dist_squared_sum)
            # print(dist_inverse_squared_sum)
            # print(dist_outshape)


    return dist_squared_sum, dist_inverse_squared_sum, dist_outshape
