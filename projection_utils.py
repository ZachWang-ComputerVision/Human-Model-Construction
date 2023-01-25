import numpy as np
from matplotlib import path


def get_projected_xy(xyz, vertex, proj_mat):
  uv = []
  for i in vertex:
    # n = np.append(xyz[i], 1.)
    n = np.expand_dims(xyz[i], axis=1)
    cam_xyz = proj_mat @ n
    uv.append(cam_xyz)
  uv = np.array(uv)
  uv = np.squeeze(uv)
  return uv



def get_visible_vertex(vertex_xyz, view_vec, faces):
  v = vertex_xyz

  visible_faces = []
  for face in faces:
    v_idx1, v_idx2, v_idx3 = face
    ver1 = v[v_idx1]
    ver2 = v[v_idx2]
    ver3 = v[v_idx3]

    fv = np.cross((ver2 - ver1), (ver3 - ver2))
    norm = fv @ view_vec
    # norm = np.sum(fv)
    if norm < 0.:
      visible_faces.append(face)

  visible_faces = np.array(visible_faces)

  visible_vertex = np.reshape(visible_faces, (1, -1))
  visible_vertex = np.unique(visible_vertex)
  return visible_vertex, visible_faces



# uv is the 2D projected xy
def get_vertex_map(uv, visible_vertex, visible_faces,
                   projected_mid_x, projected_mid_y,
                   mid_x, mid_y, x_scale, y_scale, resolution):

  vertex_map = np.zeros(resolution)
  vertex_map[:] = np.nan

  uv[:, 0] = (uv[:, 0] - projected_mid_x) / x_scale + mid_x
  uv[:, 1] = (uv[:, 1] - projected_mid_y) / y_scale + mid_y

  z_buffer = np.zeros(resolution)
  z_buffer[:] = np.Inf

  for i, face in enumerate(visible_faces):
    v1, v2, v3 = face
    # this should be all vertex, not just visible ones
    v1_idx = np.where(visible_vertex == v1)
    v2_idx = np.where(visible_vertex == v2)
    v3_idx = np.where(visible_vertex == v3)

    x1, y1, z1 = uv[v1_idx][0]
    x2, y2, z2 = uv[v2_idx][0]
    x3, y3, z3 = uv[v3_idx][0]
    
    x1, y1, x2, y2, x3, y3 = int(x1), int(y1), int(x2), int(y2), int(x3), int(y3)

    target_area = path.Path([
                    tuple([x1, y1]),
                    tuple([x2, y2]),
                    tuple([x3, y3])
                  ])
    min_x = min(x1,x2,x3)
    max_x = max(x1,x2,x3)
    min_y = min(y1,y2,y3)
    max_y = max(y1,y2,y3)
    # print("minX, minY, maxX, maxY: ", min_x, min_y, max_x, max_y)
    # z_val = np.absolute(np.mean([z1,z2,z3]))
    z_val = np.mean([z1,z2,z3])

    if z_buffer[y1][x1] > z_val:
      vertex_map[y1][x1] = v1
      z_buffer[y1][x1] = z_val
    if z_buffer[y2][x2] > z_val:
      vertex_map[y2][x2] = v2
      z_buffer[y2][x2] = z_val
    if z_buffer[y3][x3] > z_val:
      vertex_map[y3][x3] = v3
      z_buffer[y3][x3] = z_val

    for r in range(min_y, max_y + 1):
      for c in range(min_x, max_x + 1):
        if target_area.contains_points([tuple([c, r])]):
          # if z_val < np.absolute(z_buffer[r][c]):
          if z_val < z_buffer[r][c]:
            z_buffer[r][c] = z_val
            vertex_map[r][c] = np.nan
  return vertex_map



def get_keypoint_from_segment(crop_segment):

  size0, size1, size2 = crop_segment.shape
  for i in range(size0):
    if np.sum(crop_segment[i]) >= 1:
      y1 = i
      break
  
  for i in range(size0 - 1, 0, -1):
    if np.sum(crop_segment[i]) >= 1:
      y2 = i + 1
      break

  for i in range(size1):
    col = crop_segment[:, i]
    if np.sum(col) >= 1:
      x1 = i
      break
  
  for i in range(size1 - 1, 0, -1):
    col = crop_segment[:, i]
    if np.sum(col) >= 1:
      x2 = i + 1
      break
  
  # x1 = int(x1  / 1080 * resolution[0])
  # y1 = int(y1  / 1080 * resolution[1])
  # x2 = int(x2  / 1080 * resolution[0])
  # y2 = int(y2  / 1080 * resolution[1])
  
  mid_x = (x1 + x2) // 2
  mid_y = (y1 + y2) // 2
  diff_x = x2 - x1
  diff_y = y2 - y1

  return mid_x, mid_y, diff_x, diff_y



def get_keypoint_from_projected_xy(projected_xy):
  # projected_xy.shape == [n, 3]
  x_axis_values = projected_xy[:, 0]
  y_axis_values = projected_xy[:, 1]

  projected_diff_x = np.amax(x_axis_values) - np.amin(x_axis_values)
  projected_diff_y = np.amax(y_axis_values) - np.amin(y_axis_values)
  projected_mid_x = (np.amax(x_axis_values) + np.amin(x_axis_values)) / 2
  projected_mid_y = (np.amax(y_axis_values) + np.amin(y_axis_values)) / 2
  
  return projected_mid_x, projected_mid_y, projected_diff_x, projected_diff_y



def get_corresponding_grid(crop_segment, xyz, faces, view_vec, proj_mat, resolution):
  mid_x, mid_y, diff_x, diff_y = get_keypoint_from_segment(crop_segment)

  visible_vertex, visible_faces = \
                    get_visible_vertex(xyz, view_vec, faces)

  projected_xy = get_projected_xy(xyz, visible_vertex, proj_mat)
  
  projected_mid_x, projected_mid_y, \
  projected_diff_x, projected_diff_y = \
                    get_keypoint_from_projected_xy(projected_xy)

  x_scale = projected_diff_x / diff_x * 1.05   # I have this factor to reduce
  y_scale = projected_diff_y / diff_y * 1.05   # the SMPL projection size
  # x_scale = projected_diff_x / diff_x
  # y_scale = projected_diff_y / diff_y

  vertex_map = get_vertex_map(projected_xy, visible_vertex, visible_faces,
                            projected_mid_x, projected_mid_y,
                            mid_x, mid_y, x_scale, y_scale, resolution)
  
  return vertex_map