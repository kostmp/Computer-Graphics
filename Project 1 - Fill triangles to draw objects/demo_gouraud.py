import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pltim


#computes a vector V in coordinates p = (x,y) from the linear interpolation of V1,V2
def interpolate_vectors(p1,p2,V1,V2,xy,dim):
    #p1: is the 2d point p1 = (x1,y1)
    #p2: is the 2d point p2 = (x1,y1)
    #V1: is 3d vector which corresponds to p1
    #V2: is 3d vector which corresponds to p2
    # if dim = 1 then xy is the coorδinate x of point p
    # if dim = 2 then xy is the coordinate y of point p
  if dim == 1:
    if p2[0] == p1[0]:
      V = np.dot(0.5,V1) + np.dot(0.5,V2)
      return V
    a = (xy - p1[0])/(p2[0] - p1[0])
    V = np.dot(1-a,V1) + np.dot(a,V2)
    return V
  elif dim == 2:
    if p2[1] == p1[1]:
      V = np.dot(0.5,V1) + np.dot(0.5,V2)
      return V
    a = (xy - p1[1])/(p2[1] - p1[1])
    V = np.dot(1-a,V1) + np.dot(a,V2)
    return V
  else:
    return V1



def Gourauds(canvas, vertices, vcolors):
  vertices = np.array(vertices)
  x0, y0 = vertices[0,0], vertices[0,1]
  x1, y1 = vertices[1,0], vertices[1,1]
  x2, y2 = vertices[2,0], vertices[2,1]
  vcolors = np.array(vcolors)
  V0R =  vcolors[0,0]
  V0G =  vcolors[0,1]
  V0B =  vcolors[0,2]
  V1R, V1G, V1B = vcolors[1,0], vcolors[1,1], vcolors[1,2] 
  V2R, V2G, V2B = vcolors[2,0], vcolors[2,1], vcolors[2,2]
  if (x0 == x1) and (x1 == x2) and (y1 == y0) and (y2 == y0):
    canvas[x0,y0,0] = (vcolors[0,0] + vcolors[1,0] + vcolors[2,0])/3
    canvas[x0,y0,1] = (vcolors[0,1] + vcolors[1,1] + vcolors[2,1])/3
    canvas[x0,y0,2] = (vcolors[0,2] + vcolors[1,2] + vcolors[2,2])/3
    return canvas
# Sort the vertices by y-coordinate
  if y0 > y1:
    x0, y0, x1, y1 = x1, y1, x0, y0
    V0R,V1R = V1R,V0R
    V0G,V1G = V1G,V0G
    V0B,V1B = V1B,V0B
  if y1 > y2:
    x1, y1, x2, y2 = x2, y2, x1, y1
    V1R,V2R = V2R,V1R
    V1G,V2G = V2G,V1G
    V1B,V2B = V2B,V1B
  if y0 > y1:
    x0, y0, x1, y1 = x1, y1, x0, y0
    V0R,V1R = V1R,V0R
    V0G,V1G = V1G,V0G
    V0B,V1B = V1B,V0B

# Calculate the slopes of the edges of the triangle
  if y1 - y0 == 0:
    inv_slope1 = 0
  else:
    inv_slope1 = (x1 - x0) / (y1 - y0)
  if y2 - y0 == 0:
    inv_slope2 = 0
  else:
    inv_slope2 = (x2 - x0) / (y2 - y0)
  if y2 - y1 == 0:
    inv_slope3 = 0
  else:
    inv_slope3 = (x2 - x1) / (y2 - y1)

# Determine the minimum and maximum y-coordinates of the triangle
  min_y = y0
  max_y = y2

# Loop over each scanline from the minimum to maximum y-coordinate
  for y in range(min_y, max_y + 1):
    # Determine the x-coordinates where the scanline intersects the left and right edges of the triangle
    if y < y1:
        x_start = int(x0 + (y - y0) * inv_slope1)
        x_end = int(x0 + (y - y0) * inv_slope2)
        V1 = interpolate_vectors(np.array([x0,y0]),np.array([x1,y1]),np.array([V0R,V0G,V0B]),np.array([V1R,V1G,V1B]),y,2)
        V2 = interpolate_vectors(np.array([x0,y0]),np.array([x2,y2]),np.array([V0R,V0G,V0B]),np.array([V2R,V2G,V2B]),y,2)
    else:
        x_start = int(x1 + (y - y1) * inv_slope3)
        x_end = int(x0 + (y - y0) * inv_slope2)
        V1 = interpolate_vectors(np.array([x1,y1]),np.array([x2,y2]),np.array([V1R,V1G,V1B]),np.array([V2R,V2G,V2B]),y,2)
        V2 = interpolate_vectors(np.array([x0,y0]),np.array([x2,y2]),np.array([V0R,V0G,V0B]),np.array([V2R,V2G,V2B]),y,2)

    if x_start > x_end:
        x_start, x_end = x_end, x_start

    # Draw a horizontal line between those x-coordinates on the current scanline
    for x in range(x_start, x_end + 1):
        V = interpolate_vectors(np.array([x_start,y]),np.array([x_end+1,y]),V1,V2,x,1)
        canvas[x,y,0] = V[0]#(vcolors[0,0] + vcolors[1,0] + vcolors[2,0])/3
        canvas[x,y,1] = V[1]#(vcolors[0,1] + vcolors[1,1] + vcolors[2,1])/3
        canvas[x,y,2] = V[2]#(vcolors[0,2] + vcolors[1,2] + vcolors[2,2])/3
  updatedcanvas = canvas
  return updatedcanvas  


def flats(canvas, vertices, vcolors):
  vertices = np.array(vertices)
  x0, y0 = vertices[0,0], vertices[0,1]
  x1, y1 = vertices[1,0], vertices[1,1]
  x2, y2 = vertices[2,0], vertices[2,1]
  vcolors = np.array(vcolors)
  if (x0 == x1) and (y0 == y1):
    if x2 > x0:
      canvas = draw_line(canvas,vcolors,x0,x2,y0,y2)
    else:
      canvas = draw_line(canvas, vcolors, x2,x0,y2,y0)
    return canvas
  if (x0 == x2) and (y0 == y2):
    if x2 > x1:
      canvas = draw_line(canvas,vcolors,x1,x2,y1,y2)
    else:
      canvas = draw_line(canvas, vcolors, x2,x1,y2,y1)
    return canvas
  if (x2 == x1) and (y2 == y1):
    if x2 > x0:
      canvas = draw_line(canvas,vcolors,x0,x2,y0,y2)
    else:
      canvas = draw_line(canvas, vcolors, x2,x0,y2,y0)
    return canvas
  if (x0 == x1) and (x1 == x2) and (y1 == y0) and (y2 == y0):
    canvas[x0,y0,0] = (vcolors[0,0] + vcolors[1,0] + vcolors[2,0])/3
    canvas[x0,y0,1] = (vcolors[0,1] + vcolors[1,1] + vcolors[2,1])/3
    canvas[x0,y0,2] = (vcolors[0,2] + vcolors[1,2] + vcolors[2,2])/3
    return canvas
# Sort the vertices by y-coordinate
  if y0 > y1:
    x0, y0, x1, y1 = x1, y1, x0, y0
  if y1 > y2:
    x1, y1, x2, y2 = x2, y2, x1, y1
  if y0 > y1:
    x0, y0, x1, y1 = x1, y1, x0, y0

# Calculate the slopes of the edges of the triangle
  if y1 - y0 == 0:
    inv_slope1 = 0
  else:
    inv_slope1 = (x1 - x0) / (y1 - y0)
  if y2 - y0 == 0:
    inv_slope2 = 0
  else:
    inv_slope2 = (x2 - x0) / (y2 - y0)
  if y2 - y1 == 0:
    inv_slope3 = 0
  else:
    inv_slope3 = (x2 - x1) / (y2 - y1)

# Determine the minimum and maximum y-coordinates of the triangle
  min_y = y0
  max_y = y2

# Loop over each scanline from the minimum to maximum y-coordinate
  for y in range(min_y, max_y + 1):
    # Determine the x-coordinates where the scanline intersects the left and right edges of the triangle
    if y < y1:
        x_start = int(x0 + (y - y0) * inv_slope1)
        x_end = int(x0 + (y - y0) * inv_slope2)
    else:
        x_start = int(x1 + (y - y1) * inv_slope3)
        x_end = int(x0 + (y - y0) * inv_slope2)
    if x_start > x_end:
        x_start, x_end = x_end, x_start

    # Draw a horizontal line between those x-coordinates on the current scanline
    for x in range(x_start, x_end + 1):
        canvas[x,y,0] = (vcolors[0,0] + vcolors[1,0] + vcolors[2,0])/3
        canvas[x,y,1] = (vcolors[0,1] + vcolors[1,1] + vcolors[2,1])/3
        canvas[x,y,2] = (vcolors[0,2] + vcolors[1,2] + vcolors[2,2])/3
  updatedcanvas = canvas
  return updatedcanvas 


def render(verts2d, faces, vcolors, depth, shade_t):
  M = 512
  N = 512
  canvas = np.ones((M,N,3))
  colors = []
  depths = []
  for i in faces:
        index = faces.index(i)
        faces[index] = [verts2d[i[0]], verts2d[i[1]], verts2d[i[2]]]
        new_depth = (depth[i[0]] + depth[i[1]] + depth[i[2]]) / 3
        colors.append([vcolors[i[0]], vcolors[i[1]], vcolors[i[2]]])
        depths.append(new_depth)
  n = len(depths)
  swapped = False
  for i in range(n-1):
        for j in range(0, n-i-1):
            if depths[j] < depths[j + 1]:
                swapped = True
                depths[j], depths[j + 1] = depths[j + 1], depths[j]
                colors[j], colors[j + 1] = colors[j + 1], colors[j]
                faces[j], faces[j + 1] = faces[j + 1], faces[j]
  if shade_t == 'flat':
    for triangle in faces:
        canvas = flats(canvas, triangle, colors[faces.index(triangle)])
  elif shade_t == 'gouraud':
    for triangle in faces:
        canvas = Gourauds(canvas, triangle, colors[faces.index(triangle)])
  else:
    print('The shade_t must be flat or gouraud')
  return canvas

def draw_line(canvas,vcolors,x0,x1,y0,y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    err = dx - dy
    
    while True:
        canvas[x0,y0,0] = (vcolors[0,0] + vcolors[1,0] + vcolors[2,0])/3
        canvas[x0,y0,1] = (vcolors[0,1] + vcolors[1,1] + vcolors[2,1])/3
        canvas[x0,y0,2] = (vcolors[0,2] + vcolors[1,2] + vcolors[2,2])/3
        
        if x0 == x1 and y0 == y1:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return canvas  
data = np.load('gdrive/MyDrive/h1.npy', allow_pickle=True)
vcolors = data[()]['vcolors'].tolist()
faces = data[()]['faces'].tolist()
depth = data[()]['depth'].tolist()
verts2d = data[()]['verts2d'].astype(int).tolist()
print('This will take some 23 seconds, because of bubblesort')
img = render(verts2d, faces, vcolors, depth, 'gouraud')
plt.imshow(img, interpolation='nearest')
plt.title('Gourauds Shading')
plt.show()
pltim.imsave('shade_gouraud.png',np.array(img))