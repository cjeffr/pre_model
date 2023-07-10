import numpy as np
import os
from clawpack.geoclaw import topotools
from clawpack.amrclaw import regiontools


def find_island_points(topo_data):
    """
    Searches through bathymetry height/depth values and 
    Gets lists of indices and lat/lons of where the barrier island exists
    """
    island_x = []
    points = []
    inds = []
    for i, lon in enumerate(topo_data.x):
        first_point = None
        row = np.array(topo_data.Z[:, i])
        not_land = np.where(row <= 0)
        land = np.where(row > 0)

        for point in land[0]:
            if topo_data.Z[point - 1, i] <= 0 and topo_data.Z[point, i] > 0:
                first_point = (point, topo_data.y[point], i, lon)
                break
        if first_point:
            for point in not_land[0]:
                if point > first_point[0] and topo_data.Z[point + 1, i] <= 0:
                    last_point = (point, topo_data.y[point], i, lon)
                    island_x.append((first_point, last_point))
                    break
    for p in island_x:
        inds.append([p[0][2], p[0][0], p[1][0]])
        for i in p:
            points.append([i[3], i[1]])

    return {
        "island_x": island_x,
        "inds": inds,
        "points": points
    }


def create_island_mask(topo_data, points_indices, 
                       save_path, island_name='moriches'):
    """
    search through island indices and mask out the island
    """
    indices = []
    for data in points_indices:
        for i in range(data[1], data[2] + 1):
            indices.append((data[0], i))
            
    mask_array = np.zeros(shape=topo_data.Z.shape)
    
    for locs in indices:
        col, row = locs
        mask_array[row, col] = 1
        
    masked_data = np.ma.masked_array(topo_data.Z, np.logical_not(mask_array))
    np.savez(save_path + f'masked_{island_name}.npz', 
             data=masked_data, mask=masked_data.mask)
    return masked_data


def calc_no_island_values(topo_data, island_idxs):
    """
    Calculates the averages of the water depth before and behind the island
    """
    replace_mask = []
    for idx in island_idxs:
        first = idx[0]
        last = idx[1]
        avg = (topo_data.Z[first[0] - 1,first[2]] + topo_data.Z[last[0] + 1,first[2]])/2
        replace_mask.append([first[0], last[0], first[2], avg])
        
    return replace_mask


def remove_island(topo_data, island_idxs):
    """
    replaces the island height with the averaged 
    depths obtained from calc_no_island_values
    """
    replace_data = calc_no_island_values(topo_data, island_idxs)
    for zdata in replace_data:
        topo_data.Z[zdata[0]:zdata[1], zdata[2]] = zdata[3]
    return topo_data


if __name__ == '__main__':
    topo = topotools.Topography('/home/catherinej/bathymetry/moriches.nc', 4)
    filter_region = [-72.885652,-72.634247,40.718299,40.828344] # Moriches Bay, NY
    topo_data = topo.crop(filter_region)
    island_coords = find_island(topo_data)
    masked_data = create_island_mask(topo_data)