import cv2
import numpy as np


def load_images(left_name, right_name, bsize):
    left = cv2.imread(left_name, 0)
    left = cv2.GaussianBlur(left, bsize, 0, 0)
    right = cv2.imread(right_name, 0)
    right = cv2.GaussianBlur(right, bsize, 0, 0)
    return left, right


def get_indices(offset, dim, direction, height):
    y_indices = []
    x_indices = []

    for i in range(0, dim):
        if direction == 'SE':
            if offset < 0:
                y_indices.append(-offset + i)
                x_indices.append(0 + i)
            else:
                y_indices.append(0 + i)
                x_indices.append(offset + i)

        if direction == 'SW':
            if offset < 0:
                y_indices.append(height + offset - i)
                x_indices.append(0 + i)
            else:
                y_indices.append(height - i)
                x_indices.append(offset + i)

    return np.array(y_indices), np.array(x_indices)


def get_path_cost(slice, P1, P2):
    other_dim = slice.shape[0]
    disparity_dim = slice.shape[1]

    disparities = [d for d in range(disparity_dim)] * disparity_dim
    disparities = np.array(disparities).reshape(disparity_dim, disparity_dim)
    
    # 假设一个方向路径上的视差是比较连续的，上个位置的视差和这个位置的差距过大要加上惩罚值
    penalties = np.zeros(shape=(disparity_dim, disparity_dim), dtype=slice.dtype)
    penalties[np.abs(disparities - disparities.T) == 1] = P1
    penalties[np.abs(disparities - disparities.T) > 1] = P2

    minimum_cost_path = np.zeros(shape=(other_dim, disparity_dim), dtype=slice.dtype)
    minimum_cost_path[0, :] = slice[0, :]

    for i in range(1, other_dim):
        previous_cost = minimum_cost_path[i - 1, :]
        current_cost = slice[i, :]
        costs = np.repeat(previous_cost, repeats=disparity_dim, axis=0).reshape(disparity_dim, disparity_dim)
        # 对于单个像素的每个视差值，求出上个位置是哪个视差值时的代价最小
        costs = np.amin(costs + penalties, axis=0) # 求出每个视差值的代价
        minimum_cost_path[i, :] = current_cost + costs - np.amin(previous_cost)
    return minimum_cost_path


def aggregate_costs(cost_volume, P1, P2):
    height = cost_volume.shape[0]
    width = cost_volume.shape[1]
    disparities = cost_volume.shape[2]
    start = -(height - 1)
    end = width - 1

    #计算8个方向上的代价
    paths = ["S", "E", "SE", "SW"]
    aggregation_cost = np.zeros(shape=(height, width, disparities, len(paths) * 2), dtype=cost_volume.dtype)
    path_id = 0
    
    for path in paths:

        aggregation = np.zeros(shape=(height, width, disparities), dtype=cost_volume.dtype)
        opposite_aggregation = np.copy(aggregation)

        if path == 'S':
            for x in range(0, width):
                south = cost_volume[0:height, x, :]
                north = np.flip(south, axis=0)
                # 计算该方向上的路径代价
                aggregation[:, x, :] = get_path_cost(south, P1, P2)
                opposite_aggregation[:, x, :] = np.flip(get_path_cost(north, P1, P2), axis=0)

        if path == 'E':
            for y in range(0, height):
                east = cost_volume[y, 0:width, :]
                west = np.flip(east, axis=0)
                aggregation[y, :, :] = get_path_cost(east, P1, P2)
                opposite_aggregation[y, :, :] = np.flip(get_path_cost(west, P1, P2), axis=0)

        if path == 'SE':
            for offset in range(start, end):
                south_east = cost_volume.diagonal(offset=offset).T
                north_west = np.flip(south_east, axis=0)
                dim = south_east.shape[0]
                # 求出对角线方向的index
                y_se_idx, x_se_idx = get_indices(offset, dim, path, None)
                y_nw_idx = np.flip(y_se_idx, axis=0)
                x_nw_idx = np.flip(x_se_idx, axis=0)
                aggregation[y_se_idx, x_se_idx, :] = get_path_cost(south_east, P1, P2)
                opposite_aggregation[y_nw_idx, x_nw_idx, :] = get_path_cost(north_west, P1, P2)

        if path == 'SW':
            for offset in range(start, end):
                south_west = np.flipud(cost_volume).diagonal(offset=offset).T
                north_east = np.flip(south_west, axis=0)
                dim = south_west.shape[0]
                y_sw_idx, x_sw_idx = get_indices(offset, dim, path, height - 1)
                y_ne_idx = np.flip(y_sw_idx, axis=0)
                x_ne_idx = np.flip(x_sw_idx, axis=0)
                aggregation[y_sw_idx, x_sw_idx, :] = get_path_cost(south_west, P1, P2)
                opposite_aggregation[y_ne_idx, x_ne_idx, :] = get_path_cost(north_east, P1, P2)

        aggregation_cost[:, :, :, path_id] = aggregation
        aggregation_cost[:, :, :, path_id + 1] = opposite_aggregation
        path_id = path_id + 2

    
    return np.sum(aggregation_cost, axis=3)

def compute_census(img, cheight, cwidth):
    height = img.shape[0]
    width = img.shape[1]
    y_offset = int(cheight / 2)
    x_offset = int(cwidth / 2)
    census_values = np.zeros((height, width), dtype=np.uint64)

    for y in range(y_offset, height - y_offset):
        for x in range(x_offset, width - x_offset):
            census = np.int64(0)
            center_pixel = img[y, x]
            for j in range(-y_offset, y_offset + 1):
                for i in range(-x_offset, x_offset + 1):
                    if j == 0 and i == 0:
                        continue
                    census = census << 1
                    if img[y + j][x + i] < center_pixel:
                        census = census | 1                        

            census_values[y, x] = census

    return census_values

def compute_hamming(x, y):
    xor = np.int64(np.bitwise_xor(x, y))
    distance = np.zeros_like(xor, dtype=np.uint32)
    while not np.all(xor == 0):
        tmp = xor - 1
        mask = xor != 0
        xor[mask] = np.bitwise_and(xor[mask], tmp[mask])
        distance[mask] = distance[mask] + 1
    
    return distance

def compute_costs(left, right, max_disparity, csize):
    height = left.shape[0]
    width = left.shape[1]
    cheight = csize[0]
    cwidth = csize[1]
    x_offset = int(cwidth / 2)
    disparity = max_disparity

    # 编码方阵中比中心小的位置
    left_census_values = compute_census(left, cheight, cwidth)
    right_census_values = compute_census(right, cheight, cwidth)
    left_cost_volume = np.zeros(shape=(height, width, disparity), dtype=np.uint32)
    right_cost_volume = np.zeros(shape=(height, width, disparity), dtype=np.uint32)
    lcensus = np.zeros(shape=(height, width), dtype=np.int64)
    rcensus = np.zeros(shape=(height, width), dtype=np.int64)
    for d in range(0, disparity):
        #计算海明距离
        rcensus[:, (x_offset + d):(width - x_offset)] = right_census_values[:, x_offset:(width - d - x_offset)]
        left_cost_volume[:, :, d] = compute_hamming(np.int64(left_census_values), rcensus)

        lcensus[:, x_offset:(width - d - x_offset)] = left_census_values[:, (x_offset + d):(width - x_offset)]
        right_cost_volume[:, :, d] = compute_hamming(np.int64(right_census_values), lcensus)

    return left_cost_volume, right_cost_volume


def normalize(src):
    dst = cv2.normalize(src, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    dst = dst.astype(np.uint8)
    return dst


def compute_disparity(aggregation_volume, max_disparity, uniqueness_ratio):

    width = aggregation_volume.shape[1]
    height = aggregation_volume.shape[0]

    disparity = np.zeros((height, width), dtype=np.float64)
    for i in range(height):
        for j in range(width):
            
            #求出最小值
            best_disparity = np.argmin(aggregation_volume[i][j])
            min_cost = aggregation_volume[i][j][best_disparity]
            #丢弃过大或者过小的视差
            if best_disparity == 0 or best_disparity == max_disparity - 1:
                disparity[i][j] = np.nan
                continue
            
            #找出次小
            sec_min_cost = min(np.min(aggregation_volume[i][j][:best_disparity]), 
                               np.min(aggregation_volume[i][j][best_disparity+1:]))

            #独特性约束
            if sec_min_cost - min_cost <= int(min_cost * (1 - uniqueness_ratio)):
                disparity[i][j] = np.nan
                continue
            
            #假定为过三个点的二次函数，求出最小值
            idx_1 = best_disparity - 1
            idx_2 = best_disparity + 1
            cost_1 = int(aggregation_volume[i][j][idx_1])
            cost_2 = int(aggregation_volume[i][j][idx_2])
            denom = max(1, cost_1 + cost_2 - 2 * min_cost)
            disparity[i][j] = float(best_disparity) + (cost_1 - cost_2) / (denom * 2.0)

    return disparity


def LR_check(disparity, right, threshold):
    width = disparity.shape[1]
    height = disparity.shape[0]

    # 遮挡区像素和误匹配区像素
    occlusions = []
    mismatches = []

    for i in range(height):
        for j in range(width):
            disp = disparity[i][j]
            if np.isnan(disp):
                mismatches.append((i, j))
                continue
            # 根据视差值找到右影像上对应的像素
            col_right = int(j - disp + 0.5) # 四舍五入
            if 0 <= col_right and col_right < width:
                # 对应像素的视差值
                disp_r = right[i][col_right]
                if abs(disp - disp_r) > threshold:
                    # 区分遮挡区和误匹配区
                    col_rl = int(col_right + disp_r + 0.5)
                    if 0 <= col_rl and col_rl < width:
                        disp_l = disparity[i][col_rl]
                        if disp_l > disp:
                            occlusions.append((i, j))
                        else:
                            mismatches.append((i, j))
                    else:
                        mismatches.append((i, j))
                    disparity[i][j] = np.nan
            else:
                disparity[i][j] = np.nan
                mismatches.append((i, j))
    return disparity, occlusions, mismatches


def fill_missing(disparity, occlusions, mismatches):
    width = disparity.shape[1]
    height = disparity.shape[0]
    disp_collects = []
    angle = np.pi / 4

    for k in range(3):
        if k == 0:
            trg_pixels = occlusions
        elif k == 1:
            trg_pixels = mismatches
        else:
            trg_pixels = []
            for i in range(height):
                for j in range(width):
                    if np.isnan(disparity[i][j]):
                        trg_pixels.append((i, j))
        
        if trg_pixels == []:
            continue

        # 遍历待处理像素
        for pix in trg_pixels:
            y, x = pix
            disp_collects = []
            for k in range(8):
                sina = np.sin(k * angle)
                cosa = np.cos(k * angle)
                # 延每个方向寻找最近的视差
                for m in range(1, max(width, height)):
                    yy = round(y + m * sina)
                    xx = round(x + m * cosa)
                    # 超过边界
                    if yy < 0 or yy >= height or xx < 0 or xx >= width:
                        break
                    disp = disparity[yy][xx]
                    if not np.isnan(disp):
                        disp_collects.append(disp)
                        break

            if disp_collects == []:
                continue
            disp_collects.sort()
            # 如果是遮挡区，则选择第二小的视差值
            # 如果是误匹配区，则选择中值
            if k == 0:
                if len(disp_collects) > 1:
                    disparity[y][x] = disp_collects[1]
                else:
                    disparity[y][x] = disp_collects[0]
            else:
                disparity[y][x] = disp_collects[len(disp_collects) // 2]
        
    return disparity
    

def sgm(image_name):

    left_name = f'images/{image_name}l.jpg'
    right_name = f'images/{image_name}r.jpg'
    output_name = f'results/{image_name}_disparity_map.jpg'
    max_disparity = 64
    P1 = 10
    P2 = 120
    uniqueness_ratio = 0.99
    threshold = 1.0
    csize = (7, 7)
    bsize = (3, 3)

    left, right = load_images(left_name, right_name, bsize)
    left_cost_volume, right_cost_volume = compute_costs(left, right, max_disparity, csize)
    left_aggregation_volume = aggregate_costs(left_cost_volume, P1, P2)
    right_aggregation_volume = aggregate_costs(right_cost_volume, P1, P2)

    left_disparity_map = compute_disparity(left_aggregation_volume, max_disparity, uniqueness_ratio)
    right_disparity_map = compute_disparity(right_aggregation_volume, max_disparity, uniqueness_ratio)

    disparity_map, occlusions, mismatches = LR_check(left_disparity_map, right_disparity_map, threshold)
    disparity_map = fill_missing(disparity_map, occlusions, mismatches)
    disparity_map = normalize(disparity_map)
    disparity_map = cv2.medianBlur(disparity_map, bsize[0])


    cv2.imwrite(f'{output_name}', disparity_map)


images = ['corridor', 'triclopsi2']

for image in images:
    sgm(image)