from time import time
import numpy as np
from scipy.signal import convolve2d
from config import FOLDERS, PATH_epigenomics, PATH_OE


def positional_encoding(length=1250, position_dim=8):
    """
    Generate sinusoidal positional encoding for input length and dimension.
    """
    assert position_dim % 2 == 0
    position = np.zeros((length, position_dim))
    for i in range(position_dim // 2):
        div_term = 10000 ** (2 * i / position_dim)
        position[:, 2 * i] = np.sin(np.arange(length) / div_term)
        position[:, 2 * i + 1] = np.cos(np.arange(length) / div_term)
    return position


def distance_mat(length=1250, max_distance=800):
    """
    Generate pairwise distance matrix capped at max_distance.
    """
    dis = np.abs(np.subtract.outer(np.arange(length), np.arange(length)))
    dis[dis > max_distance] = max_distance
    return dis


def load_epigenetic_data(cell_lines, chromosomes, epi_names, res=500, verbose=1):
    """
    Load epigenomic feature data for specified cell lines and chromosomes.
    """
    epigenetic_data = {}
    for cell_line in cell_lines:
        epi_data = {}
        for ch in chromosomes:
            epi_data[ch] = None
            for i, k in enumerate(epi_names):
                path = PATH_epigenomics
                src = f'{path}/Epi/{cell_line}/{ch}/{ch}_{res}bp_{k}.npy'
                s = np.load(src)
                s[s > 6] = 6
                if verbose:
                    print(' Loading epigenomics...', ch, k, len(s))
                if i == 0:
                    epi_data[ch] = np.zeros((len(s), len(epi_names)))
                epi_data[ch][:, i] = s
        epigenetic_data[cell_line] = epi_data
    return epigenetic_data


def load_matrices(
    micro_cell_lines, ch_coord,
    length=2000000, gap=2000000, resolution=500,
    smooth=True, thre=5, avg_min=1e-5
):
    """
        Load normalized contact maps from Micro-C data for each cell line and chromosome region.
        Returns a dictionary of processed contact matrices.
    """
    mat_size = length // resolution
    raw_thr = 1.1

    print('Loading Micro-C contact maps...')
    keys = []
    for ch in ch_coord:
        st, ed = ch_coord[ch]
        ch_keys = [f'{ch}_{pos}' for pos in range(st, ed - length, gap)]
        keys.extend(ch_keys)
    print(
        f' ... {len(micro_cell_lines) * len(keys)} matrices will be stored in the memory'
    )
    memory_size = len(micro_cell_lines) * len(keys) * mat_size**2 * 8 / 1024**3
    print(
        f' ... {memory_size}GB required'
    )

    all_matrices = {}
    for cell in micro_cell_lines:
        norm_path = f'{PATH_OE}/{cell}_{resolution}bp.npy'
        norm_array = np.load(norm_path)
        norm_array[norm_array < avg_min] = avg_min

        print(f'  ... Initializing matrices for {cell}')
        for key in keys:
            all_matrices[f'{cell}_{key}'] = np.zeros((mat_size, mat_size))
        print(f'  ... Loading matrices for {cell}')
        for ch in ch_coord:
            print(f'   ... {ch}')
            st, ed = ch_coord[ch]
            for line in open(f'{FOLDERS[cell]}/{ch}_{resolution}bp.txt'):
                lst = line.strip().split()
                if len(lst) != 8:
                    continue
                p1, p2, v0, v = int(lst[1]), int(lst[4]), float(lst[-2]), float(lst[-1])
                if abs(p1 - p2) >= length:
                    continue
                if v0 < raw_thr:
                    continue
                # Check p1 in which matrices
                p1_bins, p2_bins = set(), set()
                p1_bin, p2_bin = (p1 - st) // gap, (p2 - st) // gap
                while p1_bin >= 0:
                    if st + p1_bin * gap + length <= p1:
                        break
                    elif st + p1_bin * gap <= p1 < st + p1_bin * gap + length < ed:
                        p1_bins.add(p1_bin)
                    p1_bin -= 1
                while p2_bin >= 0:
                    if st + p2_bin * gap + length <= p2:
                        break
                    elif st + p2_bin * gap <= p2 < st + p2_bin * gap + length < ed:
                        p2_bins.add(p2_bin)
                    p2_bin -= 1
                for idx in p1_bins.intersection(p2_bins):
                    mat_st = st + idx * gap
                    pp1, pp2 = (p1 - mat_st) // resolution, (p2 - mat_st) // resolution
                    key = f'{cell}_{ch}_{mat_st}'
                    all_matrices[key][pp1, pp2] += v / norm_array[abs(pp1 - pp2)]
                    if pp1 != pp2:
                        all_matrices[key][pp2, pp1] += v / norm_array[abs(pp1 - pp2)]

    if smooth:
        for key in all_matrices:
            all_matrices[key] = convolve2d(
                np.log(all_matrices[key] + 1), np.ones((5, 5)) / 25, mode='same'
            )
            all_matrices[key][all_matrices[key] > thre] = thre
    else:
        for key in all_matrices:
            all_matrices[key] = np.log(all_matrices[key] + 1)
            all_matrices[key][all_matrices[key] > thre] = thre

    return all_matrices


def training_data_generator(
        ch_coord, micro_cell_lines, epi_names,
        max_range=625000, resolution=500, gap=125000,
        pos_enc_dim=8, n_epoches=100, batch_size=20
):
    """
        Generator for training data samples, includes positional encoding, epigenomic data, and contact matrices.
    """
    if resolution < 1000:
        smooth = True
    else:
        smooth = False

    size = max_range // resolution
    for ch in ch_coord:
        st, ed = ch_coord[ch]
        assert st % 1000 == 0
        assert ed % 1000 == 0

    # positional encoding
    _tm = time()
    pos_encoding = positional_encoding(size, pos_enc_dim)
    pos_encoding = np.array([pos_encoding for _ in range(batch_size)])

    distance = distance_mat(size, size)
    distance = np.array([distance for _ in range(batch_size)])
    # print(pos_encoding.shape)
    (t_min, t_sec) = divmod(time() - _tm, 60)
    _tm = time()
    print('Finish generating positional encoding...', '{}min:{}sec'.format(t_min, t_sec))

    # load weight matrices
    weights = np.ones((size, size))
    x1, x2 = 0, 300
    y1, y2 = 2, 1
    for i in range(size):
        for j in range(size):
            dis = abs(i - j)
            if x1 <= dis < x2:
                weights[i, j] = y1 - (y1 - y2) * (dis - x1) / (x2 - x1)
    weights = np.array([weights for _ in range(batch_size)])

    # load epigenomic data
    epi_features = load_epigenetic_data(micro_cell_lines, ch_coord.keys(), epi_names, res=resolution)
    (t_min, t_sec) = divmod(time() - _tm, 60)
    _tm = time()
    print('Finish generating epigenomic features...', '{}min:{}sec'.format(t_min, t_sec))

    # Load Micro-C contact maps
    micro_mats = load_matrices(
        micro_cell_lines, ch_coord,
        length=max_range, gap=gap, resolution=resolution,
        smooth=smooth
    )
    (t_min, t_sec) = divmod(time() - _tm, 60)
    print(f'Finish processing {len(micro_mats)} MicroC maps...', '{}min:{}sec'.format(t_min, t_sec))

    # Initiating iteration:
    all_keys = list(micro_mats.keys())
    idx = list(range(len(all_keys)))
    # print(idx)
    _tm = time()
    np.random.seed(0)

    print('Start training:')
    for _epoch in range(n_epoches):
        print('Epoch:', _epoch + 1)
        np.random.shuffle(idx)

        for _batch in range(len(idx) // batch_size):
            if _epoch != 0 or _batch != 0:
                (t_min, t_sec) = divmod(time() - _tm, 60)
                print('{}min:{}sec'.format(t_min, t_sec))
            _tm = time()

            print(' Batch:', _batch + 1)
            batch_idx = idx[_batch * batch_size: (_batch + 1) * batch_size]
            # print(batch_idx)
            # print([all_keys[__] for __ in batch_idx])

            micros = np.array(
                [micro_mats[all_keys[_id]] for _id in batch_idx]
            )

            epis = np.zeros((batch_size, size, len(epi_names)))

            for i in range(batch_size):
                _id = all_keys[batch_idx[i]]
                [cell, ch, pos] = _id.split('_')
                pos = int(pos)
                epis[i, :, :] = \
                    epi_features[cell][ch][pos // resolution: pos // resolution + size, :]

            yield _epoch+1, _batch+1, (epis, pos_encoding, distance, weights), micros * weights


