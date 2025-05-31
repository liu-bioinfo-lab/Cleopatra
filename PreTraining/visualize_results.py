import os
from time import time
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from model import Cleopatra
import matplotlib as mpl
from config import FOLDERS, PATH_epigenomics, PATH_OE


fall = np.array(
        (
            (255, 255, 255),
            (255, 255, 204),
            (255, 237, 160),
            (254, 217, 118),
            (254, 178, 76),
            (253, 141, 60),
            (252, 78, 42),
            (227, 26, 28),
            (189, 0, 38),
            (128, 0, 38),
            (0, 0, 0),
        )
    ) / 255


def list_to_colormap(color_list, name=None):
    color_list = np.array(color_list)
    if color_list.min() < 0:
        raise ValueError("Colors should be 0 to 1, or 0 to 255")
    if color_list.max() > 1.0:
        if color_list.max() > 255:
            raise ValueError("Colors should be 0 to 1 or 0 to 255")
        else:
            color_list = color_list / 255.0
    return mpl.colors.LinearSegmentedColormap.from_list(name, color_list, 256)


fall_cmap = list_to_colormap(fall)


def positional_encoding(length=1250, position_dim=8):
    assert position_dim % 2 == 0
    position = np.zeros((length, position_dim))
    for i in range(position_dim // 2):
        position[:, 2 * i] = np.sin([pp / (10000 ** (2 * i / position_dim)) for pp in range(length)])
        position[:, 2 * i + 1] = np.cos([pp / (10000 ** (2 * i / position_dim)) for pp in range(length)])
    return position


def distance_mat(length=1250, max_distance=400):
    m1 = np.tile(np.arange(length), (length, 1))
    m2 = m1.T
    dis = np.abs(m1 - m2)
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
        length=2000000, gap=500000, resolution=500,
        smooth=True, thre=8, avg_min=1e-5,
        raw_thr=0.1
):
    mat_size = length // resolution

    print('Loading Micro-C contact maps...')
    keys = []
    for idx in ch_coord:
        ch, st, ed = ch_coord[idx]
        ch_keys = [f'{ch}_{pos}' for pos in range(st, ed - length + 1, gap)]
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
        for idx in ch_coord:
            ch, st, ed = ch_coord[idx]
            print(f'   ... {ch}')
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
        max_range=1000000, resolution=500, n_distance=600, gap=1000000,
        pos_enc_dim=8, n_epoches=100, batch_size=20
):
    if resolution < 1000:
        smooth = True
    else:
        smooth = False

    size = max_range // resolution
    for idx in ch_coord:
        ch, st, ed = ch_coord[idx]
        assert st % 1000 == 0
        assert ed % 1000 == 0

    # positional encoding
    _tm = time()
    pos_encoding = positional_encoding(size, pos_enc_dim)
    pos_encoding = np.array([pos_encoding for _ in range(batch_size)])

    distance = distance_mat(size, n_distance)
    distance = np.array([distance for _ in range(batch_size)])
    # print(pos_encoding.shape)
    (t_min, t_sec) = divmod(time() - _tm, 60)
    _tm = time()
    print('Finish generating positional encoding...', '{}min:{}sec'.format(t_min, t_sec))

    # load epigenomic data
    chroms = set([ch_coord[idx][0] for idx in ch_coord])
    epi_features = load_epigenetic_data(micro_cell_lines, chroms, epi_names, res=resolution)
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

    weights = np.ones((batch_size, size, size))

    print('Start training:')
    for _epoch in range(n_epoches):
        print('Epoch:', _epoch + 1)
        # np.random.shuffle(idx)

        for _batch in range(len(idx) // batch_size):
            if _epoch != 0 or _batch != 0:
                (t_min, t_sec) = divmod(time() - _tm, 60)
                print('{}min:{}sec'.format(t_min, t_sec))
            _tm = time()

            print(' Batch:', _batch + 1)
            batch_idx = idx[_batch * batch_size: (_batch + 1) * batch_size]
            # print(batch_idx)
            kys = [all_keys[__] for __ in batch_idx]
            print(kys)

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

            yield _epoch+1, _batch+1, (epis, pos_encoding, distance, weights), micros, kys


def visualize_HiC_epigenetics(HiC_all, epis_all, output, fig_width=12.0,
                              vmin=0, vmax=None, cmap=fall_cmap, colorbar=True,
                              colorbar_orientation='vertical',
                              epi_labels=None, x_ticks=None, fontsize=24,
                              epi_colors=None, epi_yaxis=True,
                              heatmap_ratio=0.6, epi_ratio=0.1,
                              interval_after_heatmap=0.05, interval_between_epi=0.01, ):
    """
    Visualize matched HiC and epigenetic signals in one figure
    Args:
        HiC (numpy.array): Hi-C contact map, only upper triangle is used.
        epis (list): epigenetic signals
        output (str): the output path. Must in a proper format (e.g., 'png', 'pdf', 'svg', ...).
        fig_width (float): the width of the figure. Then the height will be automatically calculated. Default: 12.0
        vmin (float): min value of the colormap. Default: 0
        vmax (float): max value of the colormap. Will use the max value in Hi-C data if not specified.
        cmap (str or plt.cm): which colormap to use. Default: 'Reds'
        colorbar (bool): whether to add colorbar for the heatmap. Default: True
        colorbar_orientation (str): "horizontal" or "vertical". Default: "vertical"
        epi_labels (list): the names of epigenetic marks. If None, there will be no labels at y axis.
        x_ticks (list): a list of strings. Will be added at the bottom. THE FIRST TICK WILL BE AT THE START OF THE SIGNAL, THE LAST TICK WILL BE AT THE END.
        fontsize (int): font size. Default: 24
        epi_colors (list): colors of epigenetic signals
        epi_yaxis (bool): whether add y-axis to epigenetic signals. Default: True
        heatmap_ratio (float): the ratio of (heatmap height) and (figure width). Default: 0.6
        epi_ratio (float): the ratio of (1D epi signal height) and (figure width). Default: 0.1
        interval_after_heatmap (float): the ratio of (interval between heatmap and 1D signals) and (figure width). Default: 0.05
        interval_between_epi (float): the ratio of (interval between 1D signals) and (figure width). Default: 0.01

    No return. Save a figure only.
    """

    # Make sure the lengths match
    # len_epis = [len(epi) for epi in epis]
    # if max(len_epis) != min(len_epis) or max(len_epis) != len(HiC):
    #     raise ValueError('Size not matched!')
    nMaps = len(HiC_all)
    N = len(HiC_all[0])

    # Define the space for each row (heatmap - interval - signal - interval - signal ...)
    rs = [heatmap_ratio, interval_after_heatmap] + [epi_ratio, interval_between_epi] * len(epis_all[0])
    rs = np.array(rs[:-1])

    # Calculate figure height
    fig_height = fig_width * np.sum(rs)
    rs = rs / np.sum(rs)  # normalize to 1 (ratios)
    fig = plt.figure(figsize=(fig_width * nMaps, fig_height))
    ws = np.ones((nMaps,)) / nMaps

    # Split the figure into rows with different heights
    gs = GridSpec(len(rs), nMaps, height_ratios=rs, width_ratios=ws)

    for cnt in range(len(HiC_all)):
        HiC, epis = HiC_all[cnt], epis_all[cnt]
        # Ready for plotting heatmap
        ax0 = plt.subplot(gs[0, cnt])
        # Define the rotated axes and coordinates
        coordinate = np.array([[[(x + y) / 2, y - x] for y in range(N + 1)] for x in range(N + 1)])
        X, Y = coordinate[:, :, 0], coordinate[:, :, 1]
        # Plot the heatmap
        _max = vmax if vmax is not None else np.max(HiC)

        im = ax0.pcolormesh(X, Y, HiC, vmin=vmin, vmax=_max, cmap=cmap)
        ax0.axis('off')
        ax0.set_ylim([0, N])
        ax0.set_xlim([0, N])
        if colorbar:
            if colorbar_orientation == 'horizontal':
                _left, _width, _bottom, _height = 0.12 + (cnt + 0.12) / nMaps * 0.82, 0.25 / nMaps, 1 - rs[0] * 0.25, rs[0] * 0.03
            elif colorbar_orientation == 'vertical':
                _left, _width, _bottom, _height = 0.12 + (cnt + 0.8) / nMaps * 0.82, 0.02 / nMaps, 1 - rs[0] * 0.7, rs[0] * 0.5
                print(_left)
            else:
                raise ValueError('Wrong orientation!')
            cbar = plt.colorbar(im, cax=fig.add_axes([_left, _bottom, _width, _height]),
                                orientation=colorbar_orientation)
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.outline.set_visible(False)

        # print(rs/np.sum(rs))
        # Ready for plotting 1D signals
        if epi_labels:
            assert len(epis) == len(epi_labels)
        if epi_colors:
            assert len(epis) == len(epi_colors)

        for i, epi in enumerate(epis):
            # print(epi.shape)
            ax1 = plt.subplot(gs[2 + 2 * i, cnt])

            if epi_colors:
                ax1.fill_between(np.arange(N), 0, epi, color=epi_colors[i])
            else:
                ax1.fill_between(np.arange(N), 0, epi)
            ax1.spines['left'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['top'].set_visible(False)
            ax1.spines['bottom'].set_visible(False)

            if not epi_yaxis:
                ax1.set_yticks([])
                ax1.set_yticklabels([])
            else:
                ax1.spines['right'].set_visible(True)
                ax1.tick_params(labelsize=fontsize)
                ax1.yaxis.tick_right()

            if i != len(epis) - 1:
                ax1.set_xticks([])
                ax1.set_xticklabels([])
            # ax1.axis('off')
            # ax1.xaxis.set_visible(True)
            # plt.setp(ax1.spines.values(), visible=False)
            # ax1.yaxis.set_visible(True)

            ax1.set_xlim([-0.5, N - 0.5])
            if epi_labels:
                ax1.set_ylabel(epi_labels[i], fontsize=fontsize, rotation=0)
        ax1.spines['bottom'].set_visible(True)
        if x_ticks:
            tick_pos = np.linspace(0, N - 1, len(x_ticks))  # 这个坐标其实是不对的 差1个bin 但是为了ticks好看只有先这样了
            ax1.set_xticks(tick_pos)
            ax1.set_xticklabels(x_ticks, fontsize=fontsize)
        else:
            ax1.set_xticks([])
            ax1.set_xticklabels([])

    plt.savefig(output)
    plt.close()


if __name__ == '__main__':
    my_path = os.path.abspath(os.path.dirname(__file__))
    epi_names = ['DNase-seq', 'H3K4me1', 'H3K4me2', 'H3K4me3',
                 'H3K9ac', 'H3K9me3', 'H3K27ac', 'H3K27me3', 'H3K36me3',
                 'H3K79me2', 'H4K20me1', 'CTCF',
                 'EZH2', 'POLR2A', 'JUND', 'REST',
                 'RAD21', 'H2AFZ', 'phastCons']
    max_range = 2500000
    resolution = 2000
    n_distance = 1250
    gap = 500000

    model = Cleopatra(nMarks=len(epi_names), n_distance=n_distance, nBins=max_range // resolution)
    model.load_weights(f'{my_path}/temp_model_60.h5')

    if not os.path.exists(f'{my_path}/examples2/'):
        os.mkdir(f'{my_path}/examples2/')

    ch_coord = {
        0: ('chr6', 25150000, 28600000),
        1: ('chr19', 36050000, 39750000),
        2: ('chr5', 157000000, 160150000),
        3: ('chr1', 207650000, 210350000),
        4: ('chr4', 61350000, 64450000),
        5: ('chr8', 125000000, 129750000),
        6: ('chr6', 29750000, 32250000),
        7: ('chrX', 47125000, 49375000),
        8: ('chr1', 237300000, 240500000),
        9: ('chr7', 10500000, 13500000),
        10: ('chr8', 63000000, 66500000),
        11: ('chr4', 181300000, 184000000),
        12: ('chr3', 118750000, 121500000),
        13: ('chr9', 106620000, 109870000)
    }
    cell_types = ['H1']
    generator = training_data_generator(
        ch_coord, cell_types, epi_names,
        max_range=max_range, resolution=resolution, n_distance=n_distance, gap=gap,
        pos_enc_dim=8, n_epoches=1, batch_size=1
    )

    for epoch, batch, (epis, pos_enc, distance, weights), micros, kys in generator:
        _res = model.predict([epis, pos_enc, distance, weights])[0, :, :]
        _res = (_res + _res.T) / 2
        _epi = epis[0, :, :].T
        _micro = micros[0, :, :]
        ky = kys[0]
        print(ky)
        region_coord = '_'.join(ky.split('_')[1:])

        np.save(f'{my_path}/examples2/{ky}_pred.npy', _res)
        np.save(f'{my_path}/examples2/{ky}_micro.npy', _micro)
        # np.save(f'{my_path}/examples2/{ky}_epi.npy', _epi)

        visualize_HiC_epigenetics([_micro, _res], [_epi, _epi],
                                  f'{my_path}/examples2/vis_{ky}_micro.png', colorbar=True,
                                  interval_after_heatmap=0.,
                                  interval_between_epi=0., epi_labels=epi_names,
                                  epi_yaxis=True, fontsize=20, epi_ratio=0.045,
                                  x_ticks=['', '', '', '', '', ''], vmax=np.quantile(_micro, 0.99)
                                  )


