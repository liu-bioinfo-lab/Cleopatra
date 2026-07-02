import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl


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


def visualize_HiC_epigenetics(HiC_all, epis_all, output, fig_width=12.0,
                              vmin=0, vmax=None, cmap=fall_cmap, colorbar=True,
                              colorbar_orientation='vertical',
                              epi_labels=None, x_ticks=None, fontsize=24,
                              epi_colors=None, epi_yaxis=True,
                              heatmap_ratio=0.6, epi_ratio=0.1,
                              interval_after_heatmap=0.05, interval_between_epi=0.01, ):
    nMaps = len(HiC_all)
    N = len(HiC_all[0])

    rs = [heatmap_ratio, interval_after_heatmap] + [epi_ratio, interval_between_epi] * len(epis_all[0])
    rs = np.array(rs[:-1])

    fig_height = fig_width * np.sum(rs)
    rs = rs / np.sum(rs)
    fig = plt.figure(figsize=(fig_width * nMaps, fig_height))
    ws = np.ones((nMaps,)) / nMaps

    gs = GridSpec(len(rs), nMaps, height_ratios=rs, width_ratios=ws)

    for cnt in range(len(HiC_all)):
        HiC, epis = HiC_all[cnt], epis_all[cnt]
        ax0 = plt.subplot(gs[0, cnt])
        coordinate = np.array([[[(x + y) / 2, y - x] for y in range(N + 1)] for x in range(N + 1)])
        X, Y = coordinate[:, :, 0], coordinate[:, :, 1]
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
            else:
                raise ValueError('Wrong orientation!')
            cbar = plt.colorbar(im, cax=fig.add_axes([_left, _bottom, _width, _height]),
                                orientation=colorbar_orientation)
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.outline.set_visible(False)

        if epi_labels:
            assert len(epis) == len(epi_labels)
        if epi_colors:
            assert len(epis) == len(epi_colors)

        for i, epi in enumerate(epis):
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

            ax1.set_xlim([-0.5, N - 0.5])
            if epi_labels:
                ax1.set_ylabel(epi_labels[i], fontsize=fontsize, rotation=0)
        ax1.spines['bottom'].set_visible(True)
        if x_ticks:
            tick_pos = np.linspace(0, N - 1, len(x_ticks))
            ax1.set_xticks(tick_pos)
            ax1.set_xticklabels(x_ticks, fontsize=fontsize)
        else:
            ax1.set_xticks([])
            ax1.set_xticklabels([])

    plt.savefig(output)
    plt.close()
