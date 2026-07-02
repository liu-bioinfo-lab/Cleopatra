import numpy as np
from scipy.stats import pearsonr, spearmanr


def mse_per_patch(true_mat, pred_mat):
    """Mean squared error between two contact map patches (same shape)."""
    return np.mean((true_mat.astype(np.float64) - pred_mat.astype(np.float64)) ** 2)


def psnr_per_patch(true_mat, pred_mat, data_range=None):
    mse = mse_per_patch(true_mat, pred_mat)
    if mse <= 0:
        return np.inf
    if data_range is None:
        data_range = float(np.nanmax(true_mat) - np.nanmin(true_mat))
    if data_range <= 0:
        data_range = 1.0
    return 10.0 * np.log10((data_range ** 2) / mse)


def ssim_per_patch(true_mat, pred_mat, win_size=7, data_range=None):
    try:
        from skimage.metrics import structural_similarity as ssim_fn
    except ImportError:
        raise ImportError("SSIM requires scikit-image: pip install scikit-image")
    if data_range is None:
        data_range = float(np.nanmax(true_mat) - np.nanmin(true_mat))
    if data_range <= 0:
        data_range = 1.0
    true_f = true_mat.astype(np.float64)
    pred_f = pred_mat.astype(np.float64)
    return ssim_fn(true_f, pred_f, data_range=data_range, win_size=win_size)


def correlation_per_patch(true_mat, pred_mat):
    v1, v2 = true_mat.flatten(), pred_mat.flatten()
    spearmanR = spearmanr(v1, v2)[0]
    pearsonR = pearsonr(v1, v2)[0]
    return spearmanR, pearsonR


def compare_loop_intensities(true_mat, pred_mat, loop_locs, radius=2):
    n_bins = true_mat.shape[0]
    crop_pad = n_bins // 10                  # ignore anchors in the outer 10% of the window
    lo, hi = crop_pad, n_bins - crop_pad
    loops_true, loops_pred = [], []
    loop_anchors = []
    for (x1, y1) in loop_locs:
        if lo <= x1 < hi and lo <= y1 < hi:
            v1 = np.mean(true_mat[x1 - radius:x1 + radius + 1, y1 - radius:y1 + radius + 1])
            v2 = np.mean(pred_mat[x1 - radius:x1 + radius + 1, y1 - radius:y1 + radius + 1])
            loops_true.append(v1)
            loops_pred.append(v2)
            loop_anchors.append(x1)
            loop_anchors.append(y1)

    loop_anchors.sort()
    clean_loop_anchors = []
    temp = []
    for loc in loop_anchors:
        if len(temp) == 0:
            temp.append(loc)
        elif loc - temp[-1] <= radius:
            temp.append(loc)
        else:
            previous_anchor = int(np.round(np.mean(temp)))
            clean_loop_anchors.append(previous_anchor)
            temp = [loc]
    if len(temp):
        previous_anchor = int(np.round(np.mean(temp)))
        clean_loop_anchors.append(previous_anchor)

    pseudoloops_true, pseudoloops_pred = [], []
    for i in range(len(clean_loop_anchors) - 1):
        for j in range(i + 1, len(clean_loop_anchors)):
            x1, y1 = clean_loop_anchors[i], clean_loop_anchors[j]
            v1 = np.mean(true_mat[x1 - radius:x1 + radius + 1, y1 - radius:y1 + radius + 1])
            v2 = np.mean(pred_mat[x1 - radius:x1 + radius + 1, y1 - radius:y1 + radius + 1])
            pseudoloops_true.append(v1)
            pseudoloops_pred.append(v2)

    return loops_true, loops_pred, pseudoloops_true, pseudoloops_pred


def insulation_kernel(size=41):
    assert size % 2 == 1
    mat = np.ones((size, size))
    half = size // 2
    mat[:half, half + 1:] = -1
    mat[half + 1:, :half] = -1
    return mat


def multiscale_insulation(true_mat, pred_mat, window_sizes=[20, 100, 200]):
    ins_true, ins_pred = np.zeros((len(window_sizes), len(true_mat))), np.zeros((len(window_sizes), len(pred_mat)))
    ins_count_times = np.zeros((len(window_sizes), len(true_mat)))
    for size_i, size in enumerate(window_sizes):
        real_size = size * 2 + 1
        kernel = insulation_kernel(real_size)
        for i in range(len(true_mat) - real_size + 1):
            submat = true_mat[i:i + real_size, i:i + real_size]
            _sum = np.sum(kernel * submat)
            ins_true[size_i, i + size] = _sum
            submat = pred_mat[i:i + real_size, i:i + real_size]
            _sum = np.sum(kernel * submat)
            ins_pred[size_i, i + size] = _sum
            ins_count_times[size_i, i + size] = 1
    return ins_true, ins_pred, ins_count_times


def normalize_matrix(max_range, gap, resolution):
    size = max_range // resolution
    gap_size = gap // resolution
    assert size % gap_size == 0
    multip = size // gap_size

    def _update_initial(_size, _gap_size, st, ed):
        _mat = np.zeros((size, size))
        for i in range(st, ed):
            if i < 0:
                _mat[:i * _gap_size, :i * _gap_size] += 1
            else:
                _mat[i * _gap_size:, i * _gap_size:] += 1
        return _mat

    norm_mats = {}
    for j in range(multip):
        norm_mats[j] = _update_initial(size, gap_size, -j, multip)
    for j in range(multip - 1):
        norm_mats[-j - 1] = _update_initial(size, gap_size, -multip + 1, j + 1)

    return norm_mats


def update_strata(strata, mat, st_index):
    for i in range(len(strata)):
        diag = np.diag(mat, i)
        strata[i][st_index: st_index + len(diag)] += diag
    return strata


def calculate_scc(strata_all):
    scc = []
    for i in range(1000):
        v1, v2 = strata_all['true'][i], strata_all['pred'][i]
        corr = pearsonr(v1, v2)[0]
        scc.append(corr)
    return scc


def calculate_loop_correlation(loop_intensities):
    v1, v2 = loop_intensities['true'], loop_intensities['pred']
    spearmanR = spearmanr(v1, v2)[0]
    pearsonR = pearsonr(v1, v2)[0]
    return spearmanR, pearsonR


def calculate_insulation_correlation(insulation_scores):
    table = []
    for region_id in insulation_scores:
        mat1, mat2 = insulation_scores[region_id]['true'], insulation_scores[region_id]['pred']
        vec = [region_id]
        for i in range(len(mat1)):
            v1, v2 = mat1[i, :], mat2[i, :]
            corr2 = pearsonr(v1, v2)[0]
            vec.append(corr2)
        table.append(vec)
    return table
