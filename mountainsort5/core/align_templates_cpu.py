import numpy as np
import numpy.typing as npt

def compute_pairwise_optimal_offset(template1: npt.NDArray[np.float32], template2: npt.NDArray[np.float32]):
    T = template1.shape[0]
    best_inner_product = -np.inf
    best_offset = 0
    for offset in range(T):
        inner_product = np.sum(np.roll(template1, shift=offset, axis=0) * template2)
        if inner_product > best_inner_product:
            best_inner_product = inner_product
            best_offset = offset
    if best_offset > T // 2:
        best_offset = best_offset - T
    return best_offset, best_inner_product

def align_templates(templates: npt.NDArray[np.float32]):
    K = templates.shape[0]
    # T = templates.shape[1]
    # M = templates.shape[2]
    offsets = np.zeros((K,), dtype=np.int32)
    pairwise_optimal_offsets = np.zeros((K, K), dtype=np.int32)
    pairwise_inner_products = np.zeros((K, K), dtype=np.float32)
    for k1 in range(K):
        for k2 in range(K):
            offset, inner_product = compute_pairwise_optimal_offset(templates[k1], templates[k2])
            pairwise_optimal_offsets[k1, k2] = offset
            pairwise_inner_products[k1, k2] = inner_product
    for passnum in range(20):
        something_changed = False
        for k1 in range(K):
            weighted_sum = 0
            total_weight = 0
            for k2 in range(K):
                if k1 != k2:
                    offset = pairwise_optimal_offsets[k1, k2] + offsets[k2]
                    weight = pairwise_inner_products[k1, k2]
                    weighted_sum += weight * offset
                    total_weight += weight
            if total_weight > 0:
                avg_offset = int(weighted_sum / total_weight)
            else:
                avg_offset = 0
            if avg_offset != offsets[k1]:
                something_changed = True
                offsets[k1] = avg_offset
        if not something_changed:
            # print('Template alignment converged.')
            break
    # print('Align templates offsets: ', offsets)
    return offsets