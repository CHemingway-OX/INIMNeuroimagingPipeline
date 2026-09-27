"""Experimental multimodal deformation-based SEL candidates and calibrated scores.

An auditable adaptation, not the original Elliott implementation or calibration.
"""
import heapq
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy import ndimage
if __package__:
    from . import lesion_longitudinal_common as common
else:
    import lesion_longitudinal_common as common


SCORE_MODEL = 'inward_mm_shell_slope_and_normalized_origin_fit_mse_v1'


def annual_expansion(jacobian, years):
    if years <= 0 or not np.isfinite(jacobian).all() or np.any(jacobian <= 0):
        raise ValueError('Jacobian determinants must be positive/finite and interval positive')
    return 100 * (jacobian - 1) / years


def candidates(mask, rate, high, low, min_voxels):
    """18-connected seeds; priority growth preserves distinct seed identities."""
    structure = ndimage.generate_binary_structure(3, 2)
    seeds, n = ndimage.label(mask & (rate >= high), structure)
    labels = seeds.copy()
    support = mask & (rate >= low)
    offsets = np.argwhere(structure) - 1
    offsets = [tuple(v) for v in offsets if np.any(v)]
    queue = []
    boundary = (seeds > 0) & ~ndimage.binary_erosion(seeds > 0, structure=structure)
    for xyz in np.argwhere(boundary):
        x = tuple(xyz)
        heapq.heappush(queue, (-float(rate[x]), int(labels[x]), x))
    shape = mask.shape
    while queue:
        cost, ident, x = heapq.heappop(queue)
        for delta in offsets:
            y = tuple(a + b for a, b in zip(x, delta))
            if any(v < 0 or v >= s for v, s in zip(y, shape)) or not support[y] or labels[y]:
                continue
            labels[y] = ident
            heapq.heappush(queue, (max(cost, -float(rate[y])), ident, y))
    counts = np.bincount(labels.ravel(), minlength=n + 1)
    labels[np.isin(labels, np.flatnonzero(counts < min_voxels))] = 0
    return labels


def temporal_score(times, changes):
    times, changes = np.asarray(times, float), np.asarray(changes, float)
    if len(times) < 2:
        return {'slope_percent_per_year': None, 'constancy_nmse': None, 'temporal_status': 'not_assessable_two_visits'}
    slope = float(np.dot(times, changes) / np.dot(times, times))
    fit = slope * times
    if slope <= 0:
        return {'slope_percent_per_year': slope, 'constancy_nmse': None, 'temporal_status': 'nonpositive_slope'}
    return {'slope_percent_per_year': slope, 'constancy_nmse': float(np.mean(((changes - fit) / fit) ** 2)),
            'temporal_status': 'assessable'}


def radial_score(mask, rate, spacing, shell_mm):
    # Padding ensures a true exterior even for a candidate touching an image edge.
    distance = ndimage.distance_transform_edt(np.pad(mask, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
    bands = np.floor(distance[mask] / shell_mm).astype(int)
    xs, ys, counts = [], [], []
    for band in np.unique(bands):
        use = bands == band
        xs.append(float(distance[mask][use].mean()))
        ys.append(float(rate[mask][use].mean()))
        counts.append(int(use.sum()))
    slope = float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 3 else None
    return slope, {'distance_inward_mm': xs, 'mean_rate_percent_per_year': ys, 'voxel_count': counts}


def load_calibration(path):
    if not path:
        return None
    c = json.loads(path.read_text())
    if c.get('score_model') != SCORE_MODEL or not c.get('reference'):
        raise ValueError('Calibration requires matching score_model and reference/cohort description')
    for key in ['concentricity_mean', 'concentricity_sd', 'constancy_mean', 'constancy_sd', 'threshold']:
        if not np.isfinite(c.get(key, np.nan)):
            raise ValueError(f'Calibration missing finite {key}')
    if min(c['concentricity_sd'], c['constancy_sd']) <= 0:
        raise ValueError('Calibration standard deviations must be positive')
    return c


def classify(temporal, radial, calibration, enhancing, enhancement_known):
    if enhancing:
        return 'excluded_enhancing', None
    if temporal['constancy_nmse'] is None or radial is None:
        return 'candidate_not_fully_assessable', None
    if calibration is None:
        return 'candidate_uncalibrated', None
    c = calibration
    score = (radial - c['concentricity_mean']) / c['concentricity_sd'] - (temporal['constancy_nmse'] - c['constancy_mean']) / c['constancy_sd']
    if not enhancement_known:
        return 'candidate_enhancement_unknown', float(score)
    return ('adapted_score_pass' if score >= c['threshold'] else 'adapted_score_fail'), float(score)


def deformation_pair(a, b, args, d):
    ants = common.configure_ants(args)
    t0, t1 = ants.image_read(a['t1']), ants.image_read(b['t1'])
    if args.alignment == 'already-aligned':
        matrix = np.eye(4)
    else:
        affine = ants.registration(fixed=t0, moving=t1, type_of_transform='Affine',
            mask=ants.image_read(a['brain']), moving_mask=ants.image_read(b['brain']),
            mask_all_stages=True, outprefix=str(d / 'affine_'), random_seed=args.seed)
        matrix = common.homogeneous(ants.read_transform(affine['fwdtransforms'][0]))
    h = common.half_matrix(matrix)
    ht, hit = common.transform(ants, h), common.transform(ants, np.linalg.inv(h))
    ants.write_transform(ht, str(d / 'midpoint_to_followup.mat'))
    ants.write_transform(hit, str(d / 'midpoint_to_baseline.mat'))
    grid = common.midpoint_grid(ants, t0, t1, h)
    def half(source, tx):
        return {k: tx.apply_to_image(ants.image_read(source[k]), reference=grid,
                    interpolation='nearestneighbor' if k in ['brain', 'enhancement_mask'] else 'linear')
                for k in ['t1', 'flair', 'brain', 'enhancement_mask'] if k in source}
    left, right = half(a, hit), half(b, ht)
    def normalize(img, mask):
        values = img.numpy()[mask.numpy() > 0]
        if not values.size or np.std(values) <= 0:
            raise ValueError('Empty mask or constant registration image')
        return common.ants_like(ants, (img.numpy() - np.mean(values)) / np.std(values), img)
    params = dict(type_of_transform='SyNOnly', initial_transform='Identity',
                  grad_step=args.grad_step, flow_sigma=args.flow_sigma, total_sigma=0,
                  syn_metric='CC', syn_sampling=args.cc_radius, reg_iterations=tuple(args.iterations),
                  mask_all_stages=True, random_seed=args.seed, verbose=True)
    common.write_json(d / 'registration_parameters.json', params)
    reg = ants.registration(fixed=normalize(left['t1'], left['brain']), moving=normalize(right['t1'], right['brain']),
        mask=left['brain'], moving_mask=right['brain'], outprefix=str(d / 'syn_'),
        multivariate_extras=[('CC', normalize(left['flair'], left['brain']), normalize(right['flair'], right['brain']), 1, args.cc_radius)], **params)
    # Image resampling is a pull map: reference baseline-half coordinates -> follow-up-half coordinates.
    field = next((p for p in reg['fwdtransforms'] if p.endswith('.nii.gz')), None)
    if field is None:
        raise ValueError('Registration did not produce a displacement field')
    jac = ants.create_jacobian_determinant_image(grid, field, do_log=False, geom=True)
    canonical = ants.resample_image(t0, (1, 1, 1), use_voxels=False)
    outputs = {}
    warped_brain = ants.apply_transforms(fixed=grid, moving=right['brain'], transformlist=reg['fwdtransforms'], interpolator='nearestNeighbor')
    support = common.ants_like(ants, (left['brain'].numpy() > 0) & (warped_brain.numpy() > 0), grid)
    images = {'jacobian': jac, 'support': support, 'baseline_half_t1': left['t1'],
              'followup_warped_t1': reg['warpedmovout']}
    if 'enhancement_mask' in right:
        images['enhancement_mask'] = ants.apply_transforms(fixed=grid, moving=right['enhancement_mask'],
            transformlist=reg['fwdtransforms'], interpolator='nearestNeighbor')
    for k, im in images.items():
        interp = 'nearestneighbor' if k in ['support', 'enhancement_mask'] else 'linear'
        back = ht.apply_to_image(im, reference=canonical, interpolation=interp)
        if k == 'jacobian':
            data = back.numpy()
            valid = ht.apply_to_image(support, reference=canonical, interpolation='nearestneighbor').numpy() > 0
            if not np.isfinite(data[valid]).all() or np.any(data[valid] <= 0):
                raise ValueError('Nonpositive/nonfinite Jacobians inside analysis support')
            data[~valid] = 1
            back = common.ants_like(ants, data, canonical)
        path = str(d / (k + '.nii.gz')); ants.image_write(back, path); outputs[k] = path
    common.write_json(d / 'jacobian_convention.json', {'direction': 'baseline_to_followup_physical_pull',
        'representation': 'determinant', 'global_affine_excluded': True,
        'space': 'baseline_T1_1mm', 'outside_support': 1})
    return outputs


def main():
    p = common.parser(__doc__)
    p.add_argument('--seed-percent-per-year', type=float, default=12.5)
    p.add_argument('--grow-percent-per-year', type=float, default=4.)
    p.add_argument('--min-candidate-mm3', type=float, default=10.)
    p.add_argument('--shell-mm', type=float, default=1.)
    p.add_argument('--grad-step', type=float, default=.7)
    p.add_argument('--flow-sigma', type=float, default=2.)
    p.add_argument('--cc-radius', type=int, default=2)
    p.add_argument('--iterations', nargs='+', type=int, default=[40, 20, 0])
    p.add_argument('--calibration', type=Path)
    args = p.parse_args()
    vals = [args.seed_percent_per_year, args.grow_percent_per_year, args.min_candidate_mm3, args.shell_mm, args.grad_step, args.flow_sigma]
    if not all(np.isfinite(v) and v > 0 for v in vals) or args.seed_percent_per_year < args.grow_percent_per_year or args.cc_radius < 1 or min(args.iterations) < 0 or max(args.iterations) == 0:
        p.error('Invalid thresholds or registration parameters')
    rows = common.load_manifest(args)
    calibration = load_calibration(args.calibration)
    if args.dry_run:
        print(json.dumps({'visits': rows, 'temporal_assessable': len(rows) >= 3,
                          'score_calibrated': calibration is not None, 'method': 'research_adaptation'}, indent=2)); return
    with common.run_directory(args, rows, 'sel_deformation_analysis') as root:
        ants = common.configure_ants(args)
        prepared = [common.prepare(r, args, root) for r in rows]
        base = ants.resample_image(ants.image_read(prepared[0]['t1']), (1, 1, 1), use_voxels=False)
        basepath = root / 'baseline_t1_1mm.nii.gz'; ants.image_write(base, str(basepath))
        lesion = ants.resample_image_to_target(ants.image_read(prepared[0]['lesion']), base, interp_type='nearestNeighbor').numpy() > 0
        support = ants.resample_image_to_target(ants.image_read(prepared[0]['brain']), base, interp_type='nearestNeighbor').numpy() > 0
        enhancement_known = all(r['enhancement_mask'] for r in rows)
        enhancing = np.zeros(lesion.shape, bool)
        if rows[0]['enhancement_mask']:
            enhancing |= ants.resample_image_to_target(ants.image_read(prepared[0]['enhancement_mask']), base, interp_type='nearestNeighbor').numpy() > 0
        maps = []
        for i in range(1, len(rows)):
            pair = rows[0]['session'] + '_to_' + rows[i]['session']
            print(f'Registering {pair}', flush=True)
            paths = common.cached(root / 'pairs' / pair, lambda d: deformation_pair(prepared[0], prepared[i], args, d))
            common.qc_registration(root / 'pairs' / pair / 'registration_qc.png', paths['baseline_half_t1'], paths['followup_warped_t1'], pair)
            jac = ants.image_read(paths['jacobian']).numpy()
            annual = annual_expansion(jac, rows[i]['years'])
            ants.image_write(common.ants_like(ants, annual, base), str(root / 'pairs' / pair / 'annual_percent.nii.gz'))
            support &= ants.image_read(paths['support']).numpy() > 0
            if 'enhancement_mask' in paths: enhancing |= ants.image_read(paths['enhancement_mask']).numpy() > 0
            maps.append(jac)
        rate = annual_expansion(maps[-1], rows[-1]['years'])
        voxel = float(np.prod(base.spacing))
        labels = candidates(lesion & support, rate, args.seed_percent_per_year, args.grow_percent_per_year,
                            int(np.ceil(args.min_candidate_mm3 / voxel)))
        passed = np.zeros(labels.shape, np.uint8)
        records, details = [], []
        for ident in np.unique(labels):
            if ident == 0: continue
            mask = labels == ident
            changes = [float(100 * (j[mask].mean() - 1)) for j in maps]
            temporal = temporal_score([r['years'] for r in rows[1:]], changes)
            radial, shells = radial_score(mask, rate, base.spacing, args.shell_mm)
            status, score = classify(temporal, radial, calibration, bool(np.any(mask & enhancing)), enhancement_known)
            if status == 'adapted_score_pass': passed[mask] = 1
            records.append(dict(candidate_id=int(ident), baseline_region_mm3=float(mask.sum() * voxel),
                final_mean_expansion_percent=changes[-1], annual_mean_expansion_percent=changes[-1] / rows[-1]['years'],
                **temporal, concentricity_inward_slope=radial, heuristic_score=score, status=status))
            details.append(dict(candidate_id=int(ident), years=[r['years'] for r in rows[1:]], mean_expansion_percent=changes, shells=shells))
        for name, data in [('candidates', labels), ('adapted_score_pass', passed), ('analysis_support', support)]:
            ants.image_write(common.ants_like(ants, data, base), str(root / (name + '.nii.gz')))
        fields = ['candidate_id', 'baseline_region_mm3', 'final_mean_expansion_percent', 'annual_mean_expansion_percent',
                  'slope_percent_per_year', 'constancy_nmse', 'temporal_status', 'concentricity_inward_slope', 'heuristic_score', 'status']
        common.write_csv(root / 'candidates.csv', records, fields)
        common.write_json(root / 'candidate_profiles.json', details)
        common.plot_profiles(root / 'candidate_trajectories.png',
            [(str(r['candidate_id']), [0] + r['years'], [0] + r['mean_expansion_percent']) for r in details[:12]],
            'Years since baseline', 'Mean local expansion (%) - first 12 candidates')
        common.write_json(root / 'summary.json', {'candidate_count': len(records), 'candidate_region_mm3': float((labels > 0).sum() * voxel),
            'adapted_score_pass_region_mm3': float(passed.sum() * voxel), 'baseline_lesion_outside_support_mm3': float((lesion & ~support).sum() * voxel),
            'temporal_assessable': len(rows) >= 3, 'score_model': SCORE_MODEL, 'calibration': calibration,
            'limitations': ['T1/FLAIR, registration schedule, 1mm-grid size filter, growth and scoring are explicit adaptations.',
                           'No original cohort calibration: no claim of Elliott high-confidence classification.',
                           'Affine/global scale is excluded; region volume is not added lesion volume.']})
        common.qc_overlay(root / 'candidates_qc.png', basepath, root / 'candidates.nii.gz', 'SEL candidates (research adaptation)')
        print(f'Wrote results to {root}')


if __name__ == '__main__':
    common.execute(main)
