"""Standalone native-volume lesion tracking with reviewed chronic-tissue summaries."""
import csv
import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy import ndimage
if __package__:
    from . import lesion_longitudinal_common as common
else:
    import lesion_longitudinal_common as common


def correspondence(a, b, spacing, tolerance_mm=0):
    """Bipartite overlap graph retaining splits, merges and unmatched lesions."""
    overlap = (a > 0) & (b > 0)
    edges = set(map(tuple, np.unique(np.c_[a[overlap], b[overlap]], axis=0)))
    near = set()
    if tolerance_mm > 0 and np.any(a):
        distance, indices = ndimage.distance_transform_edt(a == 0, sampling=spacing, return_indices=True)
        nearest = a[tuple(indices)]
        use = (b > 0) & (distance <= tolerance_mm) & ~np.isin(b, [j for _, j in edges])
        near.update(map(tuple, np.unique(np.c_[nearest[use], b[use]], axis=0)))
        edges |= near
    nodes = {(0, int(i)) for i in np.unique(a) if i} | {(1, int(i)) for i in np.unique(b) if i}
    graph = {n: set() for n in nodes}
    for i, j in edges:
        x, y = (0, int(i)), (1, int(j))
        graph[x].add(y); graph[y].add(x)
    groups = []
    while nodes:
        stack, component = [min(nodes)], set()
        while stack:
            n = stack.pop()
            if n not in component:
                component.add(n); stack.extend(graph[n] - component)
        nodes -= component
        aa = sorted(i for side, i in component if side == 0)
        bb = sorted(i for side, i in component if side == 1)
        groups.append((aa, bb, any(i in aa and j in bb for i, j in near)))
    return groups


def change_metrics(v0, v1, years, threshold):
    if years <= 0:
        raise ValueError('Nonpositive interval')
    delta = v1 - v0
    pct = 100 * delta / v0 if v0 else None
    category = 'new' if not v0 else ('disappeared' if not v1 else
               'expanding' if pct > threshold else 'shrinking' if pct < -threshold else 'stable')
    return dict(baseline_mm3=v0, followup_mm3=v1, change_mm3=delta,
                change_mm3_per_year=delta / years, change_percent=pct,
                change_percent_per_year=pct / years if pct is not None else None, volume_class=category)


def native_info(row, args):
    labels, n = ndimage.label(common.lesion_in_brain(row, args), ndimage.generate_binary_structure(3, 3))
    voxel = abs(np.linalg.det(nib.load(row['lesion']).affine[:3, :3]))
    return labels, np.bincount(labels.ravel(), minlength=n + 1) * voxel


def register_pair(a, b, args, directory):
    ants = common.configure_ants(args)
    fixed, moving = ants.image_read(a['t1']), ants.image_read(b['t1'])
    if args.alignment == 'already-aligned':
        if not common.same_grid(nib.load(a['t1']), nib.load(b['t1'])):
            raise ValueError('already-aligned visits must share their physical grid')
        tx = []
    else:
        reg = ants.registration(fixed=fixed, moving=moving, type_of_transform='Rigid',
                                mask=ants.image_read(a['brain']), moving_mask=ants.image_read(b['brain']),
                                mask_all_stages=True, random_seed=args.seed, outprefix=str(directory / 'rigid_'))
        tx = reg['fwdtransforms']
    outputs = {}
    for k in ['labels', 'flair', 'brain']:
        img = ants.apply_transforms(fixed=fixed, moving=ants.image_read(b[k]), transformlist=tx,
                                    interpolator='linear' if k == 'flair' else 'nearestNeighbor')
        p = str(directory / (k + '_followup.nii.gz'))
        ants.image_write(img, p); outputs[k] = p
    return outputs


def main():
    p = common.parser(__doc__)
    p.add_argument('--min-lesion-mm3', type=float, default=50)
    p.add_argument('--stable-percent', type=float, default=10)
    p.add_argument('--match-distance-mm', type=float, default=2)
    p.add_argument('--chronic-age-days', type=float, default=365.25)
    p.add_argument('--review', type=Path)
    args = p.parse_args()
    if any(not np.isfinite(v) or v < 0 for v in [args.min_lesion_mm3, args.match_distance_mm, args.stable_percent, args.chronic_age_days]):
        p.error('Size, distance, tolerance and age must be finite and nonnegative')
    rows = common.load_manifest(args)
    if args.dry_run:
        print(json.dumps({'visits': rows, 'method': 'native_volume_rigid_matching_adaptation'}, indent=2)); return
    review = {}
    if args.review:
        with args.review.open(newline='') as f:
            for r in csv.DictReader(f):
                key = (r['pair'], int(r['group_id']))
                if key in review or r['decision'] not in ['include', 'exclude', 'pending']:
                    raise ValueError('Duplicate review or invalid decision')
                if r['decision'] != 'pending' and not r['reason'].strip():
                    raise ValueError('Review decisions need a reason')
                review[key] = r
    with common.run_directory(args, rows, 'chronic_lesion_volume_analysis') as root:
        prepared = [common.prepare(r, args, root) for r in rows]
        native = [native_info(r, args) for r in rows]
        records, templates, summaries, used = [], [], [], set()
        first_seen = {int(i): rows[0]['time'] for i in np.unique(native[0][0]) if i}
        cumulative, cumulative_complete = 0., True
        for index, (a, b) in enumerate(zip(rows, rows[1:])):
            pair = a['session'] + '_to_' + b['session']
            print(f'Analyzing {pair}', flush=True)
            d = root / 'pairs' / pair
            aligned = common.cached(d, lambda directory: register_pair(prepared[index], prepared[index + 1], args, directory))
            img = nib.load(prepared[index]['labels'])
            la = np.rint(img.get_fdata()).astype(np.int32)
            lb = np.rint(nib.load(aligned['labels']).get_fdata()).astype(np.int32)
            na, va = native[index]; nb, vb = native[index + 1]
            # Match before size filtering, so growth across the cutoff is not called new.
            ka = [int(i) for i in np.flatnonzero(va > 0) if i]
            kb = [int(i) for i in np.flatnonzero(vb > 0) if i]
            lost_a, lost_b = set(ka) - set(np.unique(la)), set(kb) - set(np.unique(lb))
            groups = correspondence(la, lb, img.header.get_zooms()[:3], args.match_distance_mm)
            groups += [([i], [], True) for i in sorted(lost_a)] + [([], [i], True) for i in sorted(lost_b)]
            settings = {k: v for k, v in vars(args).items() if k not in ['review', 'output', 'resume', 'dry_run', 'manifest']}
            files = [common.digest(r[k]) for r in [a, b] for k in
                     ['t1', 'flair', 'lesion', 'brain_mask', 'enhancement_mask', 'chronic_mask'] if r[k]]
            signature = hashlib.sha256(json.dumps([a, b, files, settings, common.digest(__file__), common.digest(common.__file__)], sort_keys=True).encode()).hexdigest()
            enhancements = [nib.load(r['enhancement_mask']).get_fdata() > 0 if r['enhancement_mask'] else None for r in [a, b]]
            chronic = nib.load(a['chronic_mask']).get_fdata() > 0 if a['chronic_mask'] else None
            next_seen, interval = {}, []
            v0 = v1 = 0.
            qc = np.zeros(la.shape, np.int16)
            for gid, (ids0, ids1, proximity) in enumerate(groups, 1):
                m0, m1 = np.isin(na, ids0), np.isin(nb, ids1)
                birth = max([first_seen.get(i, a['time']) for i in ids0], default=b['time'])
                for i in ids1: next_seen[i] = birth
                if max(va[ids0].sum(), vb[ids1].sum()) < args.min_lesion_mm3:
                    continue
                flags = []
                if proximity: flags.append('proximity_match_or_lost_component')
                lost = bool(set(ids0) & lost_a or set(ids1) & lost_b)
                if lost: flags.append('lost_during_resampling')
                if len(ids0) > 1 or len(ids1) > 1: flags.append('split_or_merge')
                if any(e is None for e in enhancements): flags.append('enhancement_unknown')
                enhancing = any(e is not None and np.any(e & m) for e, m in zip(enhancements, [m0, m1]))
                if enhancing: flags.append('enhancing')
                known = a['time'] - birth >= args.chronic_age_days
                if not known and not (chronic is not None and m0.any() and np.all(chronic[m0])):
                    flags.append('chronicity_unknown')
                if ids0 and ids1: flags.append('confluent_activity_requires_review')
                values = change_metrics(float(va[ids0].sum()), float(vb[ids1].sum()), b['years'] - a['years'], args.stable_percent)
                if values['volume_class'] == 'shrinking': flags.append('postacute_shrinkage_requires_review')
                decision = 'exclude' if not ids0 or not ids1 or enhancing else 'pending'
                if lost: decision = 'pending'
                reason = ';'.join(flags) or values['volume_class']
                key = (pair, gid)
                if key in review:
                    rev = review[key]; used.add(key)
                    if rev['signature'] != signature: raise ValueError('Review signature does not match inputs/settings')
                    decision, reason = rev['decision'], rev['reason']
                    if decision == 'include' and (not ids0 or not ids1 or enhancing or lost):
                        raise ValueError('Cannot include new, disappeared, enhancing or lost components as matched chronic tissue')
                if decision == 'include': v0 += values['baseline_mm3']; v1 += values['followup_mm3']
                rec = dict(pair=pair, group_id=gid, baseline_ids=';'.join(map(str, ids0)), followup_ids=';'.join(map(str, ids1)),
                           years=b['years'] - a['years'], **values, flags=';'.join(flags), decision=decision, reason=reason, signature=signature)
                voxel = abs(np.linalg.det(img.affine[:3, :3]))
                rec['aligned_baseline_mm3'] = float(np.isin(la, ids0).sum() * voxel)
                rec['aligned_followup_mm3'] = float(np.isin(lb, ids1).sum() * voxel)
                interval.append(rec)
                templates.append({k: rec[k] for k in ['pair', 'group_id', 'decision', 'reason', 'signature']})
                qc[np.isin(la, ids0) | np.isin(lb, ids1)] = {'include': 1, 'exclude': 2, 'pending': 3}[decision]
            first_seen = next_seen
            pending = sum(r['decision'] == 'pending' for r in interval)
            cumulative_complete &= pending == 0
            cumulative += v1 - v0
            summaries.append(dict(pair=pair, years=b['years'] - a['years'], all_native_baseline_mm3=float(va[1:].sum()),
                all_native_followup_mm3=float(vb[1:].sum()), reviewed_baseline_mm3=v0, reviewed_followup_mm3=v1,
                reviewed_change_mm3=v1 - v0, reviewed_change_mm3_per_year=(v1 - v0) / (b['years'] - a['years']),
                pending_groups=pending, review_complete=pending == 0,
                cumulative_reviewed_change_mm3=cumulative if cumulative_complete else None))
            records.extend(interval)
            qpath = d / 'review_labels.nii.gz'
            nib.save(nib.Nifti1Image(qc, img.affine), qpath)
            common.qc_overlay(d / 'review_qc.png', prepared[index]['flair'], qpath, pair + ': included=1 excluded=2 pending=3')
            common.qc_registration(d / 'registration_qc.png', prepared[index]['flair'], aligned['flair'], pair)
        if set(review) != used: raise ValueError('Unknown review pair/group IDs')
        fields = ['pair', 'group_id', 'baseline_ids', 'followup_ids', 'years', 'baseline_mm3', 'followup_mm3', 'change_mm3',
                  'change_mm3_per_year', 'change_percent', 'change_percent_per_year', 'volume_class',
                  'aligned_baseline_mm3', 'aligned_followup_mm3', 'flags', 'decision', 'reason', 'signature']
        common.write_csv(root / 'lesion_groups.csv', records, fields)
        common.write_csv(root / 'review_template.csv', templates, ['pair', 'group_id', 'decision', 'reason', 'signature'])
        common.write_csv(root / 'interval_summary.csv', summaries, list(summaries[0]))
        common.plot_profiles(root / 'native_volume_trajectory.png', [('All native lesions', [r['years'] for r in rows],
                             [float(v[1:].sum()) for _, v in native])], 'Years since baseline', 'Native lesion volume (mm3)')
        common.write_json(root / 'summary.json', {'intervals': summaries, 'method_status': 'research_adaptation',
            'limitations': ['Native volumes with rigid matching; no automatic Klistorner confluent-lesion classifier.',
                           'Chronic tissue requires explicit review of confluent activity, chronicity and enhancement.',
                           '10% stability tolerance is not calibrated for LST-AI.']})
        print(f'Wrote results to {root}')


if __name__ == '__main__':
    common.execute(main)
