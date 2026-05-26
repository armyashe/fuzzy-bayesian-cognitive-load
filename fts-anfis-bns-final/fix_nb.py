"""
Fix Figure 4.5 diagnostic prints in output.ipynb.
Updates the verification block to include trace mode, BN soft evidence,
has_soft_trace, and has_soft_params diagnostic lines.
"""
import json

NB_PATH = r'c:\Users\doqua\fuzzy-bayesian-cognitive-load\fts-anfis-bns-final\notebooks\output.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

updated = False
for cell in nb['cells']:
    if cell.get('cell_type') == 'code' and cell.get('id') == '9d658859':
        src = cell['source']
        verif_idx = None
        for i, line in enumerate(src):
            if line.strip() == '# Verification':
                verif_idx = i
                break
        if verif_idx is None:
            print('ERROR: Could not find # Verification comment in cell')
            break

        # Build new verification block with all diagnostic prints
        new_verif = [
            '# Verification\n',
            'print(f"Trace mode       : {trace.get(' + repr('mode') + ')}")\n',
            'print(f"Resolved mode    : {mode}")\n',
            'print(f"BN soft evidence : {bn_struct.get(' + repr('soft_evidence') + ', False)}")\n',
            'print(f"Has soft trace   : {has_soft_trace}")\n',
            'print(f"Has soft params  : {has_soft_params}")\n',
            'print(f"Posterior source : {_source}")\n',
            'print(f"Sample index     : {sample_idx}")\n',
            'print(f"User ID          : {user_id}")\n',
            'print(f"Final P(TASK)    : {post_task_final:.4f}")\n',
            'print(f"Final P(REST)    : {post_rest_final:.4f}")\n',
            'print(f"Threshold        : {threshold:.4f}")\n',
            'print(f"Predicted        : {predicted}")\n',
            'print()\n',
            'print(df_trace.to_string(index=False))\n',
        ]

        old_len = len(src)
        cell['source'] = src[:verif_idx] + new_verif
        print(f'Updated: kept {verif_idx} lines, replaced verification block')
        print(f'Old total: {old_len} lines, new total: {len(cell["source"])} lines')
        updated = True
        break

if updated:
    with open(NB_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print('Saved notebook.')
else:
    print('No changes made.')

