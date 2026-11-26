# Paste into the Colab notebook as a new cell and run it, then leave the tab open.
# It waits until the background runs (run_all.sh) finish, writes a summary table to Drive,
# then disconnects the runtime so no GPU time is wasted. Safe to run even if the runs already died.
import json, os, subprocess, time

DRIVE_RUNS = '/content/drive/MyDrive/cyasr/runs'
LOG = f'{DRIVE_RUNS}/qwen_rumble_dev.log'
OUT = f'{DRIVE_RUNS}/qwen_rumble_summary.txt'

def running():
    # [r]/[c]: the pattern must not match the shell that runs pgrep itself
    return subprocess.run('pgrep -f "[r]un_all.sh|[c]yasr.run_baseline"', shell=True,
                          capture_output=True).returncode == 0

while running():
    time.sleep(120)

lines = []
try:
    import jiwer
    os.chdir('/content/cyasr')
    from cyasr.normalize import normalize
    runs = [('whisper-small', 'baseline_whisper-small_fallback_dev'),
            ('whisper-medium', 'baseline_whisper-medium_fallback_dev'),
            ('whisper-large-v3', 'baseline_whisper-large-v3_fallback_dev'),
            ('qwen3-asr-0.6b', 'baseline_qwen3-asr-0.6b_greedy_dev'),
            ('rumble-2-flash', 'baseline_rumble-2-flash_greedy_dev'),
            ('rumble matched', 'baseline_rumble-2-flash_matched_greedy_dev')]
    bucket = lambda d: '<=30s' if d <= 30 else ('30-33s' if d <= 33 else '>33s')
    lines.append(f"{'model':17} {'WER basic':>9} {'nofill':>7} {'nodiac':>7} {'phon.':>6} {'CER nodiac':>10} "
                 f"{'<=30s':>6} {'30-33s':>7} {'>33s':>6} {'halluc':>6} {'cutoff':>6}")
    for label, name in runs:
        path = f'{DRIVE_RUNS}/{name}'
        if not os.path.exists(f'{path}/metrics.json'):
            lines.append(f'{label:17} missing'); continue
        m = json.load(open(f'{path}/metrics.json')); lv = m['levels']
        errs, words = {}, {}
        for r in map(json.loads, open(f'{path}/predictions.jsonl', encoding='utf-8')):
            ref, hyp = normalize(r['ref'], 'nodiac'), normalize(r['hyp'], 'nodiac')
            o = jiwer.process_words(ref, hyp)
            b = bucket(r['duration_s'])
            errs[b] = errs.get(b, 0) + o.substitutions + o.deletions + o.insertions
            words[b] = words.get(b, 0) + len(ref.split())
        w = {b: errs[b] / words[b] for b in errs}
        lines.append(f"{label:17} {lv['basic']['wer']:9.1%} {lv['nofill']['wer']:7.1%} {lv['nodiac']['wer']:7.1%} "
                     f"{lv['phonetic']['wer']:6.1%} {lv['nodiac']['cer']:10.1%} {w.get('<=30s', 0):6.1%} "
                     f"{w.get('30-33s', 0):7.1%} {w.get('>33s', 0):6.1%} {m['hallucinations']:6} "
                     f"{str(m.get('hit_max_tokens', '-')):>6}")
except Exception as e:  # never block the disconnect
    lines.append(f'summary failed: {e!r}')
lines.append('')
lines.append(open(LOG).read()[-1500:] if os.path.exists(LOG) else 'no log')
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines))
print('\n'.join(lines))

from google.colab import drive, runtime
drive.flush_and_unmount()  # make sure everything is written to Drive
runtime.unassign()         # disconnect and release the GPU
