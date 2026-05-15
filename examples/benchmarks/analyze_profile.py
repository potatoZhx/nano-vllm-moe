import json, sys

with open(sys.argv[1]) as f:
    data = json.load(f)
prof = data.get("engine_profile", {})
dc = prof.get("model_decode_count", 1)
label = sys.argv[2] if len(sys.argv) > 2 else ""

print(f"=== {label} ===")
print(f"  decode_count: {dc}")
print(f"  sample_decode_ms (wall):     {prof.get('model_sample_decode_ms', 0):.2f} total, {prof.get('model_sample_decode_ms',0)/max(dc,1):.3f}/call")
print(f"  sample_gpu_decode_ms (GPU):  {prof.get('model_sample_gpu_decode_ms', 0):.2f} total, {prof.get('model_sample_gpu_decode_ms',0)/max(dc,1):.3f}/call")
print(f"  run_model_decode_ms:         {prof.get('model_run_model_decode_ms', 0):.2f} total, {prof.get('model_run_model_decode_ms',0)/max(dc,1):.3f}/call")
if "model_standard_graph_replay_ms" in prof:
    print(f"  standard_graph_replay_ms:    {prof.get('model_standard_graph_replay_ms', 0):.2f} total, {prof.get('model_standard_graph_replay_ms',0)/max(dc,1):.3f}/call")
if "model_draft_graph_replay_ms" in prof:
    print(f"  draft_graph_replay_ms:       {prof.get('model_draft_graph_replay_ms', 0):.2f} total, {prof.get('model_draft_graph_replay_ms',0)/max(dc,1):.3f}/call")
print(f"  sample_gpu_pct_of_sample_ms: {prof.get('model_sample_gpu_decode_ms', 0)/max(prof.get('model_sample_decode_ms', 1), 0.001)*100:.1f}%")
print(f"  sample_pct_of_run_model:     {prof.get('model_sample_decode_ms', 0)/max(prof.get('model_run_model_decode_ms', 1), 0.001)*100:.1f}%")
