# MLIE-CoT controlled router ablations

The entry point is `inference/main_method/route_ablation.py`.
It compares learned routing with fixed L1, L2, and L3 while retaining the same
diagnosis, action, review, and model-identity checks.

```bash
python inference/main_method/route_ablation.py --help
```

`protocol.py` freezes policy identities, `initials.py` reuses exact initial
images from a compatible source run, and `audit.py` checks output integrity.
Supply real immutable model snapshots and a compatible checkpoint for the
learned strategy. Review the CLI options for source-run reuse and audit modes;
the root README explains general inference setup.
