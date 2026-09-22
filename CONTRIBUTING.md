# Contributing to MaleCNS

This is an early student project. Clear explanations, reproducible bug reports and focused reviews are especially useful. Start by opening an issue describing one concrete problem or a roadmap item you want to explore.

## Useful first contributions

1. Map the imports and data requirements of a single entry point.
2. Reproduce one experiment and document the exact environment and inputs.
3. Review a retina/laterality mapping or a motor-response assumption.
4. Profile a bottleneck and report measurements before proposing optimization.

## A helpful report includes

- Script/version, operating system and Python/package versions.
- Data version and the exact processed-file set, without uploading private files.
- Seed, configuration, reproduction steps, expected result and observed result.
- Redacted output and a small synthetic reproduction when possible.

## Pull requests

Keep one purpose per change. Explain the observable behaviour before and after the change and how it was checked. Preserve older experiments unless cleanup is explicitly part of the issue. Never claim biological validity or speed improvements from a syntax check alone.

No project-wide code license has been chosen yet. Discuss substantial contributions or reuse with the maintainer first; this document does not grant additional rights.

## Privacy

Do not commit API keys, OBS credentials, personal screen captures or raw error logs. The `.gitignore` excludes known local outputs, but review staged files before every commit. Screenshots used in issues should come from a dedicated synthetic test scene.
