# Vision-OPD Red-Box Contamination Audit

Vision-OPD is currently treated as infrastructure validation only because local
images appear to contain baked-in red bounding-box localization cues.

Latest HPC audit result:

- `num_samples_checked = 128`
- `full_image_red_box_suspected_rate = 0.9140625`
- `bbox_image_red_box_suspected_rate = 1.0`
- `likely_conclusion = red_box_likely_baked_into_full_images`

Run the audit:

```bash
DATASET=/path/to/Vision-OPD/train.parquet \
DTOPD_OUTPUT_ROOT=/path/to/outputs \
scripts/hpc/audit_vision_opd_red_box_contamination.sh
```

The audit reads `images` and `bbox_images`, computes red-pixel and strong-red
ratios, records simple connected-component evidence, writes per-sample JSONL and
summary JSON, exports a contact sheet, and copies high-red examples for human
inspection.

Outputs:

- `red_box_audit.jsonl`
- `red_box_audit_summary.json`
- `red_box_contact_sheet.jpg`
- `examples/`

If `red_box_contamination_suspected=true`, Vision-OPD full 2C/4C builders must
not be used as main experimental evidence. The 2C builder refuses full-ish
Vision-OPD runs by default; `--allow-red-box-contaminated-images` marks the run
as `red_box_contaminated=true`, `localization_cued_ablation=true`, and
`not_main_experiment=true`.
