import json

from PIL import Image

from dual_track_opd.fc_opd.degradation import materialize_degraded_image
from dual_track_opd.fc_opd.geometry3k_adapter import load_geometry3k_records


def test_degraded_writer_uses_output_root_not_sample_dir(tmp_path, monkeypatch):
    root = tmp_path / "unzipped"
    sample_dir = root / "train" / "train" / "0"
    sample_dir.mkdir(parents=True)
    image = sample_dir / "img_diagram.png"
    Image.new("RGB", (24, 24), "white").save(image)
    Image.new("RGB", (24, 24), "gray").save(sample_dir / "img_diagram.lowres_10pct_nearest.png")
    (sample_dir / "data.json").write_text(
        json.dumps({"id": "0", "compact_text": "Question?", "choices": ["A", "B"], "answer": "A"}),
        encoding="utf-8",
    )
    (sample_dir / "logic_form.json").write_text("{}", encoding="utf-8")
    output_root = tmp_path / "outputs"
    monkeypatch.setenv("DTOPD_OUTPUT_ROOT", str(output_root))

    record = load_geometry3k_records(root)[0]
    degraded = materialize_degraded_image(record["image_path"], mode="lowres_50_bilinear_nearest")

    assert record["image_path"] == str(image.resolve(strict=False))
    assert str(degraded).startswith(str(output_root / "fc_opd" / "degraded_images"))
    assert degraded.endswith(".lowres_50_bilinear_nearest.png")


def test_lowres_10pct_uses_bilinear_downsample_and_nearest_upsample(tmp_path):
    source = tmp_path / "source.png"
    pixels = Image.new("RGB", (20, 10), "black")
    for x in range(20):
        for y in range(10):
            pixels.putpixel((x, y), (x * 10, y * 20, 0))
    pixels.save(source)

    degraded = materialize_degraded_image(
        str(source),
        mode="lowres_10pct_nearest",
        degraded_dir=str(tmp_path / "degraded"),
    )

    with Image.open(degraded) as image:
        assert image.size == (20, 10)
        # 10% of 20x10 is 2x1, then nearest upsampling creates two vertical
        # bands. A pure nearest downsample would preserve exact source pixels;
        # bilinear produces interpolated colors.
        assert len({image.getpixel((x, 0)) for x in range(20)}) == 2
        assert image.getpixel((0, 0)) not in {pixels.getpixel((0, 0)), pixels.getpixel((19, 0))}
