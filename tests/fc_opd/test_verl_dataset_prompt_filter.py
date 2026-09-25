from dual_track_opd.fc_opd.verl_dataset import FCOPDDataset


class _FakeProcessor:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def tokenizer(self, text, **kwargs):
        return {"input_ids": list(range(len(text)))}


def _make_dataset():
    return _FakeDataset(
        [
            {"prompt": [{"content": "short", "role": "user"}], "images": []},
            {"prompt": [{"content": "x" * 32, "role": "user"}], "images": []},
        ]
    )


class _FakeDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def select(self, indices):
        return _FakeDataset([self.rows[index] for index in indices])


def test_fcop_dataset_filters_overlong_prompts_instead_of_skipping():
    dataset = _make_dataset()
    instance = object.__new__(FCOPDDataset)
    instance.filter_overlong_prompts = True
    instance.max_prompt_length = 10
    instance.prompt_key = "prompt"
    instance.image_key = "images"
    instance.video_key = "videos"
    instance.audio_key = "audios"
    instance.processor = _FakeProcessor()
    instance.apply_chat_template_kwargs = {}
    instance.tool_schemas = None
    instance.image_patch_size = None
    instance.config = {}
    instance.mm_processor_kwargs = {}

    filtered = instance.maybe_filter_out_long_prompts(dataset)

    assert len(filtered) == 1
    assert filtered[0]["prompt"][0]["content"] == "short"
