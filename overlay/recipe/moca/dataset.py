"""Keep generalization images within the previously frozen pixel budget."""

from verl.utils.dataset.rl_dataset import RLHFDataset


class MoCADataset(RLHFDataset):
    def _build_messages(self, example, key):
        messages = super()._build_messages(example, key)
        if example.get("data_source", "").startswith("papo_eval/"):
            for message in messages:
                if isinstance(message["content"], list):
                    for part in message["content"]:
                        if part.get("type") == "image":
                            part["max_pixels"] = 1048576
        return messages
