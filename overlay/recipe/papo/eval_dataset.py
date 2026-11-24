"""Bound new benchmark image resolution while preserving Geo3K inputs exactly."""

from verl.utils.dataset.rl_dataset import RLHFDataset


class PAPOEvaluationDataset(RLHFDataset):
    def _build_messages(self, example, key):
        messages = super()._build_messages(example, key)
        if example.get("data_source", "").startswith("papo_eval/"):
            max_pixels = int(self.config.get("papo_eval_max_pixels", 1048576))
            if max_pixels <= 0:
                raise ValueError("papo_eval_max_pixels must be positive")
            for message in messages:
                content = message["content"]
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "image":
                            item["max_pixels"] = max_pixels
        return messages
