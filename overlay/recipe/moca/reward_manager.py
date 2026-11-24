"""Online MoCA rewards; benchmark accuracy remains the existing rule-based metric."""

import hashlib
import json
import os
from pathlib import Path

from recipe.moca.evidence import extract_evidence
from recipe.moca.verifier import VerifierClient
from recipe.papo.eval_reward import compute_score as benchmark_score

from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase


class MoCARewardManager(RewardManagerBase):
    def __init__(self, config, tokenizer, compute_score, **kwargs):
        super().__init__(config, tokenizer, compute_score)
        self.settings = config.moca
        self.verifier = VerifierClient(self.settings.verifier)

    async def run_single(self, data):
        item = data[-1]
        ids = item.batch["responses"]
        length = int(item.batch["attention_mask"][-len(ids) :].sum())
        response = self.tokenizer.decode(ids[:length], skip_special_tokens=True)
        source = item.non_tensor_batch["data_source"]
        truth = item.non_tensor_batch["reward_model"]["ground_truth"]
        extra = item.non_tensor_batch["extra_info"]
        if "moca_train" not in extra:
            raise ValueError("MoCA requires prepared data with an explicit train/validation marker")
        accuracy = float(benchmark_score(source, response, truth, extra_info=extra)["acc"])
        evidence = extract_evidence(response)
        info = {
            "acc": accuracy,
            "moca_outcome": 0.0,
            "moca_perception": 0.0,
            "moca_format_valid": float(evidence.valid),
            "moca_evidence_tokens": float(len(self.tokenizer.encode(evidence.text, add_special_tokens=False))),
            "moca_judge_calls": 0.0,
            "moca_judge_prompt_tokens": 0.0,
            "moca_judge_completion_tokens": 0.0,
        }
        if not extra["moca_train"]:
            # No LLM judging or perception reward enters reported benchmark accuracy.
            return {"reward_score": accuracy, "reward_extra_info": info}
        if source != "hiyouga/geometry3k":
            raise ValueError("The initial MoCA training recipe supports Geo3K training only")
        question = extra["moca_question"]
        outcome = await self.verifier.score(question, str(truth), response)
        calls = [outcome]
        proxy = proxy_score = None
        if evidence.valid:
            proxy = await self.verifier.solve(question, evidence.text)
            proxy_score = await self.verifier.score(question, str(truth), proxy["text"])
            calls.extend([proxy, proxy_score])
            # The paper leaves the fractional-SVV -> binary-PV threshold unspecified.
            info["moca_perception"] = float(proxy_score["score"] >= self.settings.pv_success_threshold)
        info["moca_outcome"] = float(outcome["score"])
        info["moca_judge_calls"] = float(len(calls))
        for kind in ("prompt", "completion"):
            info[f"moca_judge_{kind}_tokens"] = float(sum(c["usage"].get(f"{kind}_tokens", 0) for c in calls))
        reward = info["moca_outcome"] + float(self.settings.perception_weight) * info["moca_perception"]
        audit_dir = self.settings.get("audit_dir")
        if audit_dir:
            path = Path(audit_dir)
            path.mkdir(parents=True, exist_ok=True)
            record = {
                "id": hashlib.sha256((question + "\0" + response).encode()).hexdigest(),
                "index": extra.get("index"),
                "question": question,
                "ground_truth": truth,
                "response": response,
                "evidence": evidence.text,
                "outcome_judgment": outcome,
                "proxy_answer": proxy,
                "proxy_judgment": proxy_score,
                "reward": reward,
                "metrics": info,
            }
            with (path / f"judge-{os.getpid()}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"reward_score": reward, "reward_extra_info": info}
