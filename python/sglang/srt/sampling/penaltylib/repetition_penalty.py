import torch

from sglang.srt.sampling.penaltylib.orchestrator import _BatchedPenalizer


class BatchedRepetitionPenalizer(_BatchedPenalizer):
    """
    HuggingFace-style repetition penalty.

    For each token that has appeared in the input_ids history (prompt + generated):
      - if logit > 0: logit /= penalty
      - if logit < 0: logit *= penalty

    This penalty is multiplicative and cannot be represented as an additive logit bias.
    In overlap mode we snapshot (penalty, seen_tokens) into SamplingBatchInfo and apply
    it on the model worker side.
    """

    def _is_required(self) -> bool:
        return any(
            req.sampling_params.repetition_penalty != 1.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        # Shape: [B, 1] for broadcasting.
        self.repetition_penalties = (
            torch.tensor(
                [req.sampling_params.repetition_penalty for req in self.orchestrator.reqs()],
                dtype=torch.float32,
                device=self.orchestrator.device,
            )
        ).unsqueeze_(1)

        # Track which token ids have been seen per request.
        self.seen_tokens = torch.zeros(
            (len(self.orchestrator.reqs()), self.orchestrator.vocab_size),
            dtype=torch.bool,
            device=self.orchestrator.device,
        )

        # Seed with prompt + any already-generated output (e.g., after merges).
        for i, req in enumerate(self.orchestrator.reqs()):
            token_ids = []
            if getattr(req, "origin_input_ids", None):
                token_ids.extend(req.origin_input_ids)
            if getattr(req, "output_ids", None):
                token_ids.extend(req.output_ids)
            if token_ids:
                ids = torch.tensor(token_ids, dtype=torch.int64, device=self.orchestrator.device)
                self.seen_tokens[i].scatter_(0, ids, True)

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        # output_ids: [B]
        self.seen_tokens.scatter_(dim=1, index=output_ids.unsqueeze(1), value=True)

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: [B, V]
        penalty = self.repetition_penalties.to(dtype=logits.dtype)
        inv_penalty = (1.0 / self.repetition_penalties).to(dtype=logits.dtype)

        neg = logits < 0
        mask = self.seen_tokens
        logits.mul_(torch.where(mask & neg, penalty, 1.0))
        logits.mul_(torch.where(mask & (~neg), inv_penalty, 1.0))

    def _filter(self, keep_indices: torch.Tensor):
        self.repetition_penalties = self.repetition_penalties[keep_indices]
        self.seen_tokens = self.seen_tokens[keep_indices]

    def _merge(self, their: "BatchedRepetitionPenalizer"):
        self.repetition_penalties = torch.cat(
            [self.repetition_penalties, their.repetition_penalties], dim=0
        )
        self.seen_tokens = torch.cat([self.seen_tokens, their.seen_tokens], dim=0)

    def _teardown(self) -> None:
        for name in ("repetition_penalties", "seen_tokens"):
            if hasattr(self, name):
                delattr(self, name)

