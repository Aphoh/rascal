from torch import nn
import torch.nn.functional as F
import torch


class Router(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, topk: int):
        super().__init__()
        self.topk = topk
        self.fc = nn.Linear(hidden_size, num_classes)
        assert topk != 1, "Topk should be greater than 1"
        assert (
            topk <= num_classes
        ), "Topk should be less than or equal to number of classes"

    def forward(self, x):
        preds = self.fc(x)
        sm = F.softmax(preds, dim=-1)
        probs, top_indices = torch.topk(sm, k=self.topk, dim=-1)
        masked_gates = torch.zeros_like(preds).scatter(-1, top_indices, probs)
        return masked_gates


class RoutedLoraLLM(nn.Module):
    def __init__(self, llm, hidden_size: int, num_classes: int, topk: int):
        super().__init__()
        self.llm = llm
        self.router = Router(hidden_size, num_classes, topk)

    @torch.no_grad()
    def get_suffix_hiddens(self, llm_inputs: dict, suffix_idx: torch.Tensor):
        hiddens = self.llm(return_hidden=True, **llm_inputs)
        suffix_indices = (
            suffix_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, hiddens.shape[-1])
        )
        res = hiddens.gather(1, suffix_indices).squeeze(1)
        return res

    def forward(self, llm_inputs: dict, suffix_idx: torch.Tensor, lens: torch.Tensor) -> tuple[torch.Tensor, int]:
        b, s = llm_inputs["input_ids"].shape
        hiddens = self.get_suffix_hiddens(llm_inputs, suffix_idx)
        masked_gates = self.router(hiddens)
        logits = self.llm(masked_gates=masked_gates, **llm_inputs)
        target_mask = (
            torch.arange(0, s, device=logits.device)[None, :] > suffix_idx[:, None]
        )
        target_mask = target_mask & (
            torch.arange(0, s, device=logits.device)[None, :] < lens[:, None]
        )
        num_tokens = target_mask.sum()
        targets = -100 * ~target_mask + target_mask * llm_inputs["input_ids"]
        targets = targets[:, 1:].view(-1)
        logits = logits[:, :-1].view(-1, logits.shape[-1])
        loss = F.cross_entropy(logits, targets, ignore_index=-100, reduction="sum")
        return loss / num_tokens, num_tokens
