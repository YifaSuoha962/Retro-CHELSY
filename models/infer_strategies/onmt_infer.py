import numpy as np
import torch

from .infer_base import BeamGenerateBase
from .onmt_beamsearch import GNMTGlobalScorer, BeamSearch


class Beam_Generate(BeamGenerateBase):
    def __init__(
        self,
        beam_size: int,
        batch_size: int,
        bos_token_ids: int,
        pad_token_ids: int,
        eos_token_ids: int,
        unk_token_ids: int,
        length_penalty=1.,
        min_len=1,
        max_len=256,
        temperature=1.,
        top_k=0,
        top_p=0.,
        return_num=10,
        device='cpu'
    ):
        super(Beam_Generate, self).__init__(
            beam_size=beam_size,
            batch_size=batch_size,
            bos_token_ids=bos_token_ids,
            pad_token_ids=pad_token_ids,
            eos_token_ids=eos_token_ids,
            length_penalty=length_penalty,
            min_len=min_len,
            max_len=max_len,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            return_num=return_num,
            device=device
        )
        self.unk_token_ids = unk_token_ids

        self.beam_search = BeamSearch(
            beam_size=self.beam_size,
            batch_size=self.batch_size,
            pad=self.pad_token_ids,
            bos=self.bos_token_ids,
            eos=self.eos_token_ids,
            unk=self.unk_token_ids,
            n_best=self.beam_size,
            global_scorer=GNMTGlobalScorer(
                alpha=self.length_penalty,
                beta=0.,
                length_penalty='avg' if self.length_penalty > 0. else 'none',
                coverage_penalty='none'
            ),
            min_length=self.min_len,
            max_length=self.max_len,
            return_attention=False,
            block_ngram_repeat=0,
            exclusion_tokens=set(),
            stepwise_penalty=None,
            ratio=0.,
            ban_unk_token=False
        )

    @property
    def current_token(self) -> torch.Tensor:
        return self.beam_search.current_predictions

    @property
    def any_finish(self) -> bool:
        return self.beam_search.is_finished.any()
    
    @property
    def is_done(self) -> bool:
        return self.beam_search.done
    
    @property
    def mem_ids(self) -> torch.Tensor:
        return self.beam_search.select_indices

    def _prepare(
        self,
        memory_bank: torch.Tensor,
        src_lengths: torch.Tensor,
        src_map=None,
        target_prefix=None
    ):
        _, _, _, _ = self.beam_search.initialize(
            memory_bank=memory_bank,
            src_lengths=src_lengths,
            src_map=src_map,
            device=self.device,
            target_prefix=target_prefix
        )
        self.step = 0

    def _sample_process(
        self,
        # each sequence's sum logits score, size(batch * beam, vocab_size)
        scores: torch.Tensor,
        min_keep=2
    ):
        if self.top_k > 0:
            top_k = min(max(self.top_k, min_keep), scores.size(-1))
            # select the lowest score in topk
            remove_ids = scores < torch.topk(scores, top_k, dim=-1)[0][..., -1, None]
            scores = scores.masked_fill(remove_ids, -float('inf'))
        if self.top_p > 0.:
            sorted_logits, sorted_ids = torch.sort(scores, descending=True)
            cumsum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            # cumsum calculate the sum of each scores, which can easily find the threshold
            remove_sorted_ids = cumsum_probs > self.top_p
            # the next three step to finish the min_keep, then accept the threshold and the first token.
            remove_sorted_ids[..., :min_keep - 1] = False
            remove_sorted_ids[..., 1:] = remove_sorted_ids[..., :-1].clone()
            remove_sorted_ids[..., 0] = False
            remove_ids = remove_sorted_ids.scatter(dim=1, index=sorted_ids, src=remove_sorted_ids)
            scores = scores.masked_fill(remove_ids, -float('inf'))
        return scores

    def generate(
        self,
        # the decoder output like [unfinish_batch * beam, 1, vocab_size], before log_softmax
        dec_output: torch.Tensor,
        lat_prob=None
    ):
        dec_output = dec_output[:, -1, :]
        next_token_logits = (dec_output / self.temperature).log_softmax(dim=-1)
        if lat_prob is not None:
            next_token_logits += lat_prob
        next_token_logits = self._sample_process(next_token_logits, min_keep=self.beam_size) if self.step == 0\
            else self._sample_process(next_token_logits)
        
        self.beam_search.advance(next_token_logits, None)
        if self.any_finish:
            self.beam_search.update_finished()
        self.step += 1
    
    def finish_generate(
        self
    ):
        predictions, scores = self.beam_search.predictions, self.beam_search.scores
        max_result_len = 0
        for batch_res in predictions:
            for beam_res in batch_res:
                max_result_len = len(beam_res) if len(beam_res) > max_result_len\
                    else max_result_len
        
        seq_result, seq_scores = torch.zeros((self.batch_size, self.return_num, max_result_len), dtype = torch.long),\
            torch.zeros((self.batch_size, self.return_num), dtype = torch.float)
        for batch_id, (batch_res, batch_score) in enumerate(zip(predictions, scores)):
            for beam_id, (beam_res, beam_score) in enumerate(zip(batch_res, batch_score)):
                if beam_id >= self.return_num: break
                beam_res = beam_res.detach().cpu()
                pad_beam_res = torch.cat([beam_res, torch.tensor([self.pad_token_ids] * \
                                         (max_result_len - len(beam_res)), dtype = torch.long)], dim = 0)
                seq_result[batch_id, beam_id] = pad_beam_res
                seq_scores[batch_id, beam_id] = beam_score
    
        return seq_result, seq_scores