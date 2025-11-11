import math

from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion


@register_criterion('label_smoothed_cross_entropy_with_moe')
class LabelSmoothedCrossEntropyWithMoECriterion(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, moe_loss_coef):
        super().__init__(task, sentence_avg, label_smoothing)
        self.moe_loss_coef = moe_loss_coef

    @staticmethod
    def add_args(parser):
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument('--moe-loss-coef', default=0.01, type=float, metavar='D',
                            help='coefficient for MoE auxiliary loss')

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        moe_loss = net_output[1].get('moe_loss', 0.0)
        if moe_loss != 0.0:
            loss = loss + self.moe_loss_coef * moe_loss

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "moe_loss": moe_loss.data if hasattr(moe_loss, 'data') else moe_loss,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        LabelSmoothedCrossEntropyCriterion.reduce_metrics(logging_outputs)
        moe_loss_sum = sum(log.get("moe_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        if ntokens > 0:
            from fairseq.logging import metrics
            metrics.log_scalar("moe_loss", moe_loss_sum / ntokens, ntokens, round=3)