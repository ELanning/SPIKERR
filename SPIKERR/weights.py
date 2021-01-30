import torch as t
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.distributions import Bernoulli
from typing import Optional


def xavier_normal(
    input_size: int, output_size: int, positive_ratio: Optional[float] = None,
) -> t.Tensor:
    if input_size < 1:
        raise ValueError("input_size must be a positive integer.")
    if output_size < 1:
        raise ValueError("output_size must be a positive integer.")
    if positive_ratio is not None:
        if positive_ratio > 1 or positive_ratio < 0:
            raise ValueError(
                "positive_ratio must be None, or must be between zero and one."
            )

    result = t.empty((input_size, output_size))
    xavier_normal_(result)

    # TODO: Write a test for this.
    if positive_ratio is not None:
        bernoulli_distribution = Bernoulli(t.tensor([positive_ratio]))
        # squeeze to remove the last dimension that sample appends.
        mask = (
            bernoulli_distribution.sample((input_size, output_size)).squeeze(2).bool()
        )
        result.abs_()
        result = result.where(mask, -result)

    return result


def xavier_uniform(
    input_size: int, output_size: int, positive_ratio: Optional[float] = None,
) -> t.Tensor:
    if input_size < 1:
        raise ValueError("input_size must be a positive integer.")
    if output_size < 1:
        raise ValueError("output_size must be a positive integer.")
    if positive_ratio is not None:
        if positive_ratio > 1 or positive_ratio < 0:
            raise ValueError(
                "positive_ratio must be None, or must be between zero and one."
            )

    result = t.empty((input_size, output_size))
    xavier_uniform_(result)

    if positive_ratio is not None:
        bernoulli_distribution = Bernoulli(t.tensor([positive_ratio]))
        # squeeze to remove the last dimension that sample appends.
        mask = (
            bernoulli_distribution.sample((input_size, output_size)).squeeze(2).bool()
        )
        result.abs_()
        result = result.where(mask, -result)

    return result
