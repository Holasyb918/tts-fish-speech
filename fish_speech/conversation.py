from dataclasses import dataclass, field
from typing import Literal

import torch

from .tokenizer import MODALITY_TOKENS, FishTokenizer

CODEBOOK_PAD_TOKEN_ID = 0

def keyword_only_dataclass0(kw_only_fields):
    """
    一个装饰器，用于在 Python 3.9 及更早版本中模拟 dataclass 的 kw_only 行为。

    Args:
        kw_only_fields: 一个包含需要强制为关键字参数的字段名称的列表或元组。
    """
    def wrapper(cls):
        # 使用标准的 dataclass 装饰器来创建基本的 dataclass 功能
        cls = dataclass(cls)
        original_init = cls.__init__

        def __init__(self, *args, **kwargs):
            # 获取 dataclass 的所有字段
            all_fields = {field.name for field in fields(cls)}
            # 获取非 keyword-only 的字段（即不在 kw_only_fields 中的字段）
            positional_fields = [field for field in all_fields if field not in kw_only_fields]
            num_positional_fields = len(positional_fields)

            # 检查是否有多余的位置参数被传递给 keyword-only 字段
            if len(args) > num_positional_fields:
                unexpected_positional = list(all_fields - set(positional_fields))[len(args) - num_positional_fields - (len(args) - num_positional_fields):]
                raise TypeError(
                    f"__init__() got unexpected positional arguments for keyword-only fields: {', '.join(unexpected_positional)}"
                )

            # 调用原始的 __init__ 方法
            original_init(self, *args, **kwargs)

            # 检查 keyword-only 字段是否通过关键字参数传递
            for field_name in kw_only_fields:
                if field_name not in kwargs:
                    raise TypeError(
                        f"__init__() missing required keyword-only argument: '{field_name}'"
                    )

        cls.__init__ = __init__
        return cls
    return wrapper

def kw_only_dataclass(*, kw_only=False):
    """
    一个装饰器，用于在 Python 3.6 及更早版本中模拟 dataclass 的 kw_only=True 行为。

    Args:
        kw_only: 一个布尔值，如果为 True，则强制所有字段为关键字参数。
    """
    def wrapper(cls):
        cls = dataclass(cls)  # 先使用标准的 dataclass 装饰器

        if kw_only:
            original_init = cls.__init__

            def __init__(self, *args, **kwargs):
                if args:
                    raise TypeError(
                        f"{cls.__name__}.__init__() takes 0 positional arguments but {len(args)} were given"
                    )
                original_init(self, **kwargs)

            cls.__init__ = __init__
        return cls
    return wrapper
# dataclass = kw_only_dataclass

@kw_only_dataclass(kw_only=True)
class BasePart:
    pass


@kw_only_dataclass(kw_only=True)
class VQPart(BasePart):
    codes: torch.Tensor


@kw_only_dataclass(kw_only=True)
class TextPart(BasePart):
    text: str


@kw_only_dataclass(kw_only=True)
class EncodedMessage:
    tokens: torch.Tensor
    labels: torch.Tensor
    vq_mask_tokens: torch.Tensor = None
    vq_mask_labels: torch.Tensor = None
    vq_parts: list = None
    vq_require_losses: torch.Tensor = None


@kw_only_dataclass(kw_only=True)
class Message:
    role: Literal["system", "user", "assistant"]
    parts: list = field(default_factory=list)
    # parts: list[VQPart | TextPart] = field(default_factory=list)
    add_im_start: bool = True
    add_im_end: bool = True
    cal_loss: bool = False
    modality: Literal["text", "voice", "interleave"] = None

    # By default, ignore the loss of the auto-generated im_start token
    ignore_im_start_loss: bool = True

    def encode(
        self: "Message",
        tokenizer: FishTokenizer,
    ) -> EncodedMessage:
        all_tokens = []
        all_labels = []

        # Multi-modal tokens
        vq_parts = []
        vq_masks = []

        parts = self.parts.copy()
        if self.add_im_start:
            modality_token = MODALITY_TOKENS[self.modality] if self.modality else ""
            parts.insert(0, TextPart(text=f"<|im_start|>{self.role}\n{modality_token}"))

        if self.add_im_end:
            parts.append(TextPart(text="<|im_end|>"))

        for part in parts:
            if isinstance(part, TextPart):
                tokens = torch.tensor(
                    tokenizer.encode(part.text),
                    dtype=torch.int,
                )
            elif isinstance(part, VQPart):
                curr_codes = part.codes.clone()
                tokens = torch.tensor(
                    [
                        tokenizer.semantic_id_to_token_id[i.item()]
                        for i in curr_codes[0].int()
                    ],
                    dtype=torch.int,
                )
                vq_parts.append(curr_codes)
            else:
                raise ValueError(f"Unsupported part type: {type(part)}")

            all_tokens.append(tokens)
            if isinstance(part, VQPart):
                vq_masks.append(torch.ones_like(tokens, dtype=torch.bool))
            else:
                vq_masks.append(torch.zeros_like(tokens, dtype=torch.bool))

            if self.cal_loss:
                all_labels.append(tokens.clone())
            else:
                all_labels.append(torch.full_like(tokens, -100))

        tokens = torch.cat(all_tokens, dim=0)
        labels = torch.cat(all_labels, dim=0)
        vq_masks = torch.cat(vq_masks, dim=0)

        assert tokens.shape == labels.shape == vq_masks.shape

        if self.ignore_im_start_loss and self.add_im_start:
            labels[: len(all_tokens[0])] = -100

        return EncodedMessage(
            tokens=tokens,
            labels=labels,
            vq_parts=vq_parts,
            vq_mask_tokens=vq_masks,
            vq_mask_labels=vq_masks,
        )


@dataclass
class Conversation:
    messages: list

    def __init__(self: "Conversation", messages: list = None):
        self.messages = messages or []

    def encode(
        self: "Conversation",
        tokenizer: FishTokenizer,
        add_shift: bool = True,
        ignore_loss_tokens: list = [],
    ) -> EncodedMessage:
        # Build the input_ids and labels
        tokens = []
        labels = []
        vq_parts = []
        vq_mask_tokens = []
        vq_mask_labels = []
        vq_require_losses = []
        ignore_loss_token_ids = [tokenizer.get_token_id(i) for i in ignore_loss_tokens]

        for message in self.messages:
            encoded = message.encode(
                tokenizer,
            )
            tokens.append(encoded.tokens)
            labels.append(encoded.labels)
            vq_parts.extend(encoded.vq_parts)
            vq_mask_tokens.append(encoded.vq_mask_tokens)
            vq_mask_labels.append(encoded.vq_mask_labels)
            vq_require_losses.extend([message.cal_loss] * len(encoded.vq_parts))

        tokens = torch.cat(tokens, dim=0)
        labels = torch.cat(labels, dim=0)
        vq_mask_tokens = torch.cat(vq_mask_tokens, dim=0)
        vq_mask_labels = torch.cat(vq_mask_labels, dim=0)
        vq_require_losses = torch.tensor(vq_require_losses, dtype=torch.bool)

        if add_shift:
            tokens = tokens[:-1]
            labels = labels[1:]
            vq_mask_tokens = vq_mask_tokens[:-1]
            vq_mask_labels = vq_mask_labels[1:]

        for i in ignore_loss_token_ids:
            assert i != -100 and i is not None
            labels[labels == i] = -100

        assert tokens.dtype in [
            torch.int,
            torch.long,
        ], f"Invalid dtype: {tokens.dtype}, conv: {conversation}"

        return EncodedMessage(
            tokens=tokens,
            labels=labels,
            vq_parts=vq_parts,
            vq_mask_tokens=vq_mask_tokens,
            vq_mask_labels=vq_mask_labels,
            vq_require_losses=vq_require_losses,
        )

    def encode_for_inference(
        self: "Conversation",
        tokenizer: FishTokenizer,
        num_codebooks: int,
    ) -> EncodedMessage:
        # self.visualize(tokenizer)

        encoded = self.encode(tokenizer, add_shift=False)
        tokens = encoded.tokens
        values = torch.zeros((num_codebooks + 1, len(tokens)), dtype=torch.int)
        values[0] = tokens

        if encoded.vq_parts is None or len(encoded.vq_parts) == 0:
            return values

        vq_parts = encoded.vq_parts
        vq_parts = [part.to(values.device) for part in vq_parts]
        vq_parts = torch.cat(vq_parts, dim=1)
        values[0, encoded.vq_mask_tokens] = vq_parts[0] + tokenizer.semantic_begin_id
        values[1:, encoded.vq_mask_tokens] = vq_parts

        return values

    def visualize(
        self: "Conversation",
        tokenizer: FishTokenizer,
        ignore_loss_tokens: list = [],
    ):
        encoded = self.encode(
            tokenizer, add_shift=False, ignore_loss_tokens=ignore_loss_tokens
        )

        colors = {
            "purple": "\033[95m",
            "yellow": "\033[93m",
            "red": "\033[91m",
            "cyan": "\033[96m",
        }
        first_idx = 0
        second_idx = 0

        def print_first_group(x):
            nonlocal first_idx
            color = colors["purple"] if first_idx % 2 == 0 else colors["yellow"]
            print(f"{color}{x}\033[0m", end="")
            first_idx += 1

        def print_second_group(x):
            nonlocal second_idx
            color = colors["red"] if second_idx % 2 == 0 else colors["cyan"]
            print(f"{color}{x}\033[0m", end="")
            second_idx += 1

        for tok, lab in zip(encoded.tokens, encoded.labels):
            val = tokenizer.decode([tok])

            if lab == -100:
                print_second_group(val)
            else:
                print_first_group(val)

        print()

    def append(self: "Conversation", message: Message):
        self.messages.append(message)


if __name__ == "__main__":
    message0 = Message(
        role="user",
        parts=[
            TextPart(text="Hello, how are you?"),
            VQPart(codes=torch.zeros((4, 10))),
        ],
        cal_loss=False,
    )

    message1 = Message(
        role="assistant",
        parts=[TextPart(text="I'm fine, thank you.")],
        cal_loss=True,
    )
    conversation = Conversation([message0, message1])
    tokenizer = FishTokenizer.from_pretrained("checkpoints/Qwen2-1.5B-Instruct")
    conversation.visualize(tokenizer)

    encoded = conversation.encode(tokenizer)
    print(encoded)
    print(tokenizer.batch_decode(encoded.tokens))
