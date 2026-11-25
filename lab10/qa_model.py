from typing import Any, Dict, List, Optional, Sequence, Tuple

from transformers import AutoTokenizer


MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 384
STRIDE = 128


def load_tokenizer(model_name: str):
    """Load fast tokenizer vì cần sử dụng offset_mapping."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True
    )

    if not tokenizer.is_fast:
        raise ValueError(
            "Tokenizer phải là Fast Tokenizer để hỗ trợ offset_mapping."
        )

    return tokenizer


def tokenize_qa_examples(
    tokenizer,
    questions: Sequence[str],
    contexts: Sequence[str],
    max_length: int = MAX_LENGTH,
    stride: int = STRIDE,
) -> Dict[str, Any]:
    """
    Tokenize các cặp question-context.

    Context dài có thể được chia thành nhiều feature (sliding windows).
    overflow_to_sample_mapping cho biết mỗi feature thuộc sample gốc nào.
    """
    if len(questions) != len(contexts):
        raise ValueError(
            "Số lượng questions và contexts phải bằng nhau."
        )

    if max_length <= 0:
        raise ValueError("max_length phải lớn hơn 0.")

    if stride < 0:
        raise ValueError("stride không được âm.")

    encoding = tokenizer(
        list(questions),
        list(contexts),
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )

    return encoding


def print_feature_details(
    tokenizer,
    encoding,
    questions: Sequence[str],
    contexts: Sequence[str],
    feature_idx: int,
) -> None:
    """In chi tiết các trường của một feature."""
    input_ids = encoding["input_ids"][feature_idx]
    attention_mask = encoding["attention_mask"][feature_idx]
    offsets = encoding["offset_mapping"][feature_idx]
    sample_idx = encoding["overflow_to_sample_mapping"][feature_idx]
    sequence_ids = encoding.sequence_ids(feature_idx)

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    print("\n" + "=" * 90)
    print(f"FEATURE {feature_idx}")
    print("=" * 90)

    print(f"Source sample index: {sample_idx}")
    print(f"Question: {questions[sample_idx]}")
    print(f"Context length: {len(contexts[sample_idx])} characters")
    print(f"Number of tokens: {len(input_ids)}")

    print("\ninput_ids:")
    print(input_ids)

    print("\nattention_mask:")
    print(attention_mask)

    print("\noffset_mapping:")
    print(offsets)

    print("\noverflow_to_sample_mapping:")
    print(encoding["overflow_to_sample_mapping"])

    print("\nToken-level inspection:")
    print(
        f"{'Idx':>4} | {'Token':<18} | {'Input ID':>8} | "
        f"{'Attention':>9} | {'Offset':<14} | {'Sequence':>8}"
    )
    print("-" * 85)

    for idx, (token, token_id, mask, offset, seq_id) in enumerate(
        zip(tokens, input_ids, attention_mask, offsets, sequence_ids)
    ):
        print(
            f"{idx:4d} | {token:<18} | {token_id:8d} | "
            f"{mask:9d} | {str(tuple(offset)):<14} | {str(seq_id):>8}"
        )


def align_answer_span(
    offsets: Sequence[Tuple[int, int]],
    sequence_ids: Sequence[Optional[int]],
    answer_start: int,
    answer_end: int,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Ánh xạ character span [answer_start, answer_end)
    sang token span [start_token, end_token].

    Chỉ xét token thuộc context (sequence_id == 1).
    """
    start_token = None
    end_token = None

    for token_idx, (start, end) in enumerate(offsets):
        if sequence_ids[token_idx] != 1:
            continue

        # Special token thường có offset (0, 0).
        if start == end:
            continue

        if start <= answer_start < end:
            start_token = token_idx

        if start < answer_end <= end:
            end_token = token_idx

    if start_token is None or end_token is None:
        return None, None

    if start_token > end_token:
        return None, None

    return start_token, end_token


def inspect_answer_alignment(
    tokenizer,
    encoding,
    sample_idx: int,
    answer_text: str,
    answer_start: int,
) -> None:
    """
    Kiểm tra đáp án trên tất cả feature thuộc sample_idx.

    Nếu đáp án nằm ngoài một window, alignment trả về None.
    Một sample có thể có nhiều window, nhưng chỉ một hoặc một vài
    window chứa đầy đủ đáp án.
    """
    answer_end = answer_start + len(answer_text)
    mapping = encoding["overflow_to_sample_mapping"]

    print("\n" + "=" * 90)
    print("ANSWER SPAN ALIGNMENT")
    print("=" * 90)
    print(f"Answer: {answer_text!r}")
    print(f"Character span: [{answer_start}, {answer_end})")

    found = False

    for feature_idx, source_idx in enumerate(mapping):
        if source_idx != sample_idx:
            continue

        offsets = encoding["offset_mapping"][feature_idx]
        sequence_ids = encoding.sequence_ids(feature_idx)

        start_token, end_token = align_answer_span(
            offsets=offsets,
            sequence_ids=sequence_ids,
            answer_start=answer_start,
            answer_end=answer_end,
        )

        print(f"\nFeature {feature_idx}:")

        if start_token is None:
            print("  Answer is outside this window or truncated.")
            continue

        found = True
        input_ids = encoding["input_ids"][feature_idx]

        start_text = tokenizer.convert_ids_to_tokens(
            input_ids[start_token]
        )
        end_text = tokenizer.convert_ids_to_tokens(
            input_ids[end_token]
        )

        print(f"  start_token: {start_token} ({start_text})")
        print(f"  end_token:   {end_token} ({end_text})")

    if not found:
        print(
            "\nKhông tìm thấy feature nào chứa đầy đủ đáp án. "
            "Hãy kiểm tra answer_start, answer_text và các window."
        )


def main() -> None:
    tokenizer = load_tokenizer(MODEL_NAME)

    # Ví dụ 1: Context đủ ngắn, thường chỉ tạo một feature.
    question_1 = "What is the capital of France?"
    context_1 = (
        "Paris is the capital of France. "
        "It is one of the most famous cities in Europe."
    )

    # Ví dụ 2: Context dài để minh họa overflow/sliding window.
    question_2 = "What is Paris known for?"
    context_2 = (
        "Paris is the capital of France. "
        "France is located in Western Europe. "
        "The country has many famous cities. "
        "The population of Paris is very large. "
        "Paris has many museums, historical buildings, "
        "restaurants, parks, and cultural attractions. "
        "Millions of tourists visit the city every year. "
        "One of its most recognizable landmarks is the Eiffel Tower. "
        "The Eiffel Tower is located in Paris and attracts visitors "
        "from many different countries. "
    ) * 12

    questions = [question_1, question_2]
    contexts = [context_1, context_2]

    encoding = tokenize_qa_examples(
        tokenizer=tokenizer,
        questions=questions,
        contexts=contexts,
        max_length=MAX_LENGTH,
        stride=STRIDE,
    )

    print("MODEL:", MODEL_NAME)
    print("MAX_LENGTH:", MAX_LENGTH)
    print("STRIDE:", STRIDE)
    print("Number of original samples:", len(questions))
    print("Number of generated features:", len(encoding["input_ids"]))

    print("\nFeature-to-sample mapping:")
    for feature_idx, sample_idx in enumerate(
        encoding["overflow_to_sample_mapping"]
    ):
        print(
            f"Feature {feature_idx} -> Original sample {sample_idx}"
        )

    # In chi tiết feature đầu tiên và một feature overflow nếu có.
    feature_indices = [0]

    if len(encoding["input_ids"]) > 1:
        feature_indices.append(len(encoding["input_ids"]) - 1)

    for feature_idx in feature_indices:
        print_feature_details(
            tokenizer=tokenizer,
            encoding=encoding,
            questions=questions,
            contexts=contexts,
            feature_idx=feature_idx,
        )

    # Kiểm tra alignment đáp án của sample 0.
    answer_text = "France"
    answer_start = context_1.find(answer_text)

    inspect_answer_alignment(
        tokenizer=tokenizer,
        encoding=encoding,
        sample_idx=0,
        answer_text=answer_text,
        answer_start=answer_start,
    )

    # Kiểm tra alignment đáp án ở cuối context dài của sample 1.
    answer_text_2 = "Eiffel Tower"
    answer_start_2 = context_2.rfind(answer_text_2)

    inspect_answer_alignment(
        tokenizer=tokenizer,
        encoding=encoding,
        sample_idx=1,
        answer_text=answer_text_2,
        answer_start=answer_start_2,
    )


if __name__ == "__main__":
    main()
