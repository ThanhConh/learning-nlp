import argparse
import os
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)


@dataclass
class AnswerResult:
    """Stores the best answer extracted from the document."""

    answer: str
    score: float
    start_char: Optional[int]
    end_char: Optional[int]
    feature_index: Optional[int]


class DocumentQASystem:
    """
    A small extractive QA system.

    The user supplies both the question and the document.
    The model extracts an answer span from that document.
    """

    def __init__(
        self,
        model_name_or_path: str = "./qa_experiment_2",
        max_length: int = 384,
        stride: int = 128,
        max_answer_length: int = 30,
        device: Optional[str] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.stride = stride
        self.max_answer_length = max_answer_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        if not os.path.exists(model_name_or_path):
            raise FileNotFoundError(
                f"Không tìm thấy model: {model_name_or_path}\n"
                "Hãy truyền đường dẫn đến checkpoint đã fine-tune, "
                "ví dụ: --model ./qa_experiment_2"
            )

        print(f"Loading tokenizer: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
        )

        if not self.tokenizer.is_fast:
            raise ValueError(
                "Document QA cần Fast Tokenizer để sử dụng offset_mapping."
            )

        print(f"Loading QA model: {model_name_or_path}")
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_name_or_path
        )

        self.model.to(self.device)
        self.model.eval()

        print(f"Device: {self.device}")

    def _tokenize_document(self, question: str, document: str):
        """
        Tokenize question-document pair.

        A long document is divided into overlapping features.
        Each feature contains a window of the original document.
        """
        if self.stride >= self.max_length:
            raise ValueError(
                "stride phải nhỏ hơn max_length."
            )

        encoding = self.tokenizer(
            question,
            document,
            truncation="only_second",
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
        )

        return encoding

    def _predict_feature(
        self,
        input_ids: List[int],
        attention_mask: List[int],
    ):
        """Run the QA model on one feature/window."""
        input_tensor = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=self.device,
        )

        mask_tensor = torch.tensor(
            [attention_mask],
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_tensor,
                attention_mask=mask_tensor,
            )

        start_logits = outputs.start_logits[0].cpu()
        end_logits = outputs.end_logits[0].cpu()

        return start_logits, end_logits

    def _find_best_span(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        offsets,
        sequence_ids,
    ):
        """
        Find the best valid answer span in one feature.

        Only context tokens are considered.
        Special tokens and question tokens are ignored.
        """
        context_token_indices = [
            idx
            for idx, (offset, sequence_id) in enumerate(
                zip(offsets, sequence_ids)
            )
            if sequence_id == 1 and offset[0] < offset[1]
        ]

        if not context_token_indices:
            return None

        best_candidate = None
        best_score = float("-inf")

        for start_idx in context_token_indices:
            max_end_idx = min(
                start_idx + self.max_answer_length,
                len(offsets) - 1,
            )

            for end_idx in range(start_idx, max_end_idx + 1):
                # End token must also belong to the context.
                if sequence_ids[end_idx] != 1:
                    continue

                start_char = offsets[start_idx][0]
                end_char = offsets[end_idx][1]

                if end_char <= start_char:
                    continue

                score = (
                    start_logits[start_idx].item()
                    + end_logits[end_idx].item()
                )

                if score > best_score:
                    best_score = score
                    best_candidate = {
                        "start_token": start_idx,
                        "end_token": end_idx,
                        "start_char": start_char,
                        "end_char": end_char,
                        "score": score,
                    }

        return best_candidate

    def answer(
        self,
        question: str,
        document: str,
    ) -> AnswerResult:
        """
        Answer a question using the supplied document.

        The answer is extracted directly from the original document
        using character offsets returned by the tokenizer.
        """
        question = question.strip()
        document = document.strip()

        if not question:
            raise ValueError("Question không được để trống.")

        if not document:
            raise ValueError("Document không được để trống.")

        encoding = self._tokenize_document(
            question,
            document,
        )

        best_result = None
        best_score = float("-inf")

        num_features = len(encoding["input_ids"])

        for feature_idx in range(num_features):
            input_ids = encoding["input_ids"][feature_idx]
            attention_mask = encoding["attention_mask"][feature_idx]
            offsets = encoding["offset_mapping"][feature_idx]
            sequence_ids = encoding.sequence_ids(feature_idx)

            start_logits, end_logits = self._predict_feature(
                input_ids,
                attention_mask,
            )

            candidate = self._find_best_span(
                start_logits=start_logits,
                end_logits=end_logits,
                offsets=offsets,
                sequence_ids=sequence_ids,
            )

            if candidate is None:
                continue

            if candidate["score"] > best_score:
                best_score = candidate["score"]
                best_result = {
                    **candidate,
                    "feature_index": feature_idx,
                }

        if best_result is None:
            return AnswerResult(
                answer="",
                score=float("-inf"),
                start_char=None,
                end_char=None,
                feature_index=None,
            )

        start_char = best_result["start_char"]
        end_char = best_result["end_char"]

        # Extract the answer from the original document.
        answer_text = document[start_char:end_char].strip()

        return AnswerResult(
            answer=answer_text,
            score=best_result["score"],
            start_char=start_char,
            end_char=end_char,
            feature_index=best_result["feature_index"],
        )


def read_document_file(file_path: str) -> str:
    """Read a UTF-8 text document."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def print_result(result: AnswerResult) -> None:
    """Display the extracted answer and its source span."""
    print("\n" + "=" * 65)
    print("DOCUMENT QA RESULT")
    print("=" * 65)

    if not result.answer:
        print("Answer: No answer span found.")
        return

    print("Answer:", result.answer)
    print("Start character:", result.start_char)
    print("End character:", result.end_char)
    print("Feature index:", result.feature_index)
    print("Span score:", round(result.score, 4))

    print("=" * 65)


def interactive_mode(qa_system: DocumentQASystem) -> None:
    """Run a simple interactive question-answering loop."""
    print("\nDOCUMENT QA — INTERACTIVE MODE")
    print("Nhập 'exit' tại câu hỏi để kết thúc.\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Kết thúc Document QA.")
            break

        print(
            "Nhập document. Nhấn Enter trên một dòng trống "
            "để kết thúc document:"
        )

        document_lines = []

        while True:
            line = input()

            if not line.strip():
                break

            document_lines.append(line)

        document = "\n".join(document_lines)

        try:
            result = qa_system.answer(
                question=question,
                document=document,
            )
            print_result(result)
        except ValueError as error:
            print("Input error:", error)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mini Extractive Document QA system."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="./qa_experiment_2",
        help="Đường dẫn đến checkpoint QA đã fine-tune.",
    )

    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Câu hỏi cần trả lời.",
    )

    parser.add_argument(
        "--document",
        type=str,
        default=None,
        help="Nội dung document/context.",
    )

    parser.add_argument(
        "--document-file",
        type=str,
        default=None,
        help="Đường dẫn đến file văn bản UTF-8.",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=384,
        help="Độ dài tối đa của mỗi feature.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=128,
        help="Số token overlap giữa hai window.",
    )

    parser.add_argument(
        "--max-answer-length",
        type=int,
        default=30,
        help="Số token tối đa của một answer span.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    qa_system = DocumentQASystem(
        model_name_or_path=args.model,
        max_length=args.max_length,
        stride=args.stride,
        max_answer_length=args.max_answer_length,
    )

    # Nếu không truyền question/document thì chạy interactive mode.
    if args.question is None:
        interactive_mode(qa_system)
        return

    if args.document is not None and args.document_file is not None:
        raise ValueError(
            "Chỉ sử dụng một trong hai: --document hoặc --document-file."
        )

    if args.document_file is not None:
        document = read_document_file(args.document_file)
    elif args.document is not None:
        document = args.document
    else:
        raise ValueError(
            "Cần cung cấp --document hoặc --document-file."
        )

    result = qa_system.answer(
        question=args.question,
        document=document,
    )

    print_result(result)


if __name__ == "__main__":
    main()
