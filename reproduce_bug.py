class MockKiwi:
    def split_into_sents(self, text):
        # Mock sentence splitting: split by ". "
        class Sent:
            def __init__(self, t): self.text = t
        return [Sent(t) for t in text.split(". ")]

class HybridRetrieval:
    def __init__(self):
        self.kiwi = MockKiwi()

    def split_text(self, text: str, chunk_size: int, chunk_overlap: int):
        if not text:
            return []

        # 문장 분리
        try:
            sents = [sent.text for sent in self.kiwi.split_into_sents(text)]  # type: ignore
        except Exception:
            sents = text.split(". ")

        chunks = []
        current_chunk = []
        current_len = 0

        for sent in sents:
            sent_len = len(sent)

            # 예외: 문장이 chunk_size보다 클 경우 강제로 자릅니다.
            if sent_len > chunk_size:
                pass

            # current chunk 크기 초과 시 chunk에 저장 후 새로 시작
            if current_len + sent_len > chunk_size:
                chunks.append(" ".join(current_chunk))

                new_chunk = []
                new_len = 0
                for prev_sent in current_chunk[
                    ::-1
                ]:  # TODO: [::-1] 대신 deque 사용하기

                    # 예외: overlap 처리할 문장이 overlap보다 더 클 경우
                    if new_len + len(prev_sent) > chunk_overlap:
                        if not new_chunk:
                            new_chunk.append(prev_sent)
                        break

                    new_chunk.append(prev_sent)
                    new_len += len(prev_sent)

                # overlap 문장과 새로 들어오는 문장 병합(거꾸로 추가했으니 다시 거꾸로 돌려주고 더해주어야 함)
                current_chunk = new_chunk[::-1] + [sent]
                current_len = new_len + sent_len

            else:
                current_chunk.append(sent)
                current_len += sent_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

# Test
retriever = HybridRetrieval()
# Scenario: First sentence is larger than chunk_size
long_sentence = "A" * 150
text = long_sentence + ". " + "Short sentence"
chunk_size = 100
chunk_overlap = 20

print(f"Testing with chunk_size={chunk_size}")
chunks = retriever.split_text(text, chunk_size, chunk_overlap)
print(f"Chunks: {chunks}")
print(f"Number of chunks: {len(chunks)}")
if "" in chunks:
    print("BUG CONFIRMED: Empty chunk found.")
else:
    print("No empty chunks.")
