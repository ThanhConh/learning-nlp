import numpy as np

def softmax_stable(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)

    exp_x = np.exp(x - x_max)

    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_atttention(Q, K, V):
    d_K = Q.shape[-1]

    scores = Q @ K.T

    scores = scores / np.sqrt(d_k)

    attention_weights = softmax_stable(scores, axis=-1)

    context = attention_weights @ V

    return scores, attention_weights, context

if __name__ == "__main__":
    x = np.array([1000.0, 1001.0, 1002.0])

# Stable softmax
    stable_result = softmax_stable(x)

    # Tính theo cách tương đương nhưng với số nhỏ hơn
    x_shifted = x - np.max(x)
    manual_result = (
        np.exp(x_shifted) /
        np.sum(np.exp(x_shifted))
    )

    print("Stable softmax:")
    print(stable_result)

    print("\nManual calculation:")
    print(manual_result)

    print("\nHai kết quả có bằng nhau không?")
    print(np.allclose(stable_result, manual_result))