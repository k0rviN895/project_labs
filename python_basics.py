from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput

def count_vowels(data: TextInput) -> int:
    text = data.value
    vowels = set("aeiouAEIOU")
    return sum(1 for ch in text if ch in vowels)


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(text) == len(set(text))


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    count = 0
    while number:
        count += number & 1
        number >>= 1
    return count


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        product = 1
        while number:
            product *= number % 10
            number //= 10
        number = product
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    n = len(predicted)
    total = 0.0
    for p, e in zip(predicted, expected):
        diff = p - e
        total += diff * diff
    return total / n


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    parts = []
    divisor = 2
    n = number
    while divisor * divisor <= n:
        if n % divisor == 0:
            power = 0
            while n % divisor == 0:
                n //= divisor
                power += 1
            if power == 1:
                parts.append(f"({divisor})")
            else:
                parts.append(f"({divisor}**{power})")
        divisor += 1
    if n > 1:
        parts.append(f"({n})")
    return "".join(parts)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    k = 0
    total = 0
    while total < cube_count:
        k += 1
        total += k * k
    if total == cube_count:
        return k
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    digits = [int(ch) for ch in str(number)]
    length = len(digits)
    half = length // 2
    left_sum = sum(digits[:half])
    if length % 2 == 1:
        right_sum = sum(digits[half + 1:])
    else:
        right_sum = sum(digits[half:])
    return left_sum == right_sum