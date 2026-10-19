from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput
from python_basics import (count_vowels, has_unique_characters, count_one_bits, multiplicative_persistence, mse, prime_factorization, pyramid,is_balanced_number,)


def test_count_vowels():
    assert count_vowels(TextInput(value="hello")) == 2
    assert count_vowels(TextInput(value="AEIOU")) == 5
    assert count_vowels(TextInput(value="xyz")) == 0
    assert count_vowels(TextInput(value="")) == 0


def test_has_unique_characters():
    assert has_unique_characters(TextInput(value="abc")) is True
    assert has_unique_characters(TextInput(value="aa")) is False
    assert has_unique_characters(TextInput(value="")) is True
    assert has_unique_characters(TextInput(value="a b c")) is True


def test_count_one_bits():
    assert count_one_bits(PositiveIntegerInput(value=1)) == 1
    assert count_one_bits(PositiveIntegerInput(value=7)) == 3
    assert count_one_bits(PositiveIntegerInput(value=8)) == 1
    assert count_one_bits(PositiveIntegerInput(value=255)) == 8


def test_multiplicative_persistence():
    assert multiplicative_persistence(PositiveIntegerInput(value=39)) == 3
    assert multiplicative_persistence(PositiveIntegerInput(value=4)) == 0
    assert multiplicative_persistence(PositiveIntegerInput(value=999)) == 4
    assert multiplicative_persistence(PositiveIntegerInput(value=10)) == 1


def test_mse():
    assert mse(VectorPairInput(predicted=[1, 2, 3], expected=[1, 2, 3])) == 0.0
    assert mse(VectorPairInput(predicted=[1, 2], expected=[3, 4])) == 4.0
    assert mse(VectorPairInput(predicted=[0], expected=[2])) == 4.0


def test_prime_factorization():
    assert prime_factorization(PositiveIntegerInput(value=86240)) == "(2**5)(5)(7**2)(11)"
    assert prime_factorization(PositiveIntegerInput(value=7)) == "(7)"
    assert prime_factorization(PositiveIntegerInput(value=12)) == "(2**2)(3)"
    assert prime_factorization(PositiveIntegerInput(value=2)) == "(2)"


def test_pyramid():
    assert pyramid(PositiveIntegerInput(value=1)) == 1
    assert pyramid(PositiveIntegerInput(value=5)) == 2      
    assert pyramid(PositiveIntegerInput(value=14)) == 3     
    assert pyramid(PositiveIntegerInput(value=2)) == "It is impossible"
    assert pyramid(PositiveIntegerInput(value=30)) == 4     


def test_is_balanced_number():
    assert is_balanced_number(PositiveIntegerInput(value=1234006)) is True
    assert is_balanced_number(PositiveIntegerInput(value=123456)) is False
    assert is_balanced_number(PositiveIntegerInput(value=1)) is True       
    assert is_balanced_number(PositiveIntegerInput(value=22)) is True      
    assert is_balanced_number(PositiveIntegerInput(value=1234)) is False   