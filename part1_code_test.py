import os
from collections import defaultdict
from part1_code import estimate_emission_parameters, simple_sentiment_analysis

def test_estimate_emission_parameters():
    # Test case 1
    train_path = "Data/test/train1.txt"
    expected_output = defaultdict(lambda: defaultdict(float), {
        "I": defaultdict(float, {"N": 0.5, "P": 0.25, "O": 0.25}),
        "love": defaultdict(float, {"P": 1.0}),
        "you": defaultdict(float, {"P": 0.5, "N": 0.5}),
        "#UNK#": defaultdict(float, {"P": 0.25, "N": 0.25, "O": 0.25})
    })
    assert estimate_emission_parameters(train_path, k=1) == expected_output

    # Test case 2
    train_path = "Data/test/train2.txt"
    expected_output = defaultdict(lambda: defaultdict(float), {
        "I": defaultdict(float, {"N": 0.5, "P": 0.25, "O": 0.25}),
        "love": defaultdict(float, {"P": 1.0}),
        "you": defaultdict(float, {"P": 0.5, "N": 0.5}),
        "#UNK#": defaultdict(float, {"P": 0.25, "N": 0.25, "O": 0.25})
    })
    assert estimate_emission_parameters(train_path, k=1) == expected_output

def test_simple_sentiment_analysis():
    # Test case 1
    emission_parameters = defaultdict(lambda: defaultdict(float), {
        "I": defaultdict(float, {"N": 0.5, "P": 0.25, "O": 0.25}),
        "love": defaultdict(float, {"P": 1.0}),
        "you": defaultdict(float, {"P": 0.5, "N": 0.5}),
        "#UNK#": defaultdict(float, {"P": 0.25, "N": 0.25, "O": 0.25})
    })
    dev_in_path = "Data/test/dev1.in"
    dev_out_path = "Data/test/dev1.out"
    expected_output = [
        "I P",
        "love P",
        "you P",
        "I P",
        "hate N",
        "you N",
        "I O",
        "love O",
        "you O",
        ""
    ]
    simple_sentiment_analysis(emission_parameters, dev_in_path, dev_out_path)
    with open(dev_out_path, 'r', encoding='utf-8') as file:
        assert file.read().strip().split('\n') == expected_output

    # Test case 2
    emission_parameters = defaultdict(lambda: defaultdict(float), {
        "I": defaultdict(float, {"N": 0.5, "P": 0.25, "O": 0.25}),
        "love": defaultdict(float, {"P": 1.0}),
        "you": defaultdict(float, {"P": 0.5, "N": 0.5}),
        "#UNK#": defaultdict(float, {"P": 0.25, "N": 0.25, "O": 0.25})
    })
    dev_in_path = "Data/test/dev2.in"
    dev_out_path = "Data/test/dev2.out"
    expected_output = [
        "I P",
        "love P",
        "you P",
        "I P",
        "hate N",
        "you N",
        "I O",
        "love O",
        "you O",
        ""
    ]
    simple_sentiment_analysis(emission_parameters, dev_in_path, dev_out_path)
    with open(dev_out_path, 'r', encoding='utf-8') as file:
        assert file.read().strip().split('\n') == expected_output

test_estimate_emission_parameters()
test_simple_sentiment_analysis()