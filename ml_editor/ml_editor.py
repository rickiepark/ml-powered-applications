import argparse
import logging
import sys

import pyphen
import nltk

pyphen.language_fallback("en_US")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
console_out = logging.StreamHandler(sys.stdout)
console_out.setLevel(logging.DEBUG)
logger.addHandler(console_out)


def parse_arguments():
    """
    간단한 명령줄 매개변수 파서
    :return: 수정할 텍스트
    """
    parser = argparse.ArgumentParser(description="Receive text to be edited")
    parser.add_argument("text", metavar="input text", type=str)
    args = parser.parse_args()
    return args.text


def clean_input(text):
    """
    텍스트 정제 함수
    :param text: 사용자가 입력한 텍스트
    :return: ASCII 이외의 문자를 제거한 정제된 텍스트
    """
    # 간단하게 시작하기 위해서 ASCII 문자만 사용합니다
    return str(text.encode().decode("ascii", errors="ignore"))


def preprocess_input(text):
    """
    정제된 텍스트를 토큰화합니다
    :param text: 정제된 텍스트
    :return: 문장과 단어로 토큰화하여 분석에 투입할 준비를 마친 텍스트
    """
    sentences = nltk.sent_tokenize(text)
    tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    return tokens


def compute_flesch_reading_ease(total_syllables, total_words, total_sentences):
    """
    Computes readability score from summary statistics
    :param total_syllables: number of syllables in input text
    :param total_words: number of words in input text
    :param total_sentences: number of sentences in input text
    :return: A readability score: the lower the score, the more complex the text is deemed to be
    """
    return (
        206.85
        - 1.015 * (total_words / total_sentences)
        - 84.6 * (total_syllables / total_words)
    )


def get_reading_level_from_flesch(flesch_score):
    """
    Thresholds taken from https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
    :param flesch_score:
    :return: A reading level and difficulty for a given flesch score
    """
    if flesch_score < 30:
        return "매우 읽기 어려움"
    elif flesch_score < 50:
        return "읽기 어려움"
    elif flesch_score < 60:
        return "약간 읽기 어려움"
    elif flesch_score < 70:
        return "보통"
    elif flesch_score < 80:
        return "약간 읽기 쉬움"
    elif flesch_score < 90:
        return "읽기 쉬움"
    else:
        return "매우 읽기 쉬움"


def compute_average_word_length(tokens):
    """
    Calculate word length for a sentence
    :param tokens: a list of words
    :return: The average length of words in this list
    """
    word_lengths = [len(word) for word in tokens]
    return sum(word_lengths) / len(word_lengths)


def compute_total_average_word_length(sentence_list):
    """
    Calculate average word length for multiple sentences
    :param sentence_list: a list of sentences, each being a list of words
    :return: The average length of words in this list of sentences
    """
    lengths = [compute_average_word_length(tokens) for tokens in sentence_list]
    return sum(lengths) / len(lengths)


def compute_total_unique_words_fraction(sentence_list):
    """
    Compute fraction os unique words
    :param sentence_list: a list of sentences, each being a list of words
    :return: the fraction of unique words in the sentences
    """
    all_words = [word for word_list in sentence_list for word in word_list]
    unique_words = set(all_words)
    return len(unique_words) / len(all_words)


def count_word_usage(tokens, word_list):
    """
    Counts occurrences of a given list of words
    :param tokens: a list of tokens for one sentence
    :param word_list: a list of words to search for
    :return: the number of times the words appear in the list
    """
    return len([word for word in tokens if word.lower() in word_list])


def count_word_syllables(word):
    """
    Count syllables in a word
    :param word: a one word string
    :return: the number of syllables according to pyphen
    """
    dic = pyphen.Pyphen(lang="en_US")
    # this returns our word, with hyphens ("-") inserted in between each syllable
    hyphenated = dic.inserted(word)
    return len(hyphenated.split("-"))


def count_sentence_syllables(tokens):
    """
    Count syllables in a sentence
    :param tokens: a list of words and potentially punctuation
    :return: the number of syllables in the sentence
    """
    # Our tokenizer leaves punctuation as a separate word, so we filter for it here
    punctuation = ".,!?/"
    return sum(
        [
            count_word_syllables(word)
            for word in tokens
            if word not in punctuation
        ]
    )


def count_total_syllables(sentence_list):
    """
    Count syllables in a list of sentences
    :param sentence_list:  a list of sentences, each being a list of words
    :return: the number of syllables in the sentences
    """
    return sum(
        [count_sentence_syllables(sentence) for sentence in sentence_list]
    )


def count_words_per_sentence(sentence_tokens):
    """
    Count words in a sentence
    :param sentence_tokens: a list of words and potentially punctuation
    :return: the number of words in the sentence
    """
    punctuation = ".,!?/"
    return len([word for word in sentence_tokens if word not in punctuation])


def count_total_words(sentence_list):
    """
    Count words in a list of sentences
    :param sentence_list: a list of sentences, each being a list of words
    :return: the number of words in the sentences
    """
    return sum(
        [count_words_per_sentence(sentence) for sentence in sentence_list]
    )


def get_suggestions(sentence_list):
    """
    추천을 포함한 문자열을 반환합니다.
    :param sentence_list: 문장의 리스트. 각 문장은 단어의 리스트입니다.
    :return: 입력 텍스트를 개선하기 위한 추천
    """
    told_said_usage = sum(
        (count_word_usage(tokens, ["told", "said"]) for tokens in sentence_list)
    )
    but_and_usage = sum(
        (count_word_usage(tokens, ["but", "and"]) for tokens in sentence_list)
    )
    wh_adverbs_usage = sum(
        (
            count_word_usage(
                tokens,
                [
                    "when",
                    "where",
                    "why",
                    "whence",
                    "whereby",
                    "wherein",
                    "whereupon",
                ],
            )
            for tokens in sentence_list
        )
    )
    result_str = ""
    adverb_usage = "단어 사용량: %s told/said, %s but/and, %s wh-접속사" % (
        told_said_usage,
        but_and_usage,
        wh_adverbs_usage,
    )
    result_str += adverb_usage
    average_word_length = compute_total_average_word_length(sentence_list)
    unique_words_fraction = compute_total_unique_words_fraction(sentence_list)

    word_stats = "평균 단어 길이 %.2f, 고유한 단어의 비율 %.2f" % (
        average_word_length,
        unique_words_fraction,
    )
    # Using HTML break to later display on a webapp
    result_str += "<br/>"
    result_str += word_stats

    number_of_syllables = count_total_syllables(sentence_list)
    number_of_words = count_total_words(sentence_list)
    number_of_sentences = len(sentence_list)

    syllable_counts = "%d개 음절, %d개 단어, %d개 문장" % (
        number_of_syllables,
        number_of_words,
        number_of_sentences,
    )
    result_str += "<br/>"
    result_str += syllable_counts

    flesch_score = compute_flesch_reading_ease(
        number_of_syllables, number_of_words, number_of_sentences
    )

    flesch = "플레시 점수 %.2f: %s" % (
        flesch_score,
        get_reading_level_from_flesch(flesch_score),
    )

    result_str += "<br/>"
    result_str += flesch

    return result_str


def get_recommendations_from_input(txt):
    """
    Cleans, preprocesses, and generates heuristic suggestion for input string
    :param txt: Input text
    :return: Suggestions for a given text input
    """
    processed = clean_input(txt)
    tokenized_sentences = preprocess_input(processed)
    suggestions = get_suggestions(tokenized_sentences)
    return suggestions


if __name__ == "__main__":
    input_text = parse_arguments()
    print(get_recommendations_from_input(input_text))
