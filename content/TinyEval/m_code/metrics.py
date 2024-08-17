import re
import string
import jieba
from rouge import Rouge
from collections import Counter

jieba.setLogLevel(jieba.logging.INFO)


def normalize_zh_aswer(s):

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def normalize_en_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def classification_score(prediction, ground_truth, **kwargs):
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        # 这段有点看不懂，什么叫在答案中出现但不是答案？
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        score = 1.0 / len(em_match_list)
    else:
        score = 0.0
    return score


def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]


def rouge_zh_score(prediction, ground_truth, **kwargs):
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    score = rouge_score(prediction, ground_truth)
    return score


def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_en_answer(prediction)
    normalized_ground_truth = normalize_en_answer(ground_truth)
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens_norm = [normalize_zh_aswer(t) for t in prediction_tokens]
    ground_truth_tokens_norm = [normalize_zh_aswer(t) for t in ground_truth_tokens]
    prediction_tokens = [t for t in prediction_tokens_norm if len(t) > 0]
    ground_truth_tokens = [t for t in ground_truth_tokens_norm if len(t) > 0]
    return f1_score(prediction_tokens, ground_truth_tokens)


def GAOKAO_math(prediction, ground_truth, **kwargs):
    score = 0
    if len(ground_truth) > 1:
        pattern = r"[A-D]"
        matches = re.findall(pattern, prediction)
        prediction_answer = ""
        if matches:
            reversed_prediction = prediction[::-1]
            if len(matches) > 1:
                for i, match in enumerate(matches):
                    if i == 0:
                        prediction_answer += match
                    else:
                        distance = (
                            reversed_prediction.find(matches[i - 1])
                            - reversed_prediction.find(match)
                            - 1
                        )
                        if distance > 5:
                            break
                        prediction_answer += match
                prediction_answer = "".join(sorted(set(prediction_answer)))
            # 全选对了
            if prediction_answer == ground_truth:
                score = 1
            # 选对了一部分
            elif all(option in ground_truth for option in prediction_answer) and len(
                prediction_answer
            ) < len(ground_truth):
                score = 0.5
    else:
        pattern = r"[A-D]"
        matches = re.findall(pattern, prediction)
        if matches and matches[-1] == ground_truth:
            score = 1
    return score
