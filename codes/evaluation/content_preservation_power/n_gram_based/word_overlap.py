import sys
import argparse
import pprint
import statistics
import tensorflow as tf
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords
 
def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument('--source_file_path',
        type=str,
        default='')
    argparser.add_argument('--target_file_path',
        type=str,
        default='')
    argparser.add_argument('--scores_output_file',
        type=str,
        default='')
    argparser.add_argument('--source_style',
        type=str,
        default='') 
        #'positive' or 'negative' or ''
        #(if it is set to '' & args.remove_style_markers=True, both style markers are removed from seqs) 
    argparser.add_argument('--target_style',
        type=str,
        default='')
        #'positive' or 'negative' or ''
        #(if it is set to '' & args.remove_style_markers=True, both style markers are removed from seqs) 
    argparser.add_argument('--pos_style_markers',
            type=str,
            default="./opinion-lexicon-English/positive-words-cleaned.txt")
    argparser.add_argument('--neg_style_markers',
            type=str,
            default="./opinion-lexicon-English/negative-words-cleaned.txt")
    argparser.add_argument('--remove_stopwords',
            type=bool,
            default=False)
    argparser.add_argument('--remove_style_markers',
            type=bool,
            default=False)
    args = argparser.parse_args()
    print ('------------------------------------------------')
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(args))
    print ('------------------------------------------------')
    return args

def get_style_markers(file_name):
    with open(file_name) as style_markers_file:
        style_markers = style_markers_file.readlines()
    style_marker_set = set(word.strip() for word in style_markers)
    return style_marker_set

def get_stopwords():
    nltk_stopwords = set(stopwords.words('english'))
    sklearn_stopwords = stop_words.ENGLISH_STOP_WORDS
    all_stopwords = set()
    all_stopwords |= spacy_stopwords
    all_stopwords |= nltk_stopwords
    all_stopwords |= sklearn_stopwords
    return all_stopwords

def word_overlap_score_evaluator(args):
    actual_word_lists, generated_word_lists= list(), list() 
    with open(args.source_file_path) as source_file, open(args.target_file_path) as target_file:
        #assert len(source_file.readlines())==len(target_file.readlines()), "length error"
        for line_1, line_2 in zip(source_file, target_file):
            actual_word_lists.append(tf.keras.preprocessing.text.text_to_word_sequence(line_1))
            generated_word_lists.append(tf.keras.preprocessing.text.text_to_word_sequence(line_2))

    if args.remove_stopwords:
        english_stopwords = get_stopwords()
    else:
        english_stopwords = set([]) 
    sentiment_words_positive = get_style_markers(args.pos_style_markers)
    sentiment_words_negative = get_style_markers(args.neg_style_markers)

    sentiment_words_total = sentiment_words_negative | sentiment_words_positive
    scores = list()
    for word_list_1, word_list_2 in zip(actual_word_lists, generated_word_lists):
        score = 0
        words_1 = set(word_list_1)
        words_2 = set(word_list_2)

        if args.remove_style_markers :
            if args.source_style == 'positive':
                words_1 -= sentiment_words_positive
            elif args.source_style == 'negative':
                words_1 -= sentiment_words_negative
            else: # putting the styles to '' and '' in args, in case we want to remove both styles from both seqs
                words_1 -= sentiment_words_negative
                words_1 -= sentiment_words_positive
            
            if args.target_style == 'positive':
                words_2 -= sentiment_words_positive
            elif args.target_style == 'negative':
                words_2 -= sentiment_words_negative
            else: # putting the styles to '' and '' in args, in case we want to remove both styles from both seqs
                words_2 -= sentiment_words_negative
                words_2 -= sentiment_words_positive 
        
        words_1 -= english_stopwords
        words_2 -= english_stopwords

        word_intersection = words_1 & words_2
        word_union = words_1 | words_2

        if word_union:
            score = float(len(word_intersection)) / len(word_union)
            scores.append(score)
    with open(args.scores_output_file,'w') as fw:
        for score in scores:
            fw.write(str(score))
            fw.write('\n')
    print('Number of scores',len(scores))
    word_overlap_score = statistics.mean(scores) if scores else 0
    del english_stopwords
    del sentiment_words_positive
    del sentiment_words_negative
    return word_overlap_score

if __name__ == "__main__":
    args = load_arguments()
    word_overlap_score = word_overlap_score_evaluator(args)
    print('word_overlap_score', word_overlap_score) 
    

  
