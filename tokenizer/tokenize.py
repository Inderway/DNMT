import sentencepiece as spm
import config

def train(input_file, vocab_size, model_name, model_type, character_coverage):
    """
    search on https://github.com/google/sentencepiece/blob/master/doc/options.md to learn more about the parameters
    :param input_file: one-sentence-per-line raw corpus file. No need to run tokenizer, normalizer or preprocessor.
                       By default, SentencePiece normalizes the input with Unicode NFKC.
                       You can pass a comma-separated list of files.
    :param vocab_size: vocabulary size, e.g., 8000, 16000, or 32000
    :param model_name: output model name prefix. <model_name>.model and <model_name>.vocab are generated.
    :param model_type: model type. Choose from unigram (default), bpe, char, or word.
                       The input sentence must be pretokenized when using word type.
    :param character_coverage: amount of characters covered by the model, good defaults are: 0.9995 for languages with
                               rich character set like Japanse or Chinese and 1.0 for other languages with
                               small character set.
    """
    # In vocab, the right number is not id
    # padding==0, unknown==1, begin==2 end==3
    input_argument = f'--input=%s --model_prefix=%s --vocab_size=%s --model_type=%s --character_coverage=%s  \
                     --pad_id={config.PAD_IDX} --unk_id=1 --bos_id={config.BOS_IDX} --eos_id={config.EOS_IDX}'
    cmd = input_argument % (input_file, model_name, vocab_size, model_type, character_coverage)
    spm.SentencePieceTrainer.Train(cmd)


def run():
    en_input = './data/corpus.en'
    en_vocab_size = 32000
    en_model_name = 'eng'
    en_model_type = 'bpe'
    en_character_coverage = 1
    train(en_input, en_vocab_size, en_model_name, en_model_type, en_character_coverage)

    ch_input = './data/corpus.ch'
    ch_vocab_size = 32000
    ch_model_name = 'chn'
    ch_model_type = 'bpe'
    ch_character_coverage = 0.9995
    train(ch_input, ch_vocab_size, ch_model_name, ch_model_type, ch_character_coverage)


def test():
    sp = spm.SentencePieceProcessor()
    text = "拿破仑建立法兰西第一帝国。"
    print("input sentence: \n"+text)
    sp.Load("./tokenizer/chn.model")
    print("Pieces:")
    print(sp.EncodeAsPieces(text))
    print("Ids:")
    print(sp.EncodeAsIds(text))
    print("input id sequence:")
    a = [0, 1, 2, 3, 4, 5]
    print(a)
    print(sp.decode_ids(a))
    # use this function to check the correspondence between piece and id
    # print([[sp.IdToPiece(id), id] for id in range(10)])


if __name__ == "__main__":
    #run()
    test()
