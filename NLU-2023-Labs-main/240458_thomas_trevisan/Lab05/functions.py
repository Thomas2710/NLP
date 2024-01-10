from nltk.parse.generate import generate
from nltk.grammar import Nonterminal


def generate_sentences(
    probabilistic_grammar, depth=10, starting_symbol=Nonterminal("S"), N=10, mode="nltk"
):
    if mode == "nltk":
        for sent in generate(
            probabilistic_grammar, start=starting_symbol, depth=depth, n=N
        ):
            print(sent)
    elif mode == "pcfg":
        for sent in probabilistic_grammar.generate(N):
            print(sent + "\n")
