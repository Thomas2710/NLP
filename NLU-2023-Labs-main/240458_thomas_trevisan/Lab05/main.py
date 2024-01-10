import nltk
from nltk.grammar import PCFG

from nltk.parse.viterbi import ViterbiParser
from nltk.parse.pchart import InsideChartParser
from nltk.corpus import treebank
from pcfg import PCFG
from functions import *

sentences = [
    " Ronald plays poker on friday night with Mario ",
    " Ronald plays catan on monday with Franco ",
]
pcfg_rules = [
    "S -> NP VP [1.0]",
    "NP -> N [0.5]",
    "NP -> N PP [0.5]",
    "VP -> V NP [1.0]",
    "PP -> P NP  [0.5]",
    "PP -> P ADV  [0.5]",
    "ADV -> A ADV [0.5]",
    "ADV -> A PP [0.5]",
    'N -> "Ronald" [0.2]',
    'N -> "poker" [0.2]',
    'N -> "catan" [0.2]',
    'A -> "friday" [0.4]',
    'A -> "monday" [0.4]',
    'N -> "Mario" [0.2]',
    'N -> "Franco" [0.2]',
    'A -> "night" [0.2]',
    'V -> "plays" [1.0]',
    'P -> "on" [0.5]',
    'P -> "with" [0.5]',
]


probabilistic_grammar = nltk.PCFG.fromstring(pcfg_rules)
parser = nltk.ViterbiParser(probabilistic_grammar)
parser1 = nltk.InsideChartParser(probabilistic_grammar)

depths = [5, 10, 30]
starting_symbols = [Nonterminal("S"), Nonterminal("NP"), Nonterminal("VP")]

print("----------------------------ViterbiParser--------------------------\n")
for sent in sentences:
    for tree in parser.parse(sent.split()):
        print(tree)


print("\n----------------------------InsideChartParser--------------------------\n")
for sent in sentences:
    for tree in parser1.parse(sent.split()):
        print(tree)


for depth in depths:
    for starting_symbol in starting_symbols:
        print(
            f"\n----------------------------Sentence Generation with nltk starting symbol {starting_symbol} and depth {depth}--------------------------\n"
        )
        generate_sentences(
            probabilistic_grammar,
            depth=depth,
            starting_symbol=starting_symbol,
            N=10,
            mode="nltk",
        )


print(
    "\n----------------------------Sentence Generation with PCFG--------------------------\n"
)
probabilistic_grammar1 = PCFG.fromstring(pcfg_rules)
generate_sentences(probabilistic_grammar1, mode="pcfg")