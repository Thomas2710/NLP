from nltk.parse.dependencygraph import DependencyGraph

#Parsing the last 100 sentences of treebank
def get_dependency_tree_list(stanza_nlp, spacy_nlp, dependency_treebank, N = 100):
    stanza_dp_list = []
    spacy_dp_list = []
    for i in range(N):
        stanza_doc = stanza_nlp(' '.join(dependency_treebank.sents()[-N:][i]))
        spacy_doc = spacy_nlp(' '.join(dependency_treebank.sents()[-N:][i]))
        #Printing things
    #     for spacy_sent ,stanza_sent in zip(spacy_doc.sents, stanza_doc.sents):
    #        for spacy_token, stanza_token in zip(spacy_sent, stanza_sent):
    #            print("Spacy: {}\t{}\t{}\t{}".format(spacy_token.i, spacy_token.text, spacy_token.head, spacy_token.dep_))
    #            print("Stanza: {}\t{}\t{}\t{}".format(stanza_token.i, stanza_token.text, stanza_token.head, stanza_token.dep_))
    #            print('')

        spacy_df = spacy_doc._.pandas
        stanza_df = stanza_doc._.pandas

        spacy_tmp = spacy_df[["FORM", 'XPOS', 'HEAD', 'DEPREL']].to_string(header=False, index=False)
        stanza_tmp = stanza_df[["FORM", 'XPOS', 'HEAD', 'DEPREL']].to_string(header=False, index=False)

        stanza_dp = DependencyGraph(stanza_tmp)
        stanza_dp_list.append(stanza_dp)
        spacy_dp = DependencyGraph(spacy_tmp)
        spacy_dp_list.append(spacy_dp)
        
    return stanza_dp_list, spacy_dp_list