try:
    import xml.etree.ElementTree as ET, getopt, logging, sys, random, re, copy, os, warnings
    import numpy as np
    from collections import Counter
    import operator
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.svm import SVC
    import nltk
    from nltk.stem import PorterStemmer
    from xml.sax.saxutils import escape
    from sklearn.externals import joblib
except:
    sys.exit('Some package is missing... Perhaps <re>?')

warnings.filterwarnings("ignore") #to ignore sklearns deprecation warnings

# Stopwords, imported from NLTK (v 2.0.4)
stopwords = set(
    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
     'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
     'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
     'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
     'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
     'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
     'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
     'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
     'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])

contractions = re.compile(r"'|-|\"")
# all non alphanumeric
symbols = re.compile(r'(\W+)', re.U)
# single character removal
singles = re.compile(r'(\s\S\s)', re.I|re.U)
# separators (any whitespace)
seps = re.compile(r'\s+')

fix = lambda text: escape(text.encode('utf8')).replace('\"', '&quot;')
'''Simple fix for writing out text.'''

class Category:
    '''Category objects contain the term of the category (e.g., food, price, etc.) of a sentence.'''

    def __init__(self, term=''):
        self.term = term

    def create(self, element):
        self.term = element.attrib['category']
        return self

    def update(self, term=''):
        self.term = term	

class Instance:
    '''An instance is a sentence, modeled out of XML. It contains the text, and any aspect categories.'''

    def __init__(self, element):
        self.text = element.find('text').text
        self.id = element.get('id')
        self.aspect_categories = [Category(term='').create(e) for es in element.findall('Opinions')
                                  for e in es if
                                  es is not None]

    def get_aspect_categories(self):
        return [c.term.lower() for c in self.aspect_categories]

    def add_aspect_category(self, term):
        c = Category(term)
        self.aspect_categories.append(c)

class Corpus:
    '''A corpus contains instances, and is useful for training algorithms or splitting to train/test files.'''

    def __init__(self, elements):
        self.corpus = [Instance(e) for e in elements]
        self.size = len(self.corpus)
        self.texts = [t.text for t in self.corpus]

    def clean_tags(self):
        for i in range(len(self.corpus)):
            self.corpus[i].aspect_terms = []

    def split(self, threshold=1, shuffle=False):
        '''Split to train/test, based on a threshold. Turn on shuffling for randomizing the elements beforehand.'''
        clone = copy.deepcopy(self.corpus)
        if shuffle: random.shuffle(clone)
        train = clone[:int(threshold * self.size)]
        test = clone[int(threshold * self.size):]
        return train, test

    def write_out(self, filename, instances, short=True):
        with open(filename, 'w') as o:
            o.write('<sentences>\n')
            for i in instances:
                o.write('\t<sentence id="%s">\n' % (i.id))
                o.write('\t\t<text>%s</text>\n' % fix(i.text))
                o.write('\t\t<Opinions>\n')
                if not short:
                    for c in i.aspect_categories:
                        o.write('\t\t\t<Opinion category="%s"/>\n' % (fix(c.term)))
                o.write('\t\t</Opinions>\n')
                o.write('\t</sentence>\n')
            o.write('</sentences>')


def load_lexicon(lex_type, b):

    #entity lexica
    laptop = []
    display = []
    cpu = []
    mb = []
    hd = []
    mem = []
    bat = []
    power = []
    keyb = []
    mouse = []
    fans = []
    opt = []
    ports = []
    graph = []
    mm = []
    hw = []
    os = []
    sw = []
    warranty = []
    ship = []
    supp = []
    comp = []

    #attribute lexica
    gen = []
    pr = []
    qual = []
    oper = []
    use = []
    des = []
    port = []
    conn = []
    misc = []

    f = open(lex_type+"_laptop_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        laptop.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])
    
    f = open(lex_type+"_display_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        display.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_cpu_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        cpu.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_mb_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        mb.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_hd_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        hd.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_mem_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        mem.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_bat_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        bat.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_power_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        power.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_keyb_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        keyb.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_mouse_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        mouse.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_fans_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        fans.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_opt_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        opt.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_ports_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        ports.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_graphs_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        graph.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_mm_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        mm.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_hw_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        hw.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_os_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        os.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_sw_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        sw.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_warranty_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        warranty.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_ship_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        ship.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_supp_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        supp.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_comp_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        comp.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_gen_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        gen.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_price_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        pr.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_qual_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        qual.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_oper_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        oper.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_use_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        use.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_des_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        des.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_port_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        port.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_conn_lexicon.txt", "r")
    for line in f:
       feats = line.split()
       if b is True:
           feats[0] = tuple(feats[0].split(','))
       conn.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_misc_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        misc.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])
    
    f.close()
    return [laptop, display, cpu, mb, hd, mem, bat, power, keyb, mouse, fans, opt, ports, graph, mm, hw, os, sw, warranty, ship, supp, comp, gen, pr, qual, oper, use, des, port, conn, misc]


def load_idf(path):
    idf_dict = {}

    f = open(path+".txt", "r")
    for line in f:
        idf_dict[line.split()[0]] = float(line.split()[1])
        
    return idf_dict

def load_word2vec(path):
    w2v_model = {}
    f = open(path+".txt", "r")
    for line in f:
        vector = []
        fields = line.split()
        name = fields[0]
        for x in fields[1:]:
            vector.append(float(x))
        w2v_model[name] = np.asarray(vector)
        
    return w2v_model

def load_category_centroids(path):
    centroids = []

    f = open(path+".txt", "r")
    for line in f:
        centroids.append(line.split())

    return centroids

def clean(text): 
    text = text.lower()
    text = contractions.sub('', text)
    text = symbols.sub(r' \1 ', text)
    text = singles.sub(' ', text)
    text = seps.sub(' ', text)
    return text

def normalize_horizontal(w2v_vectors):
    '''Normalize the word embeddings horizontally, using the L2-norm.'''
    feature_vectors = []
    
    norm = np.linalg.norm(w2v_vectors)

    for vec in w2v_vectors:
        feature_vectors.append(vec/norm if norm > 0. else 0.)

    return feature_vectors


def calculate_feats(n):
    '''Calculate the max, min, median and avg of a list with floats'''
    
    max_ = max(n) if n else 0.
    min_ = min(n) if n else 0.
    avg = np.average(n) if n else 0.
    median = np.median(n) if n else 0.
    
    return max_, min_, avg, median

def append_feats_to_list(l, feats):
    '''Append the features to the feature list.'''

    for f in feats:
        l.append(f)
    return l

def assign_features(lexica, words, flag):
    feature_list = []
    
    for lex in lexica:
        precision = []
        recall = []
        f1 = []
        for entry in lex:
            for w in words:
                if flag is False: #can't compare tuples (like in bigrams) with the '==' operator
                    if w == entry[0]:
                        precision.append(entry[2])
                        recall.append(entry[3])
                        f1.append(entry[4])
                else: #so you use the 'in' keyword
                    if w in entry[0]:
                        precision.append(entry[2])
                        recall.append(entry[3])
                        f1.append(entry[4])
        pre_max, pre_min, pre_avg, pre_median = calculate_feats(precision)
        re_max, re_min, re_avg, re_median = calculate_feats(recall)
        f1_max, f1_min, f1_avg, f1_median = calculate_feats(f1)
        feature_list = append_feats_to_list(feature_list, [pre_max, pre_min, pre_avg, pre_median, re_max, re_min,
                                                                re_avg, re_median, f1_max, f1_min, f1_avg, f1_median])

    return feature_list


def createTrainModel(traincorpus):
      # classifiers with the dicts as features
    laptop_clf1 = SVC(kernel='rbf', C=5.656854249492381, gamma=0.02209708691207961, probability=True)
    display_clf1 = SVC(kernel='rbf', C=2.378414230005442, gamma=0.02209708691207961, probability=True)
    cpu_clf1 = SVC(kernel='sigmoid', C=64, gamma=0.04419417382415922, probability=True)
    mb_clf1 = SVC(kernel='linear', C=0.5946035575013605, probability=True)
    hd_clf1 = SVC(kernel='rbf', C=64.0, gamma=0.0078125, probability=True)
    memory_clf1 = SVC(kernel='sigmoid', C=53.81737057623773, gamma=0.03125, probability=True)
    battery_clf1 = SVC(kernel='sigmoid', C=22.627416997969522, gamma=0.04419417382415922, probability=True)
    power_clf1 = SVC(kernel='sigmoid', C=107.63474115247546, gamma=0.018581361171917516, probability=True)
    keyboard_clf1 = SVC(kernel='poly', C=0.0011613350732448448, gamma=1.189207115002721, probability=True)
    mouse_clf1 = SVC(kernel='rbf', C=5.656854249492381, gamma=0.0078125, probability=True)
    fans_clf1 = SVC(kernel='sigmoid', C=13.454342644059432, gamma=0.03125, probability=True)
    opt_drives_clf1 = SVC(kernel='sigmoid', C=1024, gamma=0.052556025953357156, probability=True)
    ports_clf1 = SVC(kernel='linear', C=1.189207115002721, probability=True)
    graphics_clf1 = SVC(kernel='sigmoid', C=128.0, gamma=0.03125, probability=True)
    mm_devs_clf1 = SVC(kernel='sigmoid', C=304.4370214406966, gamma=0.02209708691207961, probability=True)
    hardw_clf1 = SVC(kernel='sigmoid', C=32, gamma=0.03716272234383503, probability=True)
    os_clf1 = SVC(kernel='poly', C=0.000244140625, gamma=4.0, probability=True)
    softw_clf1 = SVC(kernel='sigmoid', C=6.727171322029716, gamma=0.03716272234383503, probability=True)
    warranty_clf1 = SVC(kernel='sigmoid', C=107.63474115247546, gamma=0.07432544468767006, probability=True)
    shipping_clf1 = SVC(kernel='linear', C=1.0, probability=True)
    support_clf1 = SVC(kernel='rbf', C=38.05462768008707, gamma=0.011048543456039806, probability=True)
    company_clf1 = SVC(kernel='rbf', C=26.908685288118864, gamma=0.02209708691207961, probability=True)
    general_clf1 = SVC(kernel='rbf', C=3.363585661014858, gamma=0.0625, probability=True)
    price_clf1 = SVC(kernel='sigmoid', C=6.727171322029716, gamma=0.03716272234383503, probability=True)
    quality_clf1 = SVC(kernel='sigmoid', C=4.0, gamma=0.026278012976678578, probability=True)
    op_perf_clf1 = SVC(kernel='rbf', C=4.0, gamma=0.125, probability=True)
    usability_clf1 = SVC(kernel='rbf', C=13.454342644059432, gamma=0.018581361171917516, probability=True)
    des_feats_clf1 = SVC(kernel='rbf', C=2, gamma=0.10511205190671431, probability=True)
    portability_clf1 = SVC(kernel='rbf', C=2.378414230005442, gamma=0.10511205190671431, probability=True)
    connectivity_clf1 = SVC(kernel='sigmoid', C=22.627416997969522, gamma=0.052556025953357156, probability=True)
    misc_clf1 = SVC(kernel='rbf', C=1.4142135623730951, gamma=0.10511205190671431, probability=True)

    laptop_clf2 = SVC(kernel='rbf', C=1.4142135623730951, gamma=1.189207115002721, probability=True)
    display_clf2 = SVC(kernel='poly', C=0.0005806675366224224, gamma=2.378414230005442, probability=True)
    cpu_clf2 = SVC(kernel='sigmoid', C=304.4370214406966, gamma=0.03716272234383503, probability=True)
    mb_clf2 = SVC(kernel='poly', C=0.000244140625, gamma=4.0, probability=True)
    hd_clf2 = SVC(kernel='poly', C=0.0011613350732448448, gamma=2.378414230005442, probability=True)
    memory_clf2 = SVC(kernel='poly', C=0.000244140625, gamma=3.363585661014858, probability=True)
    battery_clf2 = SVC(kernel='rbf', C=16.0, gamma=0.42044820762685725, probability=True)
    power_clf2 = SVC(kernel='linear', C=26.908685288118864, probability=True)
    keyboard_clf2 = SVC(kernel='rbf', C=22.627416997969522, gamma=0.5946035575013605, probability=True)
    mouse_clf2 = SVC(kernel='rbf', C=76.10925536017415, gamma=0.10511205190671431, probability=True)
    fans_clf2 = SVC(kernel='linear', C=0.125, probability=True)
    opt_drives_clf2 = SVC(kernel='sigmoid', C=1722.1558584396073, gamma=0.03125, probability=True)
    ports_clf2 = SVC(kernel='linear', C=32.0, probability=True)
    graphics_clf2 = SVC(kernel='rbf', C=26.908685288118864, gamma=0.29730177875068026, probability=True)
    mm_devs_clf2 = SVC(kernel='rbf', C=152.2185107203483, gamma=0.052556025953357156, probability=True)
    hardw_clf2 = SVC(kernel='linear', C=13.454342644059432, probability=True)
    os_clf2 = SVC(kernel='rbf', C=128.0, gamma=0.1767766952966369, probability=True)
    softw_clf2 = SVC(kernel='rbf', C=430.5389646099018, gamma=0.013139006488339289, probability=True)
    warranty_clf2 = SVC(kernel='linear', C=8.0, probability=True)
    shipping_clf2 = SVC(kernel='linear', C=16.0, probability=True)
    support_clf2 = SVC(kernel='rbf', C=26.908685288118864, gamma=0.1767766952966369, probability=True)
    company_clf2 = SVC(kernel='rbf', C=128.0, gamma=0.0625, probability=True)
    general_clf2 = SVC(kernel='rbf', C=2.378414230005442, gamma=0.3535533905932738, probability=True)
    price_clf2 = SVC(kernel='poly', C=0.0002903337683112112, gamma=4.0, probability=True)
    quality_clf2 = SVC(kernel='poly', C=0.0011613350732448448, gamma=6.727171322029716, probability=True)
    op_perf_clf2 = SVC(kernel='rbf', C=4.756828460010884, gamma=0.8408964152537145, probability=True)
    usability_clf2 = SVC(kernel='rbf', C=4.756828460010884, gamma=2.378414230005442, probability=True)
    des_feats_clf2 = SVC(kernel='rbf', C=64.0, gamma=0.25, probability=True)
    portability_clf2 = SVC(kernel='rbf', C=16.0, gamma=0.5946035575013605, probability=True)
    connectivity_clf2 = SVC(kernel='rbf', C=128.0, gamma=0.21022410381342863, probability=True)
    misc_clf2 = SVC(kernel='rbf', C=38.05462768008707, gamma=1.0, probability=True)

    stemmer = PorterStemmer()
    
    unigrams_lexica = load_lexicon("lexica/laptops/unigrams", False)
    bigrams_lexica = load_lexicon("lexica/laptops/bigram", True)
    bipos_lexica = load_lexicon("lexica/laptops/bipos", True)
    stemmed_unigrams_lexica = load_lexicon("lexica/laptops/stemmed_unigrams", False)
    stemmed_bigrams_lexica = load_lexicon("lexica/laptops/stemmed_bigrams", True)
    idf_dict = load_idf("lexica/idf_laptops")
    category_centroids = load_category_centroids("lexica/laptops/category_centroid")
    
    joblib.dump(unigrams_lexica, './models/unigrams_lexica.pkl', compress =1)
    joblib.dump(bigrams_lexica, './models/bigrams_lexica.pkl', compress =1)
    joblib.dump(bipos_lexica, './models/bipos_lexica.pkl', compress =1)
    joblib.dump(stemmed_unigrams_lexica, './models/stemmed_unigrams_lexica.pkl', compress =1)
    joblib.dump(stemmed_bigrams_lexica, './models/stemmed_bigrams_lexica.pkl', compress =1)
    joblib.dump(idf_dict, './models/idf_dict.pkl', compress =1)
    joblib.dump(category_centroids, './models/category_centroids.pkl', compress =1)

    print 'Loading Word2Vec model...'
    w2v_model = load_word2vec('lexica/word_embeds_laptops')
    joblib.dump(w2v_model, './models/w2v_model.pkl', compress =1)
    print 'Done!'
    
    train_sentences1 = [] #the sentences to be used for feature extraction
    train_sentences2 = []

    #entity labels
    laptop_labels = []
    display_labels = []
    cpu_labels = []
    mb_labels = []
    hd_labels = []
    memory_labels = []
    battery_labels = []
    power_labels = []
    keyboard_labels = []
    mouse_labels = []
    fans_labels = []
    opt_drives_labels = []
    ports_labels = []
    graphics_labels = []
    mm_devs_labels = []
    hardw_labels = []
    os_labels = []
    softw_labels = []
    warranty_labels = []
    shipping_labels = []
    support_labels = []
    company_labels = []

    #attribute labels
    general_labels = []
    price_labels = []
    quality_labels = []
    op_perf_labels = []
    usability_labels = []
    des_feats_labels = []
    portability_labels = []
    connectivity_labels = []
    misc_labels = []

    #number of categories
    cats = []

    print('Creating train feature vectors...')

    #extracting words from sentences and appending them labels
    for instance in traincorpus.corpus:     
        words = (re.findall(r"[\w']+", instance.text.lower())) #the unigrams list

        sentence_without_stopwords = ""
        for w in words:
            if w not in stopwords:
                sentence_without_stopwords = sentence_without_stopwords + " " + w
        clean_words = clean(sentence_without_stopwords).split()

        #calculate the embedding of the words of the current sentence
        sentence_vector_feats = []
        words_with_embeds = []
        for w in clean_words:
            word_vector_feats = []
            if w in w2v_model:
                words_with_embeds.append(w)
                for vector in w2v_model[w]:
                    word_vector_feats.append(vector)
                sentence_vector_feats.append(word_vector_feats)

        #calculate the centroid of the embeddings of the sentence (using tf)
        centroid_feats = []
        for vec_num in range(0,200):
            sum_vectors = 0.
            sum_idf = 0.
            for w_index, word_vector in enumerate(sentence_vector_feats):
                sum_vectors = sum_vectors + (word_vector[vec_num] * idf_dict[words_with_embeds[w_index]])
                sum_idf = sum_idf + idf_dict[words_with_embeds[w_index]]
            centroid = sum_vectors / (sum_idf) if sum_idf > 0. else 0.
            centroid_feats.append(centroid)

        normalized_centroid_feats = normalize_horizontal(centroid_feats)

        #compute the cosine similarity of the centroid of the sentence with the centroid of each category
        distances = []
        for category in category_centroids:
            distances.append(cosine_similarity(normalized_centroid_feats, category)[0][0])
            
        stemmed_words = []
        stemmed_bi_words = []
        for w in words:
            if w not in stopwords:
                stemmed_words.append(stemmer.stem(w)) #the stemmed unigrams list
            stemmed_bi_words.append(stemmer.stem(w))
                
        stemmed_bigrams = nltk.bigrams(stemmed_bi_words)
        stemmed_bigrams_list = []
        for w in stemmed_bigrams:
            stemmed_bigrams_list.append(w) #the stemmed bigrams list
                
        bigram_words = nltk.bigrams(words)
        bigram_list = []
        for w in bigram_words:
            bigram_list.append(w) #the bigram list

        tags = nltk.pos_tag(words)
        tags_set = set() #the pos list
        for _, t in tags:
                tags_set.add(t)

        bitags = nltk.bigrams(list(tags_set))
        bitag_list = []
        for t in bitags:
            bitag_list.append(t) #the pos bigrams list

        unigrams_feats = []
        bigrams_feats = []
        bipos_feats = []
        stemmed_unigrams_feats = []
        stemmed_bigrams_feats = []            

        #unigrams features
        unigrams_feats = assign_features(unigrams_lexica, words, False)

        #bigrams features
        bigrams_feats = assign_features(bigrams_lexica, bigram_list, True)

        #pos bigrams features
        bipos_feats = assign_features(bipos_lexica, bitag_list, True)
                    
        #stemmed_unigram features
        stemmed_unigrams_feats = assign_features(stemmed_unigrams_lexica, stemmed_words, False)

        #stemmed_bigram features
        stemmed_bigrams_feats = assign_features(stemmed_bigrams_lexica, stemmed_bigrams_list, True)

        train_sentences1.append(unigrams_feats + bigrams_feats + bipos_feats + stemmed_unigrams_feats + stemmed_bigrams_feats)
        train_sentences2.append(normalized_centroid_feats + distances)

        #to avoid training a sentence more than once to a category, since there are
        #categories like laptop#quality and laptop#price assigned to a sentence
        ent_set = set()
        attr_set = set()
        for c in instance.get_aspect_categories():
            ent_attr = c.split('#')
            ent_set.add(ent_attr[0]) #store the entity
            attr_set.add(ent_attr[1]) #store the attribute
            cats.append(c)
            
        #check entity category
        if "laptop" in ent_set:
            laptop_labels.append(1)
        else:
            laptop_labels.append(0)

        if "display" in ent_set:
            display_labels.append(1)
        else:
            display_labels.append(0)
            
        if "cpu" in ent_set:
            cpu_labels.append(1)
        else:
            cpu_labels.append(0)
            
        if "motherboard" in ent_set:
            mb_labels.append(1)
        else:
            mb_labels.append(0)
            
        if "hard_disc" in ent_set:
            hd_labels.append(1)
        else:
            hd_labels.append(0)
            
        if "memory" in ent_set:
            memory_labels.append(1)
        else:
            memory_labels.append(0)
            
        if "battery" in ent_set:
            battery_labels.append(1)
        else:
            battery_labels.append(0)
            
        if "power_supply" in ent_set:
            power_labels.append(1)
        else:
            power_labels.append(0)
            
        if "keyboard" in ent_set:
            keyboard_labels.append(1)
        else:
            keyboard_labels.append(0)
            
        if "mouse" in ent_set:
            mouse_labels.append(1)
        else:
            mouse_labels.append(0)
            
        if "fans_cooling" in ent_set:
            fans_labels.append(1)
        else:
            fans_labels.append(0)
            
        if "optical_drives" in ent_set:
            opt_drives_labels.append(1)
        else:
            opt_drives_labels.append(0)
            
        if "ports" in ent_set:
            ports_labels.append(1)
        else:
            ports_labels.append(0)
            
        if "graphics" in ent_set:
            graphics_labels.append(1)
        else:
            graphics_labels.append(0)
            
        if "multimedia_devices" in ent_set:
            mm_devs_labels.append(1)
        else:
            mm_devs_labels.append(0)
            
        if "hardware" in ent_set:
            hardw_labels.append(1)
        else:
            hardw_labels.append(0)
            
        if "os" in ent_set:
            os_labels.append(1)
        else:
            os_labels.append(0)
            
        if "software" in ent_set:
            softw_labels.append(1)
        else:
            softw_labels.append(0)
            
        if "warranty" in ent_set:
            warranty_labels.append(1)
        else:
            warranty_labels.append(0)
            
        if "shipping" in ent_set:
            shipping_labels.append(1)
        else:
            shipping_labels.append(0)
            
        if "support" in ent_set:
            support_labels.append(1)
        else:
            support_labels.append(0)
            
        if "company" in ent_set:
            company_labels.append(1)
        else:
            company_labels.append(0)

        #check attribute category
        if "general" in attr_set:
            general_labels.append(1)
        else:
            general_labels.append(0)
            
        if "price" in attr_set: 
            price_labels.append(1)
        else:
            price_labels.append(0)
            
        if "quality" in attr_set: 
            quality_labels.append(1)
        else:
            quality_labels.append(0)
            
        if "operation_performance" in attr_set: 
            op_perf_labels.append(1)
        else:
            op_perf_labels.append(0)
            
        if "usability" in attr_set: 
            usability_labels.append(1)
        else:
            usability_labels.append(0)
            
        if "design_features" in attr_set: 
            des_feats_labels.append(1)
        else:
            des_feats_labels.append(0)
            
        if "portability" in attr_set: 
            portability_labels.append(1)
        else:
            portability_labels.append(0)
            
        if "connectivity" in attr_set: 
            connectivity_labels.append(1)
        else:
            connectivity_labels.append(0)
            
        if "miscellaneous" in attr_set: 
            misc_labels.append(1)
        else:
            misc_labels.append(0)
    joblib.dump(cats, './models/categories.pkl')
    cat_dict = Counter(cats)

    train_features1 = np.asarray(train_sentences1)
    train_features2 = np.asarray(train_sentences2)
    
    laptop_clf1.fit(train_features1, laptop_labels)
    display_clf1.fit(train_features1, display_labels)
    cpu_clf1.fit(train_features1, cpu_labels)
    mb_clf1.fit(train_features1, mb_labels)
    hd_clf1.fit(train_features1, hd_labels)
    memory_clf1.fit(train_features1, memory_labels)
    battery_clf1.fit(train_features1, battery_labels)
    power_clf1.fit(train_features1, power_labels)
    keyboard_clf1.fit(train_features1, keyboard_labels)
    mouse_clf1.fit(train_features1, mouse_labels)
    fans_clf1.fit(train_features1, fans_labels)
    opt_drives_clf1.fit(train_features1, opt_drives_labels)
    ports_clf1.fit(train_features1, ports_labels)
    graphics_clf1.fit(train_features1, graphics_labels)
    mm_devs_clf1.fit(train_features1, mm_devs_labels)
    hardw_clf1.fit(train_features1, hardw_labels)
    os_clf1.fit(train_features1, os_labels)
    softw_clf1.fit(train_features1, softw_labels)
    warranty_clf1.fit(train_features1, warranty_labels)
    shipping_clf1.fit(train_features1, shipping_labels)
    support_clf1.fit(train_features1, support_labels)
    company_clf1.fit(train_features1, company_labels)   
    general_clf1.fit(train_features1, general_labels)
    price_clf1.fit(train_features1, price_labels)
    quality_clf1.fit(train_features1, quality_labels)
    op_perf_clf1.fit(train_features1, op_perf_labels)
    usability_clf1.fit(train_features1, usability_labels)
    des_feats_clf1.fit(train_features1, des_feats_labels) 
    portability_clf1.fit(train_features1, portability_labels)
    connectivity_clf1.fit(train_features1, connectivity_labels)
    misc_clf1.fit(train_features1, misc_labels)
    
    joblib.dump(laptop_clf1, 'models/laptop_clf1Model.pkl', compress =1)
    joblib.dump(display_clf1, 'models/display_clf1Model.pkl', compress =1)
    joblib.dump(cpu_clf1, 'models/cpu_clf1Model.pkl', compress =1)
    joblib.dump(mb_clf1, 'models/mb_clf1Model.pkl', compress =1)
    joblib.dump(hd_clf1, 'models/hd_clf1Model.pkl', compress =1)
    joblib.dump(memory_clf1, 'models/memory_clf1Model.pkl', compress =1)
    joblib.dump(battery_clf1, 'models/battery_clf1Model.pkl', compress =1)
    joblib.dump(power_clf1, 'models/power_clf1Model.pkl', compress =1)
    joblib.dump(keyboard_clf1, 'models/keyboard_clf1Model.pkl', compress =1)
    joblib.dump(mouse_clf1, 'models/mouse_clf1Model.pkl', compress =1)
    joblib.dump(fans_clf1, 'models/fans_clf1Model.pkl', compress =1)
    joblib.dump(opt_drives_clf1, 'models/opt_drives_clf1Model.pkl', compress =1)
    joblib.dump(ports_clf1, 'models/ports_clf1Model.pkl', compress =1)
    joblib.dump(graphics_clf1, 'models/graphics_clf1Model.pkl', compress =1)
    joblib.dump(mm_devs_clf1, 'models/mm_devs_clf1Model.pkl', compress =1)
    joblib.dump(hardw_clf1, 'models/hardw_clf1Model.pkl', compress =1)
    joblib.dump(os_clf1, 'models/os_clf1Model.pkl', compress =1)
    joblib.dump(softw_clf1, 'models/softw_clf1Model.pkl', compress =1)
    joblib.dump(warranty_clf1, 'models/warranty_clf1Model.pkl', compress =1)
    joblib.dump(shipping_clf1, 'models/shipping_clf1Model.pkl', compress =1)
    joblib.dump(support_clf1, 'models/support_clf1Model.pkl', compress =1)
    joblib.dump(company_clf1, 'models/company_clf1Model.pkl', compress =1)
    joblib.dump(general_clf1, 'models/general_clf1Model.pkl', compress =1)
    joblib.dump(price_clf1, 'models/price_clf1Model.pkl', compress =1)
    joblib.dump(quality_clf1, 'models/quality_clf1Model.pkl', compress =1)
    joblib.dump(op_perf_clf1, 'models/op_perf_clf1Model.pkl', compress =1)
    joblib.dump(usability_clf1, 'models/usability_clf1Model.pkl', compress =1)
    joblib.dump(des_feats_clf1, 'models/des_feats_clf1Model.pkl', compress =1)
    joblib.dump(portability_clf1, 'models/portability_clf1Model.pkl', compress =1)
    joblib.dump(connectivity_clf1, 'models/connectivity_clf1Model.pkl', compress =1)
    joblib.dump(misc_clf1, 'models/misc_clf1Model.pkl', compress =1)


    laptop_clf2.fit(train_features2, laptop_labels)
    display_clf2.fit(train_features2, display_labels)
    cpu_clf2.fit(train_features2, cpu_labels)
    mb_clf2.fit(train_features2, mb_labels)
    hd_clf2.fit(train_features2, hd_labels)
    memory_clf2.fit(train_features2, memory_labels)
    battery_clf2.fit(train_features2, battery_labels)
    power_clf2.fit(train_features2, power_labels)
    keyboard_clf2.fit(train_features2, keyboard_labels)
    mouse_clf2.fit(train_features2, mouse_labels)
    fans_clf2.fit(train_features2, fans_labels)
    opt_drives_clf2.fit(train_features2, opt_drives_labels)
    ports_clf2.fit(train_features2, ports_labels)
    graphics_clf2.fit(train_features2, graphics_labels)
    mm_devs_clf2.fit(train_features2, mm_devs_labels)
    hardw_clf2.fit(train_features2, hardw_labels)
    os_clf2.fit(train_features2, os_labels)
    softw_clf2.fit(train_features2, softw_labels)
    warranty_clf2.fit(train_features2, warranty_labels)
    shipping_clf2.fit(train_features2, shipping_labels)
    support_clf2.fit(train_features2, support_labels)
    company_clf2.fit(train_features2, company_labels)    
    general_clf2.fit(train_features2, general_labels)
    price_clf2.fit(train_features2, price_labels)
    quality_clf2.fit(train_features2, quality_labels)
    op_perf_clf2.fit(train_features2, op_perf_labels)
    usability_clf2.fit(train_features2, usability_labels)
    des_feats_clf2.fit(train_features2, des_feats_labels)
    portability_clf2.fit(train_features2, portability_labels)
    connectivity_clf2.fit(train_features2, connectivity_labels)
    misc_clf2.fit(train_features2, misc_labels)

    joblib.dump(laptop_clf2, 'models/laptop_clf2Model.pkl', compress =1)
    joblib.dump(display_clf2, 'models/display_clf2Model.pkl', compress =1)
    joblib.dump(cpu_clf2, 'models/cpu_clf2Model.pkl', compress =1)
    joblib.dump(mb_clf2, 'models/mb_clf2Model.pkl', compress =1)
    joblib.dump(hd_clf2, 'models/hd_clf2Model.pkl', compress =1)
    joblib.dump(memory_clf2, 'models/memory_clf2Model.pkl', compress =1)
    joblib.dump(battery_clf2, 'models/battery_clf2Model.pkl', compress =1)
    joblib.dump(power_clf2, 'models/power_clf2Model.pkl', compress =1)
    joblib.dump(keyboard_clf2, 'models/keyboard_clf2Model.pkl', compress =1)
    joblib.dump(mouse_clf2, 'models/mouse_clf2Model.pkl', compress =1)
    joblib.dump(fans_clf2, 'models/fans_clf2Model.pkl', compress =1)
    joblib.dump(opt_drives_clf2, 'models/opt_drives_clf2Model.pkl', compress =1)
    joblib.dump(ports_clf2, 'models/ports_clf2Model.pkl', compress =1)
    joblib.dump(graphics_clf2, 'models/graphics_clf2Model.pkl', compress =1)
    joblib.dump(mm_devs_clf2, 'models/mm_devs_clf2Model.pkl', compress =1)
    joblib.dump(hardw_clf2, 'models/hardw_clf2Model.pkl', compress =1)
    joblib.dump(os_clf2, 'models/os_clf2Model.pkl', compress =1)
    joblib.dump(softw_clf2, 'models/softw_clf2Model.pkl', compress =1)
    joblib.dump(warranty_clf2, 'models/warranty_clf2Model.pkl', compress =1)
    joblib.dump(shipping_clf2, 'models/shipping_clf2Model.pkl', compress =1)
    joblib.dump(support_clf2, 'models/support_clf2Model.pkl', compress =1)
    joblib.dump(company_clf2, 'models/company_clf2Model.pkl', compress =1)
    joblib.dump(general_clf2, 'models/general_clf2Model.pkl', compress =1)
    joblib.dump(price_clf2, 'models/price_clf2Model.pkl', compress =1)
    joblib.dump(quality_clf2, 'models/quality_clf2Model.pkl', compress =1)
    joblib.dump(op_perf_clf2, 'models/op_perf_clf2Model.pkl', compress =1)
    joblib.dump(usability_clf2, 'models/usability_clf2Model.pkl', compress =1)
    joblib.dump(des_feats_clf2, 'models/des_feats_clf2Model.pkl', compress =1)
    joblib.dump(portability_clf2, 'models/portability_clf2Model.pkl', compress =1)
    joblib.dump(connectivity_clf2, 'models/connectivity_clf2Model.pkl', compress =1)
    joblib.dump(misc_clf2, 'models/misc_clf2Model.pkl', compress =1)

    print('Done!')

def validate(filename):
    '''Validate an XML file, w.r.t. the format given in the 5th task of **SemEval '16**.'''
    tree = ET.parse(filename)
    root = tree.getroot()

    elements = []	
    aspect_terms = []
    for review in root.findall('Review'):
        for sentences in review.findall('sentences'):
            for sentence in sentences.findall('sentence'):
                elements.append(sentence)
						
    return elements

def extract_subjectives(filename, new_filename):
	'''Extract only the subjective sentences and leave out the objective sentences and the sentences with the attribute 'OutOfScope=="TRUE".'''
	tree = ET.parse(filename)
	root = tree.getroot()
		
	with open(new_filename, 'w') as o:
		o.write('<Reviews>\n')
		for review in root.findall('Review'):
			o.write('\t<Review rid="%s">\n' % review.get('rid'))
			for sentences in review.findall('sentences'):
				o.write('\t\t<sentences>\n')
				for sentence in sentences.findall('sentence'):
					if (sentence.get('OutOfScope') != "TRUE"):
						if sentence.find('Opinions') != None:
							o.write('\t\t\t<sentence id="%s">\n' % (sentence.get('id')))
							o.write('\t\t\t\t<text>%s</text>\n' % (fix(sentence.find('text').text)))       
							for opinions in sentence.findall('Opinions'):
								o.write('\t\t\t\t<Opinions>\n')
								for opinion in opinions.findall('Opinion'):
									o.write('\t\t\t\t\t<Opinion category="%s" polarity="%s" from="%s" to="%s"/>\n' % (
										fix(opinion.get('category')), opinion.get('polarity'), opinion.get('from'), opinion.get('to')))
								o.write('\t\t\t\t</Opinions>\n')
							o.write('\t\t\t</sentence>\n')
				o.write('\t\t</sentences>\n')
			o.write('\t</Review>\n')
		o.write('</Reviews>')

def main(argv=None):
    trainfile = "data/laptops/train.xml"
    testfile = "data/laptops/test.xml"

    # Examine if the file is in proper XML format for further use.
    print ('Validating the file...')
    try:
        sentences = validate(trainfile)
        print ('PASSED! This corpus has: %d sentences.' % (len(sentences)))
    except:
        print ("Unexpected error:", sys.exc_info()[0])
        raise

    print('Extracting subjective sentences...')
    trainfile_ = 'train_subjectives.xml'  
    extract_subjectives(trainfile, trainfile_)
    print('Done!')
		
    # Get the corpus and split into train/test.
    corpus = Corpus(ET.parse(trainfile_).getroot().findall('./Review/sentences/sentence'))
    print 'corp size: ',corpus.size
    domain_name = 'laptops'

    train, seen = corpus.split(threshold=1)
    # Store train/test files and clean up the test files (no aspect terms or categories are present); then, parse back the files back.
    corpus.write_out('%s--train.xml' % domain_name, train, short=False)
    traincorpus = Corpus(ET.parse('%s--train.xml' % domain_name).getroot().findall('sentence'))
    print traincorpus.size
    createTrainModel(traincorpus)
    os.remove(trainfile_)

if __name__ == "__main__": main(sys.argv[1:])