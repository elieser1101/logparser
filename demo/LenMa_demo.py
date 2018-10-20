#templateminer
#__init__
----------------------------------------
#template
#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Template(object):
    def __init__(self, index, words, logid):
        self._index = index
        self._words = words
        self._nwords = len(words)
        self._counts = 1
        self._logid = [logid]

    @property
    def index(self):
        return self._index

    @property
    def words(self):
        return self._words

    @property
    def nwords(self):
        return self._nwords

    @property
    def counts(self):
        return self._counts

    def _dump_as_json(self):
        """Dumps the data structure as a JSON format to serialize the
        object.

        This internal function is called by the TemplateManager
        class.
        """
        assert(False)

    def _restore_from_json(self, data):
        """Initializes the instance with the provided JSON data.

        This internal function is normally called by the initializer.
        """
        assert(False)

    def get_similarity_score(self, new_words):
        """Retruens a similarity score.

        Args:
          new_words: An array of words.

        Returns:
          score: in float.
        """
        assert(False)

    def update(self, new_words):
        """Updates the template data using the supplied new_words.
        """
        assert(False)

    def __str__(self):
        template = ' '.join([self.words[idx] if self.words[idx] != '' else '*' for idx in range(self.nwords)])
        return '{index}({nwords})({counts}):{template}'.format(
            index=self.index,
            nwords=self.nwords,
            counts=self._counts,
            template=' '.join([self.words[idx] if self.words[idx] != '' else '*' for idx in range(self.nwords)]))

class TemplateManager(object):
    def __init__(self):
        self._templates = []

    @property
    def templates(self):
        return self._templates

    def infer_template(self, words):
        """Infer the best matching template, or create a new template if there
        is no similar template exists.

        Args:
          words: An array of words.

        Returns:
          A template instance.

        """
        assert(False)


    def dump_template(self, index):
        """Dumps a specified template data structure usually in a text
        format.

        Args:
          index: a template index.

        Returns:
          A serialized text data of the specified template.
        """
        assert(False)

    def restore_template(self, data):
        """Creates a template instance from data (usually a serialized
        data when LogDatabase.close() method is called.

        This function is called by the LogDatabase class.

        Args:
          data: a data required to rebuild a template instance.

        Returns:
          A template instance.
        """
        assert(False)

    def _append_template(self, template):
        """Append a template.

        This internal function may be called by the LogDatabase
        class too.

        Args:
          template: a new template to be appended.

        Returns:
          template: the appended template.
        """
        assert(template.index == len(self.templates))
        self.templates.append(template)
        return template

#-------------------------------------------------
#lenma_template
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LenMa: Length Matters Syslog Message Clustering.
"""

import json
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import template

class LenmaTemplate(template.Template):
    def __init__(self, index=None, words=None, logid=None, json=None):
        if json is not None:
            # restore from the jsonized data.
            self._restore_from_json(json)
        else:
            # initialize with the specified index and words vlaues.
            assert(index is not None)
            assert(words is not None)
            self._index = index
            self._words = words
            self._nwords = len(words)
            self._wordlens = [len(w) for w in words]
            self._counts = 1
            self._logid = [logid]

    @property
    def wordlens(self):
        return self._wordlens

    def _dump_as_json(self):
        description = str(self)
        return json.dumps([self.index, self.words, self.nwords, self.wordlens, self.counts])

    def _restore_from_json(self, data):
        (self._index,
         self._words,
         self._nwords,
         self._wordlens,
         self._counts) = json.loads(data)

    def _try_update(self, new_words):
        try_update = [self.words[idx] if self._words[idx] == new_words[idx]
                      else '' for idx in range(self.nwords)]
        if (self.nwords - try_update.count('')) < 3:
            return False
        return True

    def _get_accuracy_score(self, new_words):
        # accuracy score
        # wildcard word matches any words
        fill_wildcard = [self.words[idx] if self.words[idx] != ''
                         else new_words[idx] for idx in range(self.nwords)]
        ac_score = accuracy_score(fill_wildcard, new_words)
        return ac_score

    def _get_wcr(self):
        return self.words.count('') / self.nwords

    def _get_accuracy_score2(self, new_words):
        # accuracy score 2
        # wildcard word matches nothing
        wildcard_ratio = self._get_wcr()
        ac_score = accuracy_score(self.words, new_words)
        return (ac_score / (1 - wildcard_ratio), wildcard_ratio)

    def _get_similarity_score_cosine(self, new_words):
        # cosine similarity
        wordlens = np.asarray(self._wordlens).reshape(1, -1)
        new_wordlens = np.asarray([len(w) for w in new_words]).reshape(1, -1)
        cos_score = cosine_similarity(wordlens, new_wordlens)
        return cos_score

    def _get_similarity_score_jaccard(self, new_words):
        ws = set(self.words) - set('')
        nws = set([new_words[idx] if self.words[idx] != '' else ''
                   for idx in range(len(new_words))]) - set('')
        return len(ws & nws) / len(ws | nws)

    def _count_same_word_positions(self, new_words):
        c = 0
        for idx in range(self.nwords):
            if self.words[idx] == new_words[idx]:
                c = c + 1
        return c

    def get_similarity_score(self, new_words):
        # heuristic judge: the first word (process name) must be equal
        if self._words[0] != new_words[0]:
            return 0

        # check exact match
        ac_score = self._get_accuracy_score(new_words)
        if  ac_score == 1:
            return 1

        cos_score = self._get_similarity_score_cosine(new_words)

        case = 6
        if case == 1:
            (ac2_score, ac2_wcr) = self._get_accuracy_score2(new_words)
            if ac2_score < 0.5:
                return 0
            return cos_score
        elif case == 2:
            (ac2_score, ac2_wcr) = self._get_accuracy_score2(new_words)
            return (ac2_score + cos_score) / 2
        elif case == 3:
            (ac2_score, ac2_wcr) = self._get_accuracy_score2(new_words)
            return ac2_score * cos_score
        elif case == 4:
            (ac2_score, ac2_wcr) = self._get_accuracy_score2(new_words)
            print(ac2_score, ac2_wcr)
            tw = 0.5
            if ac2_score < tw + (ac2_wcr * (1 - tw)):
                return 0
            return cos_score
        elif case == 5:
            jc_score = self._get_similarity_score_jaccard(new_words)
            if jc_score < 0.5:
                return 0
            return cos_score
        elif case == 6:
            if self._count_same_word_positions(new_words) < 3:
                return 0
            return cos_score

    def update(self, new_words, logid):
        self._counts += 1
        self._wordlens = [len(w) for w in new_words]
        #self._wordlens = [(self._wordlens[idx] + len(new_words[idx])) / 2
        #                  for idx in range(self.nwords)]
        self._words = [self.words[idx] if self._words[idx] == new_words[idx]
                       else '' for idx in range(self.nwords)]
        self._logid.append(logid)

    def print_wordlens(self):
        print('{index}({nwords})({counts}):{vectors}'.format(
            index=self.index,
            nwords=self.nwords,
            counts=self._counts,
            vectors=self._wordlens))

    def get_logids(self):
        return self._logid

class LenmaTemplateManager(template.TemplateManager):
    def __init__(self,
                 threshold=0.9,
                 predefined_templates=None):
        self._templates = []
        self._threshold = threshold
        if predefined_templates:
            for template in predefined_templates:
                self._append_template(template)

    def dump_template(self, index):
        return self.templates[index]._dump_as_json()

    def restore_template(self, data):
        return LenmaTemplate(json=data)

    def infer_template(self, words, logid):
        nwords = len(words)

        candidates = []
        for (index, template) in enumerate(self.templates):
            if nwords != template.nwords:
                continue
            score = template.get_similarity_score(words)
            if score < self._threshold:
                continue
            candidates.append((index, score))
        candidates.sort(key=lambda c: c[1], reverse=True)
        if False:
            for (i,s) in candidates:
                print('    ', s, self.templates[i])

        if len(candidates) > 0:
            index = candidates[0][0]
            self.templates[index].update(words, logid)
            return self.templates[index]

        new_template = self._append_template(
            LenmaTemplate(len(self.templates), words, logid))
        return new_template

if __name__ == '__main__':
    import datetime
    import sys

    from templateminer.basic_line_parser import BasicLineParser as LP

    parser = LP()
    templ_mgr = LenmaTemplateManager()

    nlines = 0
    line = sys.stdin.readline()
    while line:
        if False:
            if nlines % 1000 == 0:
                print('{0} {1} {2}'.format(nlines, datetime.datetime.now().timestamp(), len(templ_mgr.templates)))
            nlines = nlines + 1
        (month, day, timestr, host, words) = parser.parse(line)
        t = templ_mgr.infer_template(words)
        line = sys.stdin.readline()

    for t in templ_mgr.templates:
        print(t)

    for t in templ_mgr.templates:
        t.print_wordlens()

#------------------------------------------------
"""
Description: This file implements the Lenma algorithm for log parsing
Author: LogPAI team
License: MIT
"""


import pandas as pd
import re
import os
import hashlib
from collections import defaultdict
from datetime import datetime

class LogParser(object):
    def __init__(self, indir, outdir, log_format, threshold=0.9, predefined_templates=None, rex=[]):
        self.path = indir
        self.savePath = outdir
        self.logformat = log_format
        self.rex = rex
        self.wordseqs = []
        self.df_log = pd.DataFrame()
        self.wordpos_count = defaultdict(int)
        self.templ_mgr = lenma_template.LenmaTemplateManager(threshold=threshold, predefined_templates=predefined_templates)
        self.logname = None

    def parse(self, logname):
        print('Parsing file: ' + os.path.join(self.path, logname))
        self.logname = logname
        starttime = datetime.now()
        headers, regex = self.generate_logformat_regex(self.logformat)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logname), regex, headers, self.logformat)
        for idx, line in self.df_log.iterrows():
            line = line['Content']
            if self.rex:
                for currentRex in self.rex:
                    line = re.sub(currentRex, '<*>', line)
            words = line.split()
            self.templ_mgr.infer_template(words, idx)
        self.dump_results()
        print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - starttime))

    def dump_results(self):
        if not os.path.isdir(self.savePath):
            os.makedirs(self.savePath)

        df_event = []
        templates = [0] * self.df_log.shape[0]
        template_ids = [0] * self.df_log.shape[0]
        for t in self.templ_mgr.templates:
            template = ' '.join(t.words)
            eventid = hashlib.md5(' '.join(template).encode('utf-8')).hexdigest()[0:8]
            logids = t.get_logids()
            for logid in logids:
                templates[logid] = template
                template_ids[logid] = eventid
            df_event.append([eventid, template, len(logids)])

        self.df_log['EventId'] = template_ids
        self.df_log['EventTemplate'] = templates

        pd.DataFrame(df_event, columns=['EventId', 'EventTemplate', 'Occurrences']).to_csv(os.path.join(self.savePath, self.logname + '_templates.csv'), index=False)
        self.df_log.to_csv(os.path.join(self.savePath, self.logname + '_structured.csv'), index=False)


    def log_to_dataframe(self, log_file, regex, headers, logformat):
        ''' Function to transform log file to dataframe '''
        log_messages = []
        linecount = 0
        with open(log_file, 'r') as fin:
            for line in fin.readlines():
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    pass
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf

    def generate_logformat_regex(self, logformat):
        '''
        Function to generate regular expression to split log messages
        '''
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex
#---------------------------------------------------------
#!/usr/bin/env python
import sys
sys.path.append('../')
#.demo
input_dir  = '../logs/HDFS/' # The input directory of log file
output_dir = 'Lenma_result/' # The output directory of parsing results
log_file   = 'HDFS_2k.log' # The input log file name
log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>' # HDFS log format
threshold  = 0.9 # TODO description (default: 0.9)
regex      = [] # Regular expression list for optional preprocessing (default: [])

parser = LenMa.LogParser(input_dir, output_dir, log_format, threshold=threshold, rex=regex)
parser.parse(log_file)
