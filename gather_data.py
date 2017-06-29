import subprocess
import time
import os
import re
import platform

from itertools import tee
from itertools import izip
from sklearn.cluster import KMeans
import numpy as np

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

#TODO: add propper taken from GitStats mention
ON_LINUX = (platform.system() == 'Linux')

def getpipeoutput(cmds, quiet = False, cwd=None):
	start = time.time()
	if not quiet and ON_LINUX and os.isatty(1):
		print '>> ' + ' | '.join(cmds),
		sys.stdout.flush()
	p0 = subprocess.Popen(cmds[0], stdout = subprocess.PIPE, shell = True, cwd=cwd)
	p = p0
	for x in cmds[1:]:
		p = subprocess.Popen(x, stdin = p0.stdout, stdout = subprocess.PIPE, shell = True, cwd=cwd)
		p0 = p
	output = p.communicate()[0]
	end = time.time()
	if not quiet:
		if ON_LINUX and os.isatty(1):
			print '\r',
		print '[%.5f] >> %s' % (end - start, ' | '.join(cmds))
	return output.rstrip('\n')


class Commit(object):
    def __init__(self, timestamp, author, nb_files_changed, nb_insert, nb_delete, sha1):
        self.timestamp = timestamp
        self.author = author
        self.nb_files_changed = nb_files_changed
        self.nb_insert = nb_insert
        self.nb_delete = nb_delete
        self.sha1 = sha1

    def size(self):
        return self.nb_insert - self.nb_delete

class CommitBuilder(object):
    def __init__(self):
        self.timestamp = None
        self.author = None
        self.sha1 = None
        self.nb_files_changed = None
        self.nb_insert = None
        self.nb_delete = None

    def build(self):
        res = Commit(self.timestamp, self.author, self.nb_files_changed,
                      self.nb_insert, self.nb_delete, self.sha1)
        if abs(res.size()) > 1000:
            print "BIG commit: %d, %s" % (res.size(), res.sha1)
        return res
class Utils(object):

    @staticmethod
    def group_by(data, f):
        res = {}
        for x in data:
            curr_group = res.setdefault(f(x), [])
            curr_group.append(x)
        return res

    @staticmethod
    def approximate_to_format(fmt, timestamp):
        local_time = time.localtime(timestamp)
        str_local_time = time.strftime(fmt, local_time)
        struct_time = time.strptime(str_local_time, fmt)
        return time.mktime(struct_time)

    @classmethod
    def get_local_day(cls, timestamp):
        return cls.approximate_to_format('%Y-%m-%d', timestamp)

    @classmethod
    def get_local_week(cls, timestamp):
        day = cls.get_local_day(timestamp)
        return day - int(time.strftime('%w', time.localtime(day))) * 3600 * 24.0

    @staticmethod
    def average(l):
        if len(l) == 0:
            return 0.0
        return float(sum(l)) / float(len(l))


class DataCollector(object):
    def __init__(self, directory):
        self.directory = directory
        self.commits = []
        self.per_day = {}
        self.per_week = {}

    def parse_files_changed_line(self, line):
        numbers = [int(x) for x in re.findall('\d+', line)]
        files = numbers[0]
        inserted = 0
        deleted = 0
        valid_line = True
	if len(numbers) == 3:
	    (inserted, deleted) = (numbers[1], numbers[2])
        elif "+" in line:
            inserted = numbers[1]
        elif "-" in line:
            deleted = numbers[1]
        else:
            valid_line = False
	    print 'Warning: failed to handle line "%s"' % (line,)
        return valid_line, files, inserted, deleted

    def run(self):
        lines = getpipeoutput(['git log --shortstat --pretty=format:"%at %an %H"'],
                              cwd=self.directory).split('\n')
        commit_builder = None
        for line in filter(lambda x: bool(x),
                           map(lambda x: x.strip(), lines)):
            if any (x in line for x in ('file changed', 'files changed')):
                valid_line, files, inserted, deleted = self.parse_files_changed_line(line)
                if valid_line:
                    commit_builder.nb_files_changed = files
                    commit_builder.nb_insert = inserted
                    commit_builder.nb_delete = deleted
                    self.commits.append(commit_builder.build())
            else:
                splited_line = line.split(' ')
                commit_builder = CommitBuilder()
                commit_builder.timestamp = float(splited_line[0])
                commit_builder.author = ' '.join(splited_line[1:-1])
                commit_builder.sha1 = splited_line[-1]
        self.per_day = Utils.group_by(self.commits,
                                      lambda x: Utils.get_local_day(x.timestamp))
        self.per_week = Utils.group_by(self.commits,
                                       lambda x: Utils.get_local_week(x.timestamp))
    @staticmethod
    def sort_data(data):
        return sorted(data.iteritems(),
                      key=lambda x: x[0])

    @staticmethod
    def filter_outliers(data, threshold=100):
        kmeans = KMeans(init='random', n_clusters=2, n_init=10)
        X = np.array([[abs(xy[1])] for xy in data])
        kmeans.fit(X)
        res_one, res_two = [], []
        res_one_center = None
        for xy in data:
            center = kmeans.predict(abs(xy[1]))
            if res_one_center is None or res_one_center == center:
                res_one_center = center
                res_one.append(xy)
            else:
                res_two.append(xy)
        def f(xys):
            return [abs(xy[1]) for xy in xys]
        y_one, y_two = f(res_one), f(res_two)
        min_one, max_one = min(y_one), max(y_one)
        min_two, max_two = min(y_two), max(y_two)
        if max_one > max_two:
            temp = res_one
            res_one = res_two
            res_two = temp
        y_one, y_two = f(res_one), f(res_two)
        max_one, min_two = max(y_one), min(y_two)
        diff = min_two - max_one
        thresold_estimator = diff / max_one
        print min_one, max_one, min_two, max_two
        print thresold_estimator
        if thresold_estimator > threshold:
            print "skipping %d out of %d elements (%.2f)" % (len(res_two), len(data), len(res_two) / float(len(data)) * 100.0)
            return res_one
        return data


    @classmethod
    def put_explicit_zeros(cls, data, delta):
        x = [xy[0] for xy in data]
        min_x = int(min(x))
        max_x = int(max(x))
        res = {x: 0 for x in range(min_x, max_x + delta, delta)}
        for xy in data:
            res[float(xy[0])] = xy[1]
        return sorted(res.iteritems(),
                      key=lambda x: x[0])

    @classmethod
    def format_data(cls, data, delta, add_zero=True):
        res = cls.filter_outliers(cls.sort_data(data))
        if add_zero:
            res = cls.put_explicit_zeros(res, delta)
        return res

    @classmethod
    def nb_commits(cls, data, delta):
        return cls.format_data({k: len(v) for (k, v) in data.iteritems()},
                               delta)

    @classmethod
    def nb_added_lines(cls, data, delta):
        return cls.format_data({k: sum(x.nb_insert - x.nb_delete for x in v)
                                for (k, v) in data.iteritems()},
                               delta)

    @classmethod
    def nb_lines(cls, data, delta):
        if not data:
            return data

        def f(accum, xy):
            x, y = xy[0], xy[1]
            new_point = [x, accum[-1][1] + y]
            accum.append(new_point)
            return accum
        added_lines = cls.format_data({k: sum(x.nb_insert - x.nb_delete for x in v)
                                       for (k, v) in data.iteritems()},
                                      delta,
                                      add_zero=False)
        return reduce(f, added_lines[1:], [added_lines[0]])

    @classmethod
    def avg_size_of_commits(cls, data, delta):
        return cls.format_data({k: Utils.average([x.size() for x in v])
                                for (k, v) in data.iteritems()},
                               delta,
                               add_zero=False)

    @classmethod
    def nb_active_authors(cls, data, delta):
        return cls.format_data({k: len(set([x.author for x in v]))
                                for (k, v) in data.iteritems()},
                               delta)
