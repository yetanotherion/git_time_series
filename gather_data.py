import subprocess
import time
import os
import re
import platform
from itertools import tee
from itertools import izip


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
    def __init__(self, timestamp, author, nb_files_changed, nb_insert, nb_delete):
        self.timestamp = timestamp
        self.author = author
        self.nb_files_changed = nb_files_changed
        self.nb_insert = nb_insert
        self.nb_delete = nb_delete

    def size(self):
        return self.nb_insert - self.nb_delete

class CommitBuilder(object):
    def __init__(self):
        self.timestamp = None
        self.author = None
        self.nb_files_changed = None
        self.nb_insert = None
        self.nb_delete = None

    def build(self):
        return Commit(self.timestamp, self.author, self.nb_files_changed,
                      self.nb_insert, self.nb_delete)
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
        lines = getpipeoutput(['git log --shortstat --pretty=format:"%at %an"'],
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
                commit_builder.author = ' '.join(splited_line[1:])
        self.per_day = Utils.group_by(self.commits,
                                      lambda x: Utils.get_local_day(x.timestamp))
        self.per_week = Utils.group_by(self.commits,
                                       lambda x: Utils.get_local_week(x.timestamp))
    @staticmethod
    def sort_data(data):
        return sorted(data.iteritems(),
                      key=lambda x: x[0])

    @staticmethod
    def filter_outliers(data):
        values = [x[1] for x in data]
        pos_values = [x for x in values if x > 0]
        average_pos = Utils.average(pos_values)
        neg_values = [x for x in values if x < 0]
        average_neg = Utils.average(neg_values)

        threshold = 10
        max_value = average_pos * threshold
        min_value = average_neg * threshold
        def f(x):
            if x > 0:
                if x < max_value:
                    return x
                return max_value + threshold # to find outliers
            else:
                if x > min_value:
                    return x
                return min_value - threshold
        return [(k, f(v)) for (k, v) in data]

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
