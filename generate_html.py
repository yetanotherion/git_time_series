import os
import shutil

class HtmlGenerator(object):
    def __init__(self, collector, directory, curr_dir):
        self.collector = collector
        self.output_directory = os.path.join(directory, "git_stats")
        self.curr_dir = curr_dir
        self.data_to_plot = {}
        for data, time_granularity in ((self.collector.per_day, 'day'),
                                       (self.collector.per_week, 'week')):
            data_at_time_granularity = self.data_to_plot.setdefault(time_granularity, {})
            data_at_time_granularity['nb_commits'] = self.collector.nb_commits(data)
            data_at_time_granularity['nb_added_lines'] = self.collector.nb_added_lines(data)
            data_at_time_granularity['avg_size_of_commits'] = self.collector.avg_size_of_commits(data)
            data_at_time_granularity['nb_active_authors'] = self.collector.nb_active_authors(data)

    @staticmethod
    def compute_varname(name, value):
        return name + "_" + value

    @classmethod
    def compute_divid(cls, name, value):
        return cls.compute_varname(name, value) + "_id"

    @staticmethod
    def to_data_object(date, value):
        return "[%.2f,%.2f]" % (date, value)

    @staticmethod
    def to_ms(xys):
        return [(xy[0] * 1000.0, xy[1]) for xy in xys]

    def generate_scripts(self):
        res = []
        for time_granularity in ('day', 'week'):
            related_time = self.data_to_plot[time_granularity]
            for (k, v) in related_time.iteritems():
                varname = self.compute_varname(time_granularity, k)
                divid = self.compute_divid(time_granularity, k)
                v = self.to_ms(v)
                values = ','.join([self.to_data_object(kk, vv)
                                   for (kk, vv) in v])
                res.append("var %s = [%s];" % (varname, values))
                res.append("plotData(\"%s\", {" % (divid,))
                res.append("   \"title\": \"%s\"," % (varname,))
                res.append("   \"yAxisTitle\": \"%s\"," % ("count",))
                res.append("}, %s);" % (varname,))
        return res

    def generate_divs(self):
        res = []
        for time_granularity in ('day', 'week'):
            related_time = self.data_to_plot[time_granularity]
            for k in related_time:
                divid = self.compute_divid(time_granularity, k)
                res.append("<div id=\"%s\"></div>" % (divid,))
        return res

    @staticmethod
    def indent(n, lines):
        toindent = n * " "
        return [toindent + line for line in lines]

    def generate_index(self):
        return """<!doctype html>
<html>
  <head>
    <title>Git stats</title>
    <script src="highcharts.js"></script>
    <script src="exporting.js"></script>
    <script src="display-chart.js"></script>
  </head>
  <body>
%s
    <script>
%s
    </script>
  </body>
</html>""" % ("\n".join(self.indent(4, self.generate_divs())),
              "\n".join(self.indent(6, self.generate_scripts())))

    def generate(self):
        html_output = os.path.join(self.output_directory, "html")
        if os.path.exists(html_output):
            shutil.rmtree(html_output)
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        shutil.copytree(os.path.join(self.curr_dir, "html"), html_output)
        with open(os.path.join(html_output, 'index.html'), 'w') as f:
            f.write(self.generate_index())
