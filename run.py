import sys
import os

from gather_data import DataCollector
from generate_html import HtmlGenerator

if __name__ == '__main__':
    curr_dir = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
    output = sys.argv[1]
    collector = DataCollector(output)
    collector.run()
    generator = HtmlGenerator(collector, output, curr_dir)
    generator.generate()
