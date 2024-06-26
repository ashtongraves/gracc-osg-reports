import re
import traceback
import sys
import copy
from collections import defaultdict
import argparse

from opensearchpy import Search

from gracc_reporting import ReportUtils

LOGFILE = 'osgprojectreporter.log'
MAXINT = 2**31 - 1


# Helper Functions
def key_to_lower(bucket):
    return bucket.key.lower()


def parse_report_args():
    """
    Specific argument parser for this report.
    :return: Namespace of parsed arguments
    """
    parser = argparse.ArgumentParser(parents=[ReportUtils.get_report_parser()])
    parser.add_argument("-r", "--report-type", dest="report_type",
                        type=str, help="Report type (OSG, XD. or OSG-Connect")
    parser.add_argument('--nosum', dest="isSum", action='store_false',
                        help="Do not show a total line")
    return parser.parse_args()


class OSGProjectReporter(ReportUtils.Reporter):
    """Class to hold the information for and run the OSG Project Report

    :param str report_type: OSG, XD. or OSG-Connect
    :param str config_file: Configuration file
    :param str start: Start time for report range
    :param str end: End time for report range
    :param bool isSum: Show a total line at bottom of report, defaults to True
    """
    def __init__(self, report_type, config_file, start, end=None, isSum=True,
                 **kwargs):

        super(OSGProjectReporter, self).__init__(report_type=report_type, 
                                          config_file=config_file, 
                                          start=start, 
                                          end=end, 
                                          **kwargs)
        self.isSum = isSum
        self.report_type = self._validate_report_type(report_type)
        self.header = ["Project Name", "PI", "Institution", "Field of Science",
                     "Wall Hours"]
        self.logger.info("Report Type: {0}".format(self.report_type))
        self.tgmatch = re.compile('TG-')

    def run_report(self):
        """Higher level method to handle the process flow of the report
        being run"""
        self.send_report()

    def query(self):
        """Method to query Elasticsearch cluster for OSGProjectReporter information

        :return elasticsearch_dsl.Search: Search object containing ES query
        """
        # Gather parameters, format them for the query
        starttimeq = self.start_time.isoformat()
        endtimeq = self.end_time.isoformat()

        probes = self.config['project'][self.report_type.lower()]['probe_list']

        if self.verbose:
            self.logger.debug(probes)
            self.logger.info(self.indexpattern)

        # Elasticsearch query and aggregations
        s = Search(using=self.client, index=self.indexpattern) \
                .filter("range", EndTime={"gte": starttimeq, "lt": endtimeq}) \
                .filter("range", WallDuration={"gt": 0}) \
                .filter("terms", ProbeName=probes) \
                .filter("term", ResourceType="Payload")[0:0]
        # Size 0 to return only aggregations
        # Bucket, metric aggs
        Bucket = s.aggs.bucket("ProjectName", "terms", field="ProjectName",
                               size=MAXINT, order={"_term":"asc"},
                               missing="UNKNOWN")\
                    .bucket("OIM_PIName", "terms", field="OIM_PIName", missing="UNKNOWN", size=MAXINT)\
                    .bucket("OIM_Organization", "terms", field="OIM_Organization", missing="UNKNOWN", size=MAXINT)\
                    .bucket("OIM_FieldOfScience", "terms", field="OIM_FieldOfScience", missing="UNKNOWN", size=MAXINT)

        Bucket.metric("CoreHours", "sum", field="CoreHours")

        return s

    def generate_report_file(self):
        """Takes data from query response and parses it to send to other
        functions for processing"""
        results = self.run_query()

        unique_terms = ['ProjectName', 'OIM_PIName', 'OIM_Organization',
                        'OIM_FieldOfScience']
        metrics = ['CoreHours']

        def recurseBucket(curData, curBucket, index, data):
            """
            Recursively process the buckets down the nested aggregations

            :param curData: Current parsed data that describes curBucket and will be copied and appended to
            :param bucket curBucket: A elasticsearch bucket object
            :param int index: Index of the unique_terms that we are processing
            :param data: list of dicts that holds results of processing

            :return: None.  But this will operate on a list *data* that's passed in and modify it
            """
            curTerm = unique_terms[index]

            # Check if we are at the end of the list
            if not curBucket[curTerm]['buckets']:
                # Make a copy of the data
                nowData = copy.deepcopy(curData)
                data.append(nowData)
            else:
                # Get the current key, and add it to the data
                for bucket in self.sorted_buckets(curBucket[curTerm], key=key_to_lower):
                    nowData = copy.deepcopy(
                        curData)  # Hold a copy of curData so we can pass that in to any future recursion
                    nowData[curTerm] = bucket['key']
                    if index == (len(unique_terms) - 1):
                        # reached the end of the unique terms
                        for metric in metrics:
                            nowData[metric] = bucket[metric].value
                            # Add the doc count
                        nowData["Count"] = bucket['doc_count']
                        data.append(nowData)
                    else:
                        recurseBucket(nowData, bucket, index + 1, data)

        data = []
        recurseBucket({}, results, 0, data)
        allterms = copy.copy(unique_terms)
        allterms.extend(metrics)

        print(data)
        for entry in data:
            yield [entry[field] for field in allterms]

    def format_report(self):
        """Report formatter.  Returns a dictionary called report containing the
        columns of the report.

        :return dict: Constructed dict of report information for
        Reporter.send_report to send report from"""
        report = defaultdict(list)

        for result_list in self.generate_report_file():
            if not self._validate_type_results(result_list[0]):
                continue

            if self.verbose:
                print("{0}\t{1}\t{2}\t{3}\t{4}".format(*result_list))
            mapdict = dict(list(zip(self.header, result_list)))
            for key, item in mapdict.items():
                report[key].append(item)

        if self.isSum:
            tot = sum(report['Wall Hours'])
            for field in self.header:
                if field == 'Project Name':
                    report[field].append('Total')
                elif field == 'Wall Hours':
                    report[field].append(tot)
                else:
                    report[field].append('')

            if self.verbose:
                self.logger.info("The total wall hours in this report are "
                                 "{0}".format(tot))

        return report

    def _validate_report_type(self, report_type):
        """
        Validates that the report being run is one of three types.  Sets
        title of report if it's given a valid report type

        :param str report_type: One of OSG, XD, or OSG-Connect
        :return report_type: report type
        """
        validtypes = {"OSG": "OSG-Direct", "XD": "OSG-XD",
                      "OSG-Connect": "OSG-Connect"}
        fmt = "%Y-%m-%d %H:%M"
        if report_type in validtypes:
            self.title = "{0} Project Report for {1} - {2}".format(
                validtypes[report_type], self.start_time.strftime(fmt),
                self.end_time.strftime(fmt))
            return report_type
        else:
            raise Exception("Must use report type {0}".format(
                ', '.join((name for name in validtypes)))
            )

    def _validate_type_results(self, pname):
        """
        Makes sure we include ONLY "TG-" projects in XD report and only non-
        "TG-" projects in other reports
        :param str pname: Project name to validate
        :return bool: Whether or not the project name pname is valid for the report type
        """
        return True if \
            (self.report_type == 'XD' and self.tgmatch.match(pname) or
             self.report_type != 'XD' and not self.tgmatch.match(pname)) \
            else False


def main():
    args = parse_report_args()
    logfile_fname = args.logfile if args.logfile is not None else LOGFILE

    try:
        r = OSGProjectReporter(report_type=args.report_type,
                        config_file=args.config,
                        start=args.start,
                        end=args.end,
                        isSum=args.isSum,
                        verbose=args.verbose,
                        is_test=args.is_test,
                        no_email=args.no_email,
                        logfile=logfile_fname,
                        template=args.template)
        r.run_report()
        r.logger.info("OSG Project Report executed successfully")

    except Exception as e:
        ReportUtils.runerror(args.config, e, traceback.format_exc(), args.logfile)
        sys.exit(1)
    sys.exit(0)


if __name__=="__main__":
    main()
