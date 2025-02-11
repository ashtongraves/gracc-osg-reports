import os
import traceback
import smtplib
import email.utils
from email.mime.text import MIMEText
import sys
import copy
import argparse

from opensearchpy import Search

from utils import ReportUtils


MAXINT = 2**31 - 1


def parse_report_args():
    """
    Specific argument parser for this report.
    :return: Namespace of parsed arguments
    """
    parser = argparse.ArgumentParser(parents=[ReportUtils.get_report_parser()])
    parser.add_argument("-r", "--report-type", dest="report_type",
                        type=str, help="Report type (OSG or OSG-Connect")
    return parser.parse_args()


class MissingProjectReport(ReportUtils.Reporter):
    """
    Class to hold information for and to run OSG Missing Projects Report 
    :param: 
    """
    def __init__(self, report_type, config_file, start, end=None, **kwargs):

        super(MissingProjectReport, self).__init__(report_type=report_type, 
                                                   config_file=config_file, 
                                                   start=start,
                                                   end=end,
                                                   **kwargs)

        self.report_type = self._validate_report_type(report_type)
        self.logger.info("Report Type: {0}".format(self.report_type))
        
        # Temp files
        self.fname = 'OIM_Project_Name_Request_for_{0}'.format(self.report_type)
        if os.path.exists(self.fname):
            os.unlink(self.fname)

    def run_report(self):
        """Higher level method to handle the process flow of the report
        being run"""
        self.generate()

    def query(self):
        """
        Method to query Elasticsearch cluster for OSGReporter information

        :return elasticsearch_dsl.Search: Search object containing ES query
        """

        # Gather parameters, format them for the query
        starttimeq = self.start_time.isoformat()
        endtimeq = self.end_time.isoformat()

        probes = self.config['project'][self.report_type.lower()]['probe_list']

        if self.verbose:
            print(probes)
        s = Search(using=self.client, index=self.indexpattern) \
                .filter("range", EndTime={"gte": starttimeq, "lt": endtimeq}) \
                .filter("range", WallDuration={"gt": 0}) \
                .filter("terms", ProbeName=probes) \
                .filter("term", ResourceType="Payload") \
                .filter("exists", field="RawProjectName")[0:0]

        self.unique_terms = ['OIM_PIName', 'RawProjectName', 'ProbeName',
                 'CommonName', 'VOName']
        self.metrics = ['CoreHours']

        curBucket = s.aggs.bucket("OIM_PIName", "missing", field="OIM_PIName")

        for term in self.unique_terms[1:]:
            curBucket = curBucket.bucket(term, "terms", field=term, size=MAXINT)

        curBucket.metric(self.metrics[0], 'sum', field=self.metrics[0])

        return s

    def generate(self):
        """Higher-level method that calls the lower-level functions
        to generate the raw data for this report and pass it to the correct
        checkers
        """
        results = self.run_query()
        unique_terms = self.unique_terms
        metrics = self.metrics

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
                for bucket in curBucket[curTerm]['buckets']:
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
        recurseBucket({}, results.OIM_PIName, 1, data)
        if self.verbose:
            self.logger.info(data)

        if len(data) == 1 and not data[0]:  # No data.
            return

        # Check the missing projects
        for item in data:
            self._check_project(item)

        # Send the emails, delete temp files
        if os.path.exists(self.fname):
            self.send_email()
            os.unlink(self.fname)

    @staticmethod
    def no_name(name):
        return name == 'N/A' or name.upper() == "UNKNOWN"

    def create_request_to_register_oim(self, name, source, p=None, altfile=None):
        """Creates file with information related to project that will be sent later to OSG secretary
        Args:
            name(str) - project name
            source(str) - OSG, or  OSG-Connect"
            p(Project) - project
            altfile(str) - alternative file to write to
        """
        if not altfile:
            filename = "OIM_Project_Name_Request_for_{0}".format(source)
        else:
            filename = altfile

        with open(filename, 'a') as f:
            f.write("Project names that are reported from {0} but not "
                    "registered in OIM\n".format(source))
            f.write("ProjectName: {0}\n".format(name))

        return filename

    def _check_osg_or_osg_connect(self, data):
        """
        Checks to see if data describing project is OSG's responsibility to
        maintain

        :param dict data: Aggregated data about a missing project from ES query
        :return bool:
        """
        return ((self.report_type == 'OSG-Connect')
                or (self.report_type == 'OSG' and data['VOName'].lower() in
                    ['osg', 'osg-connect'])
                )

    def _check_project(self, data):
        """
        Handles the logic for what to do with records that don't have OIM info

        :param dict data: Aggregated data about a missing project from ES query
        :return:
        """
        p_name = data.get('RawProjectName')

        if not p_name or self.no_name(p_name):
            # No real Project Name in records
            self._write_noname_message(data)
            return
        elif self._check_osg_or_osg_connect(data):
            # OSG should have kept this up to date
            self.create_request_to_register_oim(p_name, self.report_type)
            return
        else:
            return

    def _write_noname_message(self, data):
        """
        Message to be sent to GOC for records with no project name.

        :param dict data: Aggregated data about a missing project from ES query
        :return:
        """

        for field in ('CommonName', 'VOName', 'ProbeName', 'CoreHours',
                      'RawProjectName', 'Count'):
            if not data.get(field):
                data[field] = "{0} not reported".format(field)

        fmt = "%Y-%m-%d %H:%M"

        msg = "{count} Payload records dated between {start} and {end} with:\n" \
              "\t CommonName: {cn}\n" \
              "\t VOName: {vo}\n" \
              "\t ProbeName: {probe}\n" \
              "\t Wall Hours: {ch}\n " \
              "were reported with no ProjectName (\"{pn}\") to GRACC.  Please " \
              "investigate.\n\n".format(count=data['Count'],
                                        start=self.start_time.strftime(fmt),
                                        end=self.end_time.strftime(fmt),
                                        cn=data['CommonName'],
                                        vo=data['VOName'],
                                        probe=data['ProbeName'],
                                        ch=data['CoreHours'],
                                        pn=data['RawProjectName'])

        with open(self.fname, 'a') as f:
            f.write(msg)

        return

    def send_email(self):
        """
        Sets email parameters and sends email.
        :return:
        """
        COMMASPACE = ', '

        fname = self.fname

        if self.check_no_email(self.email_info['to']['email']):
            return

        try:
            smtpObj = smtplib.SMTP(self.email_info['smtphost'])
        except Exception as e:
            self.logger.error(e)
            return

        with open(fname, 'r') as f:
            msg = MIMEText(f.read())

        to_stage = [email.utils.formataddr(pair)
                    for pair in zip(
                *(self.email_info['to'][key]
                 for key in ('name', 'email')))]
        msg['Subject'] = 'Records with no Project or Projects not ' \
                             'registered in OIM'
        msg['To'] = COMMASPACE.join(to_stage)
        msg['From'] = email.utils.formataddr((self.email_info['from']['name'],
                                              self.email_info['from']['email']))

        try:
            smtpObj = smtplib.SMTP(self.email_info["smtphost"])
            smtpObj.sendmail(
                self.email_info['from']['email'],
                self.email_info['to']['email'],
                msg.as_string())
            smtpObj.quit()
            self.logger.info("Sent email from file {0} to recipients {1}"
                             .format(fname, self.email_info['to']['email']))
        except Exception as e:
            self.logger.exception("Error:  unable to send email.\n{0}\n".format(e))
            raise

        return

    @staticmethod
    def _validate_report_type(report_type):
        """
        Validates that the report being run is one of three types.

        :param str report_type: One of OSG  or OSG-Connect
        :return report_type: report type
        """
        validtypes = {"OSG": "OSG-Direct", "OSG-Connect": "OSG-Connect"}
        if report_type in validtypes:
            return report_type
        else:
            raise Exception("Must use report type {0}".format(
                ', '.join((name for name in validtypes)))
            )


def main():
    args = parse_report_args()

    try:
        r = MissingProjectReport(report_type=args.report_type,
                                 config_file=args.config,
                                 start=args.start,
                                 end=args.end,
                                 verbose=args.verbose,
                                 is_test=args.is_test,
                                 no_email=args.no_email)
        r.run_report()
        r.logger.info("OSG Missing Project Report executed successfully")
    except Exception as e:
        ReportUtils.runerror(args.config, e, traceback.format_exc())
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()
