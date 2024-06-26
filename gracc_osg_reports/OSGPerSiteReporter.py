import traceback
import sys
import datetime
from collections import defaultdict

from opensearchpy import Search

from gracc_reporting import ReportUtils, TimeUtils

LOGFILE = 'osgpersitereport.log'
OPPORTUNISTIC_VOS = ['glow', 'gluex', 'hcc', 'osg', 'sbgrid'] # Default if not specified in config


# Helper Functions
def monthrange(date):
    """
    Takes a start date and finds out the start and end of the month that that
    input date is in

    :param datetime.datetime date:  Input date for which we find the start and end date
        of the month it's in
    :return tuple:datetime.datetime objects that span the month
    (2016-12-05 --> 2016-12-01, 2016-12-31)
    """
    try:
        assert isinstance(date, datetime.datetime) or \
            isinstance(date, datetime.date)
        startdate = date
    except AssertionError:
        raise TypeError("monthrange will only work with a"
                        " datetime.datetime object")

    start = startdate.replace(day=1)
    nextmonth_first = (startdate.replace(day=28) +
                       datetime.timedelta(days=4)).replace(day=1)
    end = nextmonth_first - datetime.timedelta(days=1)
    return start, end


def prev_month_shift(date):
    """
    Take the date passed in, go back a month, and return the range of that
    previous month
    (2016-12-05 --> 2016-11-01, 2016-11-30)

    :param datetime.datetime date: datetime.datetime object
    :return tuple: Consists of two datetime.datetime
    objects that span the range of the _previous_ month
    """
    sd = date.replace(day=1)-datetime.timedelta(days=1)
    return monthrange(sd)


def perc(num, den):
    """
    Converts fraction to percent

    :param float num: Numerator of fraction
    :param float den: Denominator of fraction
    :return float: Returns fraction converted to percentage
    """
    try:
        return float(num/den) * 100.
    except ZeroDivisionError:
        if num == 0:
            return 0.
        else:
            raise "Error: Tried to do division of {0}/{1}".format(num, den)


def perc_change(old, new):
    """Calculates the percentage change between two numbers

    :param float old: Baseline number
    :param float new: Changed number
    :return float: Percentage change from old to new
    """
    try:
        return perc(new - old, old)
    except:
        if new != 0:
            return 100.     # If we have something like (10-0) / 0, return 100%


class VO(object):
    """
    Class to hold VO information

    :param str voname: Name of VO
    """
    def __init__(self, voname):
        self.name = voname
        self.sites = {}
        self.sites = defaultdict(lambda: defaultdict(int))
        self.pos = 0
        self.total = defaultdict(int)
        self.totalcalc = self.running_totalhours()

    @staticmethod
    def __key_lookup(current):
        return 'current' if current else 'past'

    def add_site(self, sitename, corehours, current=True):
        """Add a new site data to the VO.  Also updates the core hours of
         sitename and the total for the VO

         :param str sitename: Name of site a VO's job set ran on
         :param float corehours: Core hours in the summary record
         :param bool current: Whether or not we're looking at current or past time period
         """
        key = self.__key_lookup(current)
        self.sites[sitename][key] += corehours
        self.totalcalc.send((corehours, key))
        return

    def get_cur_sitehours(self, sitename):
        """Get the current value of core hours for a particular sitename for
        the VO"""
        return self.sites[sitename]['current']

    def get_cur_totalhours(self):
        """Get the current value of total core hours for the VO"""
        return self.total['current']

    def get_old_sitehours(self, sitename):
        """Get the previous month's value of core hours for a particular
        sitename for the VO"""
        return self.sites[sitename]['past'] if sitename in self.sites else 0

    def get_old_totalhours(self):
        """Get the previous month's value of total core hours for the VO"""
        return self.total['past']

    @ReportUtils.coroutine
    def running_totalhours(self):
        """Keeps running total of core hours for the VO"""
        while True:
            newhrs, key = yield
            self.total[key] += newhrs


class OSGPerSiteReporter(ReportUtils.Reporter):
    """Class to store information and perform actions for the OSG Per Site
    Report

    :param str config_file: Report Configuration file
    :param str start: Start time of report range
    :param str end: End time of report range
    """
    def __init__(self, config_file, start, end, **kwargs):

        report = 'siteusage'

        super(OSGPerSiteReporter, self).__init__(report_type=report, 
                                                 config_file=config_file, 
                                                 start=start, 
                                                 end=end,
                                                 **kwargs)

        self.header = ["Site", "Total", "Opportunistic Total",
                       "Percent Opportunistic", "Prev. Month Opp. Total",
                       "Percentage Change Month-Month"]
        self.start_time, self.end_time = \
            monthrange(self.start_time)
        fmt = "%Y-%m-%d %H:%M"
        self.title = 'VOs Usage of OSG Sites: {0} - {1}'.format(
            self.start_time.strftime(fmt), self.end_time.strftime(fmt))
        self.opp_vos = self.__get_opportunistic_vos()
        self.current = True
        self.vodict = {}
        self.sitelist = []

    def __get_opportunistic_vos(self):
        """Private method to determine opportunistic VOs for the purposes of 
        this report
        
        :return: List of opportunistic VOs
        :rtype: list
        """
        return self.config[self.report_type].get('opportunistic_vos', 
            OPPORTUNISTIC_VOS)

    def run_report(self):
        """Method to run the OSG per site report"""
        self.generate()
        self.send_report()
        return

    def query(self):
        """Method to query Elasticsearch cluster for OSG Per Site Report
        information

        :return elasticsearch_dsl.Search: Search object containing ES query
        """
        starttimeq = self.start_time.isoformat()
        endtimeq = self.end_time.isoformat()

        if self.verbose:
            self.logger.info(self.indexpattern)

        s = Search(using=self.client, index=self.indexpattern) \
            .filter("range", EndTime={"gte": starttimeq, "lt": endtimeq}) \
            .filter('term', ResourceType="Batch") \
            .filter('term', Grid="OSG")[0:0]  # ignore 'Local' records

        # Note:  Using ?: operator in painless language to coalesce the
        # 'OIM_Site' and 'SiteName' fields.
        s.aggs.bucket('vo_bucket', 'terms', field='VOName', size=2**31-1) \
            .bucket('site_bucket', 'terms',
                    script={"inline": "doc['OIM_Site'].value ?: doc['SiteName'].value", "lang": "painless"},
                    size=2**31-1) \
            .metric('sum_core_hours', 'sum', field='CoreHours')

        return s

    def generate(self):
        """Higher-level method to run other methods to
        generate the raw data for the report."""
        consumer = self._create_vo_objects()

        # Run our query twice - once for this month, once for last month
        for self.start_time, self.end_time in (
                monthrange(self.start_time),
                prev_month_shift(self.start_time)):
            results = self.run_query()
            self._parse_results(results, consumer)
            self.current = False

        return

    @ReportUtils.coroutine
    def _create_vo_objects(self):
        """Coroutine to create the VO objects and store the information
        in them"""
        while True:
            vo, site, wallhrs = yield

            if not self.current \
                    and (vo not in self.vodict or site not in self.sitelist):
                continue
            elif self.current and vo not in self.vodict:
                    V = VO(vo)
                    self.vodict[vo] = V
            self.vodict[vo].add_site(site, wallhrs, current=self.current)

            if site not in self.sitelist:
                self.sitelist.append(site)

    @staticmethod
    def _parse_results(results, consumer):
        """Method that parses the result and passes the values to the
        consumer coroutine
        :param Response.aggregations results: ES Response object from ES query
        :param function consumer: Consumer function to pass data to
        """
        for vo_bucket in results.vo_bucket.buckets:
            vo = vo_bucket['key'].lower()
            for site_bucket in vo_bucket.site_bucket.buckets:
                site = site_bucket['key']
                wallhrs = site_bucket['sum_core_hours']['value']
                consumer.send((vo, site, wallhrs))
        return

    def format_report(self):
        """Report formatter.  Returns a dictionary called report containing the
        columns of the report.

        Note:  Each column here is a VO (or an aggregating column like Total)

        :return dict: Constructed dict of report information for
        Reporter.send_report to send report from
        """
        report = {}
        sitelist = sorted(self.sitelist)

        # Populate Site column
        report["Site"] = [site for site in sitelist]

        # Add VO Data to the report
        inspos = 2
        for vo, vo_object in sorted(self.vodict.items()):
            if vo not in self.header:
                if vo in self.opp_vos:
                    # Insert the opportunistic VOs into the header
                    # in order after the Site and Total columns
                    self.header.insert(inspos, vo)
                    inspos += 1
                else:
                    # Tack the other VO columns to the end
                    self.header.append(vo)

            report[vo] = [vo_object.get_cur_sitehours(site)
                          if site in vo_object.sites else 0
                          for site in sitelist]
            report[vo].append(vo_object.get_cur_totalhours())

        # Column for Previous Month Opportunistic Total
        report["Prev. Month Opp. Total"] = [
            sum((self.vodict[col].get_old_sitehours(site)
                 for col in report if col in self.opp_vos))
            for site in sitelist]
        stagecol = report["Prev. Month Opp. Total"]
        stagecol.append(sum(stagecol))  # Append the total for this column

        report["Site"].append("Total")  # Add "Total" line at bottom of report

        # This is the per-site total column, not the same "Total" as just above
        # Add all of the data for every vo; do this for each site
        report["Total"] = [sum((report[col][pos] for col in report
                                if col not in
                                ("Site", "Total", "Percentage Change Month-Month",
                                 "Prev. Month Opp. Total")))
                           for pos in range(len(report["Site"]))]

        # Do the same as above, but for only the opp. VOs in the report
        report["Opportunistic Total"] = [sum((report[col][pos]
                                              for col in report
                                              if col in self.opp_vos))
                                         for pos in range(len(report["Site"]))]

        # Calculate the percent opportunistic usage from the above two columns
        report["Percent Opportunistic"] = [perc(report["Opportunistic Total"][pos],
                                                report["Total"][pos])
                                           for pos in
                                           range(len(report["Site"]))]

        # Percent Change Month-Month for opportunistic VOs
        report["Percentage Change Month-Month"] = [
            perc_change(report["Prev. Month Opp. Total"][pos],
                        report["Opportunistic Total"][pos])
            for pos in range(len(report["Site"]))]

        # In any modifications, the order of the next four sections must be
        # preserved.  They all have to do with handling the previous month's
        # data

        # Previous month totals and percent change by VO
        for col, values in report.items():
            if col == "Site":
                values.append("Prev. Month Total")
                values.append("Percent Change over Prev. Month")
            elif col in self.vodict:
                values.append(self.vodict[col].get_old_totalhours())    # Total
                values.append(perc_change(values[-1], values[-2]))    # Percent change

        # Handle opportunistic total column for the previous month
        stagecol = report['Opportunistic Total']
        stagecol.append(sum((report[vo][-2] for vo in report
                             if vo in self.opp_vos)))
        stagecol.append(perc_change(stagecol[-1], stagecol[-2]))

        # Handle total column for the previous month
        stagecol = report['Total']
        stagecol.append(sum((report[vo][-2] for vo in self.vodict)))
        stagecol.append(perc_change(stagecol[-1], stagecol[-2]))

        # Handle percent opportunistic overall for previous month
        report['Percent Opportunistic'].extend(
            ((report["Opportunistic Total"][-2] /
              report["Total"][-2] * 100), 'N/A'))

        # Fill in the rest of the report
        report["Prev. Month Opp. Total"].extend(('N/A', 'N/A'))
        report["Percentage Change Month-Month"].extend(('N/A', 'N/A'))

        # Insert a blank line
        for values in report.values():
            values.insert(-3, '')

        return report


def main():
    args = ReportUtils.get_report_parser().parse_args()
    logfile_fname = args.logfile if args.logfile is not None else LOGFILE

    if args.end is not None:
        try:
            assert [d.date() for d in 
                monthrange(TimeUtils.parse_datetime(args.start))] ==\
                [d.date() for d in 
                monthrange(TimeUtils.parse_datetime(args.end))]
        except AssertionError:
            raise ValueError("Currently, the OSG Per Site Reporter only supports running over"\
                " a month time range (e.g. 2017-03-01 to 2017-03-31).  Please"\
                " either specify -s <start> -e <end> with both <start> and <end>"\
                " within the same month, or only specify a start time with the -s"\
                " flag.")

    start, end = monthrange(TimeUtils.parse_datetime(args.start))

    try:
        osgreport = OSGPerSiteReporter(config_file=args.config,
                                       start=start,
                                       end=end,
                                       template=args.template,
                                       verbose=args.verbose,
                                       is_test=args.is_test,
                                       no_email=args.no_email,
                                       logfile=logfile_fname)

        osgreport.run_report()
        print('OSG Per Site Report Execution finished')
        sys.exit(0)
    except Exception as e:
        ReportUtils.runerror(args.config, e, traceback.format_exc(), logfile_fname)
        sys.exit(1)


if __name__ == '__main__':
    main()
