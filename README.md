gracc-osg-reports Developer Documentation
================

These are the developer docs for _gracc-osg-reports_.  They're by no means complete, but I've attempted to 
note the important info for anyone coming in and trying to write a report.

_gracc-osg-reports_ is a set of reports and a couple of helper modules built on top of [_gracc-reporting_](https://github.com/opensciencegrid/gracc-reporting),
that query [GRACC](https://gracc.opensciencegrid.org/) elasticsearch, aggregate and process the results
into a report, and email those to stakeholders.

Currently, the _gracc-osg-reports_ package assume that queries to the GRACC instance will be made using
Honza Kral's [elasticsearch-dsl-py](https://github.com/elastic/elasticsearch-dsl-py) library, and 
indeed, this is a dependency of this package.  Any report queries should be made using this API.  However, this may change in the 
future, since the ultimate aim is to provide flexibility for future report writers to simply 
send an HTTP POST or GET request to the GRACC instance, or use another API of their choosing.

.
# Parts of a gracc-osg report 

There are two main ways that the reports are structured.  The first uses the gracc-reporting library's 
text-processing utilities to automatically generate the report from the organized data, and should be used for 
all simpler reports (and whenever possible).  An example of this is the Flocking report.  The second method is to 
do all of the text manipulation and populate the HTML template manually.  An example of this is the Top Opportunistic Usage by Facility (news) report.

For both methods, you'll need to create a class for the report that subclasses ReportUtils.Reporter from gracc_reporting, and pass its required parameters into __init__ (config, start, end).
 This class will have to implement the following methods:

## run_report(self)


This is the higher-level method that simply calls all/any of the other methods.  In its simplest form, 
you can get away with just calling "send_report" (see below) from there.

## query(self)

This is where you define the elasticsearch query and the necessary aggregations.  To work with the run_query() method from ReportUtils.Reporter, query() should return the elasticsearch_dsl.Search() object after defining the query and aggregations.

## generate(self)

This is the higher level method that calls ReportUtils.Reporter.run_query(), goes through the results, and 
generates the raw data for the report.  Usually, it's a good idea to have this method yield rows of data 
for some other method to parse or act on.

## format_report(self)

This method is not required to be defined. Best practice, however, is to define either this or generate_report_file() (see below).
This method should take the data after parsing, and organize it into a dictionary of lists, with each list 
representing a column.  In that case, self.header (a list) should list the columns in order from left-to-right.  ReportUtils.Reporter.send_report will then insert this into the HTML template, 
a CSV file, and plaintext and send it.  Again, for an example, please see the Flocking Report (OsgFlockingReporter.py).

## generate_report_file(self)

This is the standard name used for a method that, instead of format_report, will populate the self.text 
attribute with the HTML that will be sent.  If you use this method, you should read in the HTML template, 
insert the text as needed along with any HTML, and save that HTML text to self.text.  ReportUtils.Reporter.send_report() 
will use this and send an HTML report.  An example that uses this method is the Top Opportunistic Usage by Facility Report (TopOppUsageByFacility.py).



# Putting it together into a report

The easiest way, then to write a report from these is to simply instantiate the report class from main() and then to call run_report() from that instantiation.  Without error handling, that might look like this:

```python
def main():
    CONFIG = "my_config.toml"
    m = MyReport(config_file=CONFIG, 
             start="2018-09-01", 
         end="2018-09-08", 
         template="/path/to/template_file")
    m.run_report()
    print "Yay, it worked!"
```

Of course, it's a better idea to actually do error checking.  To help with that, Reporter.ReportUtils has a 
function runerror(config_file, exception, traceback, logfile), that can be used to email the admins 
configured in your configuration file (see below) in case of an error.  Using this, you'd pass any exception up 
through the calls until you got here, and then run that function on the error inside an except clause.



## Helper methods in Reporter.ReportUtils


### runerror

Function for handling errors during execution of report.  Ideally, all errors are passed to the top 
level of the report, which then has _runerror_ in an *except* clause.  _runerror_ will log the error, 
the traceback, and email the admins (test.emails).

### coroutine

A helper decorator that advances a coroutine to its first yield point.  Adapted from
http://www.dabeaz.com/coroutines/Coroutines.pdf

### get_report_parser

Creates a parser for evaluating command-line options.  Can be called with time options (start, end) by 
default, or without by calling get_report_parser(no_time_options=True).  




# Configuration

The configuration file that all _gracc_reporting_-based packages expect is a toml file.  It should, at the minimum, have
the following:

```toml
 [elasticsearch]
    hostname = 'https://gracc.opensciencegrid.org/q'

# Email
# Set the global email related values under this section
[email]
    # This is the FQDN of the mail server, which GRACC will use to send the email
    smtphost = 'smtp.example.com'

    [email.from]
        name = 'GRACC Operations'  # This is the real name from which the report appears to be emailed from
        email = 'somebody@somewhere.com'  # This is the email from which the reports appears to be emailed from

    # Tester emails
    [email.test]
        names = ['Test Recipient', ]
    emails = ['admin@somwhere.com', ]

# Report-specific
[report_name]
    index_pattern='some.index.pattern'
    to_emails = ['nobody@example.com', ]
    to_names = ['Recipient Name', ]
 ```

 If using the Report.get_report_parser, the command-line flag to specify a config file is -c.  There is a sample toml file included in the repository.


# How to build gracc-osg-reports

Building _gracc-osg-reports_ is quite simple.  To build the package, first make any necessary changes in 
setup.py in the root of the repository.  Then, simply run 
```
python setup.py sdist
```
You can then unwind the tarball created in the _dist_ directory wherever you want to deploy this, and run 
```
python setup.py install
```
Make sure _gracc-reporting_ is installed before you try to install _gracc-osg-reports_.

For an OSG installation, for which we use a docker container to build reports on, navigate to 
_packaging/_ and run _create_docker_image.sh_.  Keep in mind that you must have write access to the Open 
Science Grid docker hub gracc-reporting repository in order to run this script.


# gracc-reporting Developer Documentation
================

These are the developer docs for _gracc-reporting_.  They're by no means complete, but I've attempted to 
note the important info for anyone coming in and trying to write a report.

_gracc-reporting_ consists of a number of libraries meant to be used as helpers on top of which reports
can be built.  The current format of reports (and thus of _gracc-reporting_) is very simply:

1. Query the elasticsearch query endpoint of the GRACC instance
2. Parse those results in a useful manner for your raw data
3. Aggregate the data into a reportable format 
4. Populate the report template (an HTML template)
5. Send the report

Currently, the _gracc-reporting_ libraries assume that queries to the GRACC instance will be made using
Honza Kral's [elasticsearch-dsl-py](https://github.com/elastic/elasticsearch-dsl-py) library, and 
indeed, this is a dependency of the _gracc-reporting_ package.  However, this may change in the 
future, since the ultimate aim is to provide flexibility for future report writers to simply 
send an HTTP POST or GET request to the GRACC instance, or use another API of their choosing.

I'll address each module and what it does first, and then discuss building the package and writing
a report on top of _gracc-reporting_.

# Modules

## ReportUtils.py

This is the main module in _gracc-reporting_.  It exports the [Reporter class](#class-reporter) and a 
few helper functions.


### class Reporter

The Reporter class is the class all gracc-reports are built on.  Instantiating the class, one must
provide:

* report_type (string):  Which report is being run. The name given here must match whatever name you
use in the configuration file.  Gets added as self.report\_type
* config_file (string):  The path to the configuration file. Gets added as dict self.config after parsing
* start and end (string):  Start and end times of report range (given in local TZ).  These can be in 
format that [dateutil.parser](https://dateutil.readthedocs.io/en/stable/parser.html) understands 
(e.g. YYYY-MM-dd HH:mm:ss).  Added as self.start_time and self.end_time (datetime.datetime 
objects) after parsing

You can also provide other optional keyword arguments.  These are the supported ones, along with their 
defaults (any others will result in an error, though that might change):

* althost_key (None): key to look for in config file to designate the elasticsearch host.  If this is 
None, Reporter will assume that the key to look for is 'host'.
* index_key ('index_pattern'): key to look for in config file for the elasticsearch index pattern to
use.  If this is not given, Reporter will look for 'index\_pattern' to be set in the config file.  If
that is not configured, it will simply use gracc.osg.summary.
* vo (None):  Virtual organization to run report on.  Doesn't apply to most OSG reports
* template (None):  HTML template file to use
* is_test (False):  Test mode or not (If True, will send emails only to admins as set in config file 
(test.emails and test.names)
* no_email (False): Don't send any emails at all.  Just run the report
* verbose (False)

These are the main methods of the Reporter class.

#### query

Must be overridden. Define the elasticsearch query using elasticsearch_dsl, return the elasticsearch_dsl 
Search object.

#### run_query:   
Execute the query and check the status code before returning the relevant info (as either a Search 
object to run the scan/scroll API on, or an aggregations object if that's what the query requested).

#### generate_report_file or format_report:

Pick one!  

Use generate_report_file if you're building the data structure yourself (most complex reports need 
this).  Populate _self.text_ with the final HTML.  

Use format_report if you're fine with creating a report as a dict of the columns, that can be used for 
simultaneous CSV and HTML generation.  format_report should return this dict, and send_report will 
handle the HTML/CSV generation in that case.

#### send_report

Will email the report produced by either of the previous methods.  Checks if self.format_report returns
anything.  If not, send_report assumes self.text is populated (presumably by self.generate_report_file), 
and will send that as the HTML report.  Otherwise, it will use the dict returned by self.format_report
and generate the HTML and CSV files, and send those.


#### run_report

Must be overridden. Use it to run all of the above and any other helper methods/functions involved
in the generation of the report.


#### Helper methods

* Reporter.indexpattern_generate will grab the index pattern from the configuration file and will 
try to use IndexPattern.indexpattern\_generate to create a more specific index pattern to optimize 
query speed.
* check_no_email will look at the self.no_email flag, and if it's set, logs some info.
* __establish_client is a hidden method, but I wanted to mention it because it is where the connection
to the GRACC host is established.  It is not meant to be used in any reports.


### runerror

Function for handling errors during execution of report.  Ideally, all errors are passed to the top 
level of the report, which then has _runerror_ in an *except* clause.  _runerror_ will log the error, 
the traceback, and email the admins (test.emails).

### coroutine

A helper decorator that advances a coroutine to its first yield point.  Adapted from
http://www.dabeaz.com/coroutines/Coroutines.pdf

### get_report_parser

Creates a parser for evaluating command-line options.  Can be called with time options (start, end) by 
default, or without by calling get_report_parser(no_time_options=True).  



## TimeUtils.py

TimeUtils is a library of helper functions, built heavily on datetime,
time, and dateutil, to help with the conversions of timestamps in gracc-
reporting.  Note that in this module, parse_datetime is the only function
that can accept non-UTC timestamps.  epoch_to_datetime assumes you're giving it an epoch time, and 
returns a UTC datetime, and get_epoch_time_range_utc assumes both start_time and end_time are 
UTC datetime objects.

## IndexPattern.py

Generates gracc-reporting index patterns.  indexpattern_generate accepts a pattern that can be 
processed by datetime.strftime, and given the time range of the report (also passed to the function),
it tries to narrow down the index pattern to specific indices.  For example, if the pattern given is
_gracc.osg.raw-%Y.%m_ and the start and end dates are 2018-07-01 and 2018-07-02, indexpattern_generate 
will return _gracc.osg.raw-2018.07*_.  If the dates are 2018-07-01 and 2018-08-01, indexpattern_generate 
will return _gracc.osg.raw-2018*_.  Without such filtering, we'd be searching gracc.osg.raw-* in these
examples.

## TextUtils.py

This module provides static methods to create ascii, csv, and html attachment and send email to 
specified group of people.  It's not been touched in a very long time, and eventually, it should be 
reviewed and possibly improved.

## NiceNum.py

Returns a nicely formatted string for the floating point number
provided.  This number will be rounded to the supplied accuracy
and commas and spaces will be added.  Use the niceNum function from this module for formatting tables,
especially when Reporter.generate_report_file is used. 


# Configuration

The configuration file that _gracc_reporting_ expects is a toml file.  It should, at the minimum, have
the following:

```toml
 [elasticsearch]
    hostname = 'https://gracc.opensciencegrid.org/q'

# Email
# Set the global email related values under this section
[email]
    # This is the FQDN of the mail server, which GRACC will use to send the email
    smtphost = 'smtp.example.com'

    [email.from]
        name = 'GRACC Operations'  # This is the real name from which the report appears to be emailed from
        email = 'somebody@somewhere.com'  # This is the email from which the reports appears to be emailed from

    # Tester emails
    [email.test]
        names = ['Test Recipient', ]
    emails = ['admin@somwhere.com', ]

# Report-specific
[report_name]
    index_pattern='some.index.pattern'
    to_emails = ['nobody@example.com', ]
    to_names = ['Recipient Name', ]
 ```

 If using the Report.get_report_parser, the command-line flag to specify a config file is -c.



# How to build gracc_reporting

Building _gracc_reporting_ is quite simple.  To build the package, first make any necessary changes in 
setup.py in the root of the repository.  Then, simply run 
```
python setup.py sdist
```
If you want to simply  write reports on top of gracc_reporting, you can take the tarball created in the
_dist_ directory, move it where you need, untar it, and run 
```
python setup.py install
```
or simply run the last two commands in order to install the gracc-reporting package in place.

For an OSG installation, for which we use a docker container to build reports on, navigate to 
_packaging/_ and run _create_docker_image.sh_.  Keep in mind that you must have write access to the Open 
Science Grid docker hub gracc-reporting repository in order to run this script.

An example report building directly on top of _gracc_reporting_ along with its sample config file is 
provided in the _example_ directory here.  You can run the report by doing:
```
python SampleReport.py -c sample.toml -d -v -s '2018-08-20 00:00' -e '2018-08-21 00:00'
```
