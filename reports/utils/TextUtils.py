"""This module provides static methods to create ascii, csv, and html attachment and send email to specified group of people. """

import sys
import datetime
import smtplib
from email.message import EmailMessage
import tabulate
import pandas as pd
from email.utils import formataddr



##########################################
# This code is partially taken from      #
# AccountingReports.py                   #
##########################################
class TextUtils:
    """Formats the text to create ascii, csv, and html attachment  and send email to specified group of people. """

    def __init__(self, table_header):
        """Args:
            table_header(list of str) - the header row for the output table
        """
        self.table_header = table_header

    def getWidth(self, l):
        """Returns max length of string in the list - needed for text formating of the table
            l(list of str)
        """

        return max(len(repr(s)) for s in l)

    def getLength(self, text):
        """Returns number of rows in the table
        Args:
            text(list)
        """

        return len(text[self.table_header[0]])

    def printAsTextTable(self, format_type, text, template=False):
        """"Prepares input text to send as attachment
        Args:
            format_type(str) - text, csv, html
            text (dict of lists) - {column_name:[values],column_name:[values]} where column_name corresponds to header name
                                   or pandas dataframe
        """

        if not isinstance(text, pd.DataFrame):
            # Convert list of dicts to pandas data frame
            df = pd.DataFrame.from_dict(text, orient='index').transpose()
            # Order the columns according to the header
            df = df[self.table_header]
        else:
            df = text

        # TODO: Remove this alignment code when python-tabulate recognizes
        # numbers with comma separators.
        alignment_list = ["left"] * (len(self.table_header) - 1)
        alignment_list.append("right")
        # the order is defined by header list
        if format_type == "text":
            return tabulate.tabulate(df, tablefmt="grid", headers=self.table_header,
                                     showindex=False, floatfmt=',.0f', colalign=alignment_list)
        elif format_type == "html":
            return tabulate.tabulate(df, tablefmt="html", headers=self.table_header,
                                     showindex=False, floatfmt=',.0f', colalign=alignment_list)
        elif format_type == "csv":
            return df.to_csv(index=False)


def sendEmail(toList, subject, content, fromEmail=None, smtpServerHost=None, smtpPort=None, smtpUser=None, smtpPassword=None, html_template=False):
    """
    This turns the "report" into an email attachment and sends it to the EmailTarget(s).
    Args:
    toList(list of str) - list of emails addresses
    content(str) - email content
    fromEmail (str) - from email address
    smtpServerHost(str) - smtpHost
    """

    #Charset.add_charset('utf-8', Charset.QP, Charset.QP, 'utf-8')

    if toList[1] is None:
        print("Cannot send mail (no To: specified)!", file=sys.stderr)
        sys.exit(1)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = formataddr(fromEmail)
    msg["To"] = _toStr(toList) 

    if html_template:
        msg.set_content(content["html"], subtype="html")
        msg.add_alternative(content.get("text", ""), subtype="plain")
        attachment_html = content["html"]
    else:
        msg.set_content("<pre>" + content["text"] + "</pre>", subtype="html")
        msg.add_alternative(content["text"], subtype="plain")
        attachment_html = "<html><head><title>%s</title></head><body>%s</body></html>" % (subject, content["html"])

    msg.add_attachment(attachment_html, filename="report_{}.html".format(datetime.datetime.now().strftime('%Y_%m_%d')))
    if "csv" in content:
        msg.add_attachment(content["csv"], filename="report_{}.csv".format(datetime.datetime.now().strftime('%Y_%m_%d')))

    msg = msg.as_string()

    if len(toList[1]) != 0:
        server = smtplib.SMTP_SSL(host=smtpServerHost, port=smtpPort)
        server.login(user=smtpUser, password=smtpPassword)
        server.sendmail(fromEmail[1], toList[1], msg)
        server.quit()
        print("Succesfully sent email")
    else:
        # The email list isn't valid, so we write it to stderr and hope
        # it reaches somebody who cares.
        print("Problem in sending email to: ", toList, file=sys.stderr)


def _toStr(toList):
    """Formats outgoing address list
    Args:
    toList(list of str) - email addresses
    """

    names = [formataddr(i) for i in zip(*toList)]
    return ', '.join(names)


if __name__ == "__main__":
    text = {}
    title = ["Time", "Hours", "AAAAAAAAAAAAAAA"]
    a = TextUtils(title)
    content = {"Time": ["aaa", "ccc", "bbb", "Total"],
               "Hours": [10000, 30, 300000, "", ],
               "AAAAAAAAAAAAAAA": ["", "", "", 10000000000]}
    text["text"] = a.printAsTextTable("text", content)
    text["csv"] = a.printAsTextTable("csv", content)
    text["html"] = a.printAsTextTable("html", content)
    text[
        "html"] = "<html><body><h2>%s</h2><table border=1>%s</table></body></html>" % (
    "aaaaa", a.printAsTextTable("html", content),)
    sendEmail((["Tanya Levshina", ], ["tlevshin@fnal.gov", ]), "balalala",
              text, ("Gratia Operation", "tlevshin@fnal.gov"), "smtp.fnal.gov")
