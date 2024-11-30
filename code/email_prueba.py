import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os

def send_email(subject, body, attachment_file=None, is_admin=False):
    sender = "holaquetal@upv.es" # Short name of the server (if server is cvblab01.htech.upv.es --> cvblab01)
    recipient = "clferma1@upvnet.upv.es"

    # Create the message
    msg = MIMEMultipart('alternative') # To indicate that message is encoded in HTML
    msg['From'] = sender
    msg['To'] = recipient
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'html'))  # Attach the message body (in HTML format)

    # Attach the log file (only when job has finished or has been cancelled)
    if attachment_file:
        with open(attachment_file, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())  # Read the file content
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", "attachment; filename=LogFile_1.html")
            msg.attach(part)

    # Send the email
    server = smtplib.SMTP(host="smtp.upv.es", port=25)
    server.sendmail(sender, recipient, msg.as_string())

if __name__ == "__main__":
    send_email("Mensaje de prueba","HOLA CLAUDIO")

    print("hola")
