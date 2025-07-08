from twilio.rest import Client

# Fill in from Twilio Console
ACCOUNT_SID = 'Your Twilio Account SID'
AUTH_TOKEN = 'Auth Token generated on Twilio'
FROM_SMS = 'Your Twilio phone number'
TO_SMS = 'Destination number'
FROM_WHATSAPP = 'Twilio sandbox number'
TO_WHATSAPP = 'Your WhatsApp number'

client = Client(ACCOUNT_SID, AUTH_TOKEN)

def send_sms_alert(message):
    client.messages.create(
        body=message,
        from_=FROM_SMS,
        to=TO_SMS
    )

def send_whatsapp_alert(message):
    client.messages.create(
        body=message,
        from_=FROM_WHATSAPP,
        to=TO_WHATSAPP
    )
