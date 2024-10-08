import requests

url = 'http://localhost:5000/question'

# update the question for sending request through GUnicorn for API endpoint
question="A 31-year-old G2P2 female at 40 weeks gestation presents to the hospital following a rush of water that came from her vagina. She is 4 cm dilated and 80% effaced. Fetal heart tracing shows a pulse of 155/min with variable decelerations. About 12 hours after presentation, she gives birth to a 6 lb 15 oz baby boy with APGAR scores of 8 and 9 at 1 and 5 minutes, respectively. Which of the following structures is responsible for inhibition of female internal genitalia?"

data = {
        'question': question
        }

response = requests.post(url, json=data).json()

print(response)
