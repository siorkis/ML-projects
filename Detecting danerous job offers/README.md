# Neuron Network developed for "Hackathon Against Human Trafficking" 2023

The idea of the following network is to analyze job offers and predict how potentially dangerous it is.

## Metrics

- ### Vague Job Description: 
Legitimate job listings usually have clear, detailed descriptions of the role, responsibilities, and qualifications. If the description is vague or overly simple, it might be a red flag.

- ### Unrealistic Salaries or Benefits: 
Offers that promise unusually high pay for minimal work or basic tasks are often too good to be true.

- ### Requests for Personal Information: 
Be wary of job offers that ask for personal or financial information (like your social security number or bank details) early in the process or via unsecure methods.

- ### Upfront Payment Requests: 
Any job that requires you to pay upfront fees (for training, supplies, or other reasons) is likely a scam.

- ### Unprofessional Communication: 
Look out for poor grammar, spelling mistakes, and unprofessional email addresses in communication, as these can be indicators of fraudulent offers.

- ### Too Good to Be True: 
Trust your instincts. If a job offer seems too perfect or easy, it's worth investigating further.

- ### No Interview Process: 
Legitimate jobs usually have some form of an interview or vetting process. Offers made without an interview or with only a brief, informal chat can be suspicious.

- ### High Pressure Tactics: 
Scammers often try to pressure you into making fast decisions, claiming that the opportunity will disappear quickly.

- ### Lack of Online Presence: 
Research the company online. If they have no website, social media presence, or reviews, it could be a scam.

- ### Unsolicited Offers: 
Be cautious of job offers that come from unsolicited emails or messages, especially if you didn't apply for the role.

## Training data
Training data was generated according to the metrics (dangerous and safe job offers).

## Model architecture

Model takes 1 input feature, which is text (job description) and output prediction (0 or 1).

Training data - 80% of generated job offers (with labels)

Test data - 20% of generated job offers (without labels)

Validation data - test data with right labels 

### Input(Text Data) -> [Embedding Layer] -> [LSTM Layer] -> [Dense Layer with Sigmoid Activation] -> Output(Probability)


#### Embedding Layer: 
Transforms the input text data (sequence of integers representing words) into dense vectors of fixed size (32 in this case). This layer is crucial for handling text data, as it allows the model to learn meaningful representations of words in a lower-dimensional space.
#### LSTM Layer: 
Processes the sequences of word embeddings. It's designed to learn dependencies in sequence data, making it effective for tasks involving text.
#### Dense Layer with Sigmoid Activation: 
The final layer that outputs the probability that the input text belongs to a certain class (in your case, whether a job offer is dangerous/scam or not).

## Model performance 
![Alt text](image.png)

During training was reached ~95% accuracy. 

![Alt text](image-1.png)

Testing model on the actual data from LinkedIn (human prediction - this is safe offer)

![Alt text](image-2.png)

Testing model on the actual data from 999 (job listing website) (human prediction - this is dangerous offer)

