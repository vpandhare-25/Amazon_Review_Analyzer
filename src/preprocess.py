import re
def preprocess_text(text):
    text = text.lower() # Convert text to lowercase
    text = re.sub(r"<.*?>", "", text) # Remove HTML tags
    text = re.sub(r"http\S+|www\S+", "", text) # Remove links from text
    return text.strip()  # Strip remaining whitespace around text

print(preprocess_text("ML is <fun>"))